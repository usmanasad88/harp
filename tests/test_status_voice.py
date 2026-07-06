"""StatusVoice player: manifest resolution, fail-safe behavior, serialization.

No audio hardware — the sink (what actually plays a clip) is faked, so these
cover the resolution/robustness logic, not sounddevice itself. The guarantee
under test is that a status announcement is best-effort: a bad id, a missing
file, a broken manifest, or a dead audio device is always a logged no-op,
never an exception that could take the supervisor down.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

from harp.orchestrator.status_voice import StatusVoice


def _manifest(tmp_path, lines: dict) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"lines": lines}), encoding="utf-8")


def _clip(tmp_path, rel: str) -> str:
    """Create a stand-in clip file (contents irrelevant — the sink is faked)."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"RIFF")
    return rel


async def test_play_resolves_id_to_clip_and_calls_sink(tmp_path):
    played: list = []
    _manifest(tmp_path, {"starting_up": {"en": {"file": _clip(tmp_path, "en/starting_up.wav")}}})
    await StatusVoice(tmp_path, sink=played.append).play("starting_up")
    assert played == [tmp_path / "en" / "starting_up.wav"]


async def test_unknown_id_is_a_noop(tmp_path):
    played: list = []
    _manifest(tmp_path, {"starting_up": {"en": {"file": _clip(tmp_path, "en/starting_up.wav")}}})
    await StatusVoice(tmp_path, sink=played.append).play("does_not_exist")
    assert played == []  # no clip resolved, and crucially no crash


async def test_id_present_but_file_missing_is_a_noop(tmp_path):
    played: list = []
    _manifest(tmp_path, {"ghost": {"en": {"file": "en/ghost.wav"}}})  # file never created
    await StatusVoice(tmp_path, sink=played.append).play("ghost")
    assert played == []


async def test_sink_failure_is_swallowed(tmp_path):
    _manifest(tmp_path, {"boom": {"en": {"file": _clip(tmp_path, "en/boom.wav")}}})

    def boom(_path):
        raise RuntimeError("no audio device")

    await StatusVoice(tmp_path, sink=boom).play("boom")  # must not raise


async def test_lang_falls_back_to_english(tmp_path):
    played: list = []
    _manifest(tmp_path, {"ready": {"en": {"file": _clip(tmp_path, "en/ready.wav")}}})
    # Configured for Urdu, but only an English clip exists → speak English.
    await StatusVoice(tmp_path, lang="ur", sink=played.append).play("ready")
    assert played == [tmp_path / "en" / "ready.wav"]


async def test_missing_manifest_is_all_noop(tmp_path):
    played: list = []
    await StatusVoice(tmp_path, sink=played.append).play("starting_up")  # no manifest.json
    assert played == []


async def test_clips_do_not_overlap(tmp_path):
    """The internal lock serializes playback — a second play() waits for the
    first to finish, so two quick transitions never talk over each other."""
    _manifest(tmp_path, {
        "a": {"en": {"file": _clip(tmp_path, "en/a.wav")}},
        "b": {"en": {"file": _clip(tmp_path, "en/b.wav")}},
    })
    lock = threading.Lock()
    live = {"now": 0, "max": 0}

    def sink(_path):  # runs in a worker thread (asyncio.to_thread)
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        time.sleep(0.05)
        with lock:
            live["now"] -= 1

    sv = StatusVoice(tmp_path, sink=sink)
    await asyncio.gather(sv.play("a"), sv.play("b"))
    assert live["max"] == 1  # never both clips playing at once
