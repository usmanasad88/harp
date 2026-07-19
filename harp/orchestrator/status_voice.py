"""Pre-programmed status voice — speech that works WITHOUT a live model.

Boot lines, connectivity notices, and error phrases ("Starting up", "Connection
established", "No internet", "Going on standby") play from local audio so HARP
can speak during startup, outages, and cloud failures — exactly the moments the
cloud voice is unavailable.

The clips + a manifest are generated offline (assets/status_voice/, see
scripts/generate_status_voice.py). Callers reference a stable id, e.g.
`play("starting_up")`, never a raw path, so the backing clips can change freely.
WHICH id plays at which life-cycle moment is not decided here or by callers ad
hoc — that policy is the rule book, orchestrator/status_rules.py.

Design notes:
  - Playback is SERIALIZED (one clip at a time) via an asyncio lock, so two
    quick transitions don't talk over each other.
  - Every failure is swallowed with a log line — a missing clip, a broken
    manifest, or a machine with no audio output must NEVER take the supervisor
    down. A status announcement is best-effort by definition.
  - The audio backend is injected (`sink`) so the player is unit-tested without
    a sound card; the default sink lazily imports sounddevice/numpy so importing
    this module needs no audio stack.
"""

from __future__ import annotations

import asyncio
import json
import logging
import wave
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _play_wav(path: Path) -> None:
    """Blocking one-shot playback of a WAV clip on the default output device.
    Imported lazily so merely importing this module never needs an audio stack
    (tests inject a fake sink instead)."""
    import numpy as np
    import sounddevice as sd

    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    sd.play(audio, samplerate=rate)
    sd.wait()


class StatusVoice:
    """Plays canned status clips by stable id, resolved through the manifest."""

    def __init__(
        self,
        assets_dir: Path,
        *,
        lang: str = "en",
        sink: Callable[[Path], None] = _play_wav,
    ) -> None:
        self._dir = Path(assets_dir)
        self._lang = lang
        self._sink = sink
        self._lock = asyncio.Lock()
        self._lines = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Read the id → lang → clip map once. A missing/corrupt manifest just
        means every play() is a logged no-op (never a crash)."""
        try:
            data = json.loads((self._dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("status_voice: no usable manifest in %s (%s)", self._dir, exc)
            return {}
        lines = data.get("lines")
        return lines if isinstance(lines, dict) else {}

    def _resolve(self, line_id: str, lang: str) -> Path | None:
        """id + lang → an existing clip path, or None. Falls back to the default
        language then English so a partially-translated manifest still speaks."""
        entry = self._lines.get(line_id)
        if not isinstance(entry, dict):
            return None
        clip = entry.get(lang) or entry.get(self._lang) or entry.get("en")
        if not isinstance(clip, dict) or "file" not in clip:
            return None
        path = self._dir / clip["file"]
        return path if path.exists() else None

    async def play(self, line_id: str, lang: str | None = None) -> None:
        """Play one status clip and return once it finishes. An unknown id, a
        missing file, or a dead audio device are all no-ops (logged), never
        exceptions — callers can `await` this on any code path without guarding."""
        path = self._resolve(line_id, lang or self._lang)
        if path is None:
            logger.warning("status_voice: no clip for %r (lang=%s)", line_id, lang or self._lang)
            return
        async with self._lock:  # never let two clips overlap
            try:
                await asyncio.to_thread(self._sink, path)
            except Exception:
                logger.exception("status_voice: playback failed for %s", line_id)


def play(line_id: str, lang: str = "en") -> None:  # pragma: no cover - convenience shim
    """One-shot blocking play, no orchestration. Handy from scripts / a REPL;
    the running agent uses a long-lived StatusVoice (serialized, fail-safe)."""
    from ..config import STATUS_VOICE_DIR

    asyncio.run(StatusVoice(STATUS_VOICE_DIR, lang=lang).play(line_id))
