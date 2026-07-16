"""Offline push-to-talk gate recorder — hear exactly what the model would.

Debugging the push-to-talk mic gate against the live API burns tokens on every
attempt. This spike runs the SAME production audio path with no provider
connection and writes the result to WAV files instead:

    mic -> gated_mic_payload(ptt gate) -> LoudnessGate -> file (not the API)

It reuses the real components, not copies: `PushToTalk` (with its real pynput
key listener, combo parsing from harp.yaml, and edge-triggered press/release),
`Microphone`, `gated_mic_payload`, and `LoudnessGate` — the identical chain
`VoiceBridge._pump_mic` sends to the provider. So what lands in the WAV is
byte-identical to what a live session would have streamed.

The recorder runs continuously, like a live gated session: while the key is up
it records digital silence, while the key is held it records real mic audio.
Releasing the key saves one WAV (the leading silence + your speech) to
.harp/ptt_test/ and starts the next one. Ctrl+C to quit.

Run:
    uv run python spike_ptt_gate.py [--provider gemini|openai]

Then open the WAVs: they should be flat silence up to your press, clean speech
while held, and nothing after release. The PushToTalk instance is created in
exclusive mode, which makes `mic_open` track the key directly — the same gate
behavior a press-started (or exclusive-mode) live session has.

The key combo, `release_debounce_seconds` (for a hardware button that re-taps
the combo instead of holding it — see the PushToTalk module docstring), and
`voice_tuning.near_field_level` come from harp.yaml, exactly as in app.py; the
sample rate comes from the provider's session config, so the files compare 1:1
with that provider's live input. `--key`, `--debounce`, and `--debug-keys`
override/diagnose without touching harp.yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import wave
from pathlib import Path

from harp.config import REPO_ROOT, build_session_config, load_settings
from harp.core.bus import Bus
from harp.interaction.push_to_talk import PushToTalk
from harp.listener.detector import rms_level
from harp.voice.audio_io import Microphone
from harp.voice.bridge import gated_mic_payload
from harp.voice.loudness_gate import LoudnessGate

OUT_DIR = REPO_ROOT / ".harp" / "ptt_test"


def start_key_debugger():
    """A second, raw pynput listener that dumps every key event the OS delivers
    (its own hook, alongside PushToTalk's). Shows both the raw key and the
    canonical form the production combo handler matches against — so a hold
    that reaches PushToTalk as press/release oscillation is visible at the
    source: OS auto-repeat, injected events from a remapper/hotkey app, etc."""
    from pynput import keyboard

    t0 = time.monotonic()

    def dump(kind: str, key) -> None:
        vk = getattr(key, "vk", None)
        print(
            f"\n[{time.monotonic() - t0:7.3f}s] {kind:<7} "
            f"raw={key!r} vk={vk} canonical={listener.canonical(key)!r}",
            flush=True,
        )

    listener = keyboard.Listener(
        on_press=lambda k: dump("press", k),
        on_release=lambda k: dump("RELEASE", k),
    )
    listener.start()
    return listener


def save_wav(path: Path, pcm: bytes, rate: int) -> None:
    """Write raw 16-bit mono PCM — the exact bytes the gate produced — as WAV."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


async def record(ptt: PushToTalk, rate: int, near_field_level: float, out_dir: Path) -> None:
    """The offline stand-in for VoiceBridge._pump_mic: same two gates in series,
    same chunk source — but each release-edge flushes the stream to a WAV
    instead of the chunks going to conn.send_audio()."""
    gate = LoudnessGate(lambda: near_field_level)
    ptt_gate = lambda: ptt.mic_open
    buf = bytearray()
    was_held = False
    held_chunks = 0
    peak = 0.0
    async with Microphone(rate) as mic:
        print(f"Recording at {rate} Hz. Hold the key to talk; release to save; Ctrl+C to quit.")
        async for pcm in mic.chunks():
            payload = gate.process(gated_mic_payload(pcm, ptt_gate))
            buf += payload
            held = ptt.held
            if held:
                level = rms_level(payload)
                peak = max(peak, level)
                held_chunks += 1
                print(f"\r  mic OPEN  level={level:.3f} (peak {peak:.3f})   ", end="", flush=True)
            if was_held and not held:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / time.strftime("ptt-%Y%m%d-%H%M%S.wav")
                save_wav(path, bytes(buf), rate)
                total_s = len(buf) / 2 / rate
                held_s = held_chunks * (len(pcm) / 2) / rate
                print(
                    f"\r  saved {path.relative_to(REPO_ROOT)}: {total_s:.1f}s total, "
                    f"{held_s:.1f}s held, peak level {peak:.3f}"
                    + ("  <- all-silent while held; check the mic!" if peak == 0.0 else "")
                )
                buf.clear()
                held_chunks = 0
                peak = 0.0
            was_held = held


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider", choices=("gemini", "openai"), default="gemini",
        help="whose session config sets the mic sample rate (default: gemini)",
    )
    parser.add_argument(
        "--key", default=None,
        help="override harp.yaml's push_to_talk.key (e.g. --key space) to test "
        "whether a problem is specific to one key/combo",
    )
    parser.add_argument(
        "--debug-keys", action="store_true",
        help="print every raw key event the OS delivers (press/release, raw + "
        "canonical form) to diagnose combo detection",
    )
    parser.add_argument(
        "--debounce", type=float, default=None, metavar="SECONDS",
        help="override harp.yaml's push_to_talk.release_debounce_seconds (for a "
        "button that re-taps the combo instead of holding it)",
    )
    args = parser.parse_args()

    # Show the production log lines — 'push-to-talk armed', or the exception
    # if the key listener can't start (permissions / no display).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    rate = build_session_config(args.provider).input_rate
    level = settings.voice_tuning.near_field_level
    key = args.key or settings.push_to_talk.key
    debounce = (
        args.debounce if args.debounce is not None
        else settings.push_to_talk.release_debounce_seconds
    )
    print(
        f"push_to_talk.key = {key!r}"
        + (" (--key override)" if args.key else " (from harp.yaml)")
        + f", release_debounce_seconds = {debounce}"
        + (" (--debounce override)" if args.debounce is not None else "")
        + f", voice_tuning.near_field_level = {level}"
    )
    if args.debug_keys:
        start_key_debugger()

    # Exclusive mode: mic_open == key held, with no orchestrator/state machine
    # needed — the same per-chunk gate a press-started live session applies.
    ptt = PushToTalk(Bus(), key=key, exclusive=True, release_debounce_seconds=debounce)
    ptt_task = asyncio.create_task(ptt.run())
    record_task = asyncio.create_task(record(ptt, rate, level, OUT_DIR))
    # ptt.run() only returns early if the key listener failed to start; the
    # recorder would then capture silence forever, so stop instead of misleading.
    done, _ = await asyncio.wait(
        (ptt_task, record_task), return_when=asyncio.FIRST_COMPLETED
    )
    for task in done:
        task.result()  # surface any crash as a traceback
    if ptt_task in done:
        record_task.cancel()
        sys.exit("push-to-talk listener did not start — see the log line above.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")
