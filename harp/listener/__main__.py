"""Calibration + manual test for the wake listener.

    uv run python -m harp.listener

Shows your live mic level against the two thresholds from harp.yaml, and runs
the real detector + Whisper transcriber so you can see exactly what would wake
HARP. Adjust `wake_level` / `transcribe_level` in harp.yaml until quiet room ≈
below both, normal speech ≈ above transcribe_level, and a raised voice/clap ≈
above wake_level. Ctrl+C to quit.
"""

from __future__ import annotations

import asyncio
import sys

from ..config import load_settings
from ..voice.audio_io import Microphone
from .detector import LoudSound, Phrase, WakeDetector, match_wake_word, rms_level
from .transcriber import PhraseTranscriber

_BAR_WIDTH = 50
_BAR_SCALE = 0.5  # full bar = this RMS level; speech rarely exceeds it


def _bar(level: float, transcribe_level: float, wake_level: float) -> str:
    filled = min(_BAR_WIDTH, int(level / _BAR_SCALE * _BAR_WIDTH))
    chars = ["#" if i < filled else "-" for i in range(_BAR_WIDTH)]
    for threshold, mark in ((transcribe_level, "T"), (wake_level, "W")):
        pos = min(_BAR_WIDTH - 1, int(threshold / _BAR_SCALE * _BAR_WIDTH))
        chars[pos] = mark
    return "".join(chars)


async def main() -> None:
    s = load_settings().listener
    detector = WakeDetector(s)
    transcriber = PhraseTranscriber(s.whisper_model)
    print(f"thresholds: transcribe_level(T)={s.transcribe_level}  wake_level(W)={s.wake_level}")
    print(f"wake words: {', '.join(s.wake_words)}")
    print("speak normally to test the wake words; raise your voice / clap to")
    print("test the loudness wake. Tweak harp.yaml until it feels right.\n")

    async with Microphone(16000) as mic:
        async for chunk in mic.chunks():
            level = rms_level(chunk)
            sys.stdout.write(f"\r{_bar(level, s.transcribe_level, s.wake_level)} {level:.3f}")
            sys.stdout.flush()
            decision = detector.feed(chunk)
            if isinstance(decision, LoudSound):
                print(f"\n>> LOUD SOUND (level {decision.level:.2f}) → would wake HARP\n")
            elif isinstance(decision, Phrase):
                print("\n   transcribing…")
                text = await asyncio.to_thread(transcriber.transcribe, decision.pcm, 16000)
                word = match_wake_word(text, s.wake_words)
                verdict = f'→ would wake HARP (wake word "{word}")' if word else "→ no wake word, stays asleep"
                print(f'   heard: "{text}"  {verdict}\n')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")
