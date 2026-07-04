"""Pure wake-decision logic: audio levels in, decisions out.

No microphone, no bus, no Whisper model — just arithmetic on PCM chunks, so it
can be unit-tested with synthetic audio. Timing is derived from sample counts,
not wall clocks, which keeps tests deterministic.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import ListenerSettings


@dataclass
class LoudSound:
    """Level ≥ wake_level: wake immediately, no words needed."""

    level: float


@dataclass
class Phrase:
    """A captured stretch of speech-level audio, ready to transcribe."""

    pcm: bytes  # 16-bit mono PCM at the detector's sample rate


def rms_level(chunk: bytes) -> float:
    """Loudness of one PCM chunk, normalized to 0.0–1.0 of int16 full scale."""
    samples = np.frombuffer(chunk, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((samples / 32768.0) ** 2)))


class WakeDetector:
    """Feed mic chunks in; get LoudSound / Phrase / None out.

    A few chunks of pre-roll are kept so the first syllable of "hello" isn't
    clipped off the captured phrase.
    """

    def __init__(
        self,
        settings: ListenerSettings,
        sample_rate: int = 16000,
        pre_roll_chunks: int = 3,
    ) -> None:
        self._s = settings
        self._rate = sample_rate
        self._pre: deque[bytes] = deque(maxlen=pre_roll_chunks)
        self._buf: list[bytes] = []
        self._captured = 0.0  # seconds captured so far
        self._silence = 0.0  # consecutive quiet seconds while capturing

    def feed(self, chunk: bytes) -> LoudSound | Phrase | None:
        level = rms_level(chunk)
        seconds = len(chunk) / 2 / self._rate

        if level >= self._s.wake_level:
            self._reset()
            return LoudSound(level=level)

        if not self._buf:  # idle: waiting for speech-level sound
            if level >= self._s.transcribe_level:
                self._buf = list(self._pre) + [chunk]
                self._captured = seconds
                self._silence = 0.0
            else:
                self._pre.append(chunk)
            return None

        # capturing a phrase
        self._buf.append(chunk)
        self._captured += seconds
        self._silence = (
            0.0 if level >= self._s.transcribe_level else self._silence + seconds
        )
        if (
            self._silence >= self._s.silence_seconds
            or self._captured >= self._s.max_phrase_seconds
        ):
            phrase = Phrase(pcm=b"".join(self._buf))
            self._reset()
            return phrase
        return None

    def _reset(self) -> None:
        self._buf = []
        self._pre.clear()
        self._captured = 0.0
        self._silence = 0.0


_WORDS = re.compile(r"\w+", re.UNICODE)


def match_wake_word(transcript: str, wake_words: list[str]) -> str | None:
    """Return the first wake word found in the transcript, else None.

    Matches whole words only ("hi" must not match inside "this"), works across
    punctuation and case, and supports multi-word entries and non-Latin script.
    """
    spoken = " ".join(_WORDS.findall(transcript.lower()))
    if not spoken:
        return None
    for word in wake_words:
        wanted = " ".join(_WORDS.findall(str(word).lower()))
        if wanted and f" {wanted} " in f" {spoken} ":
            return str(word)
    return None
