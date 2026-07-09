"""A loudness/proximity gate for mic audio: only audio at or above a live RMS
threshold passes; anything quieter becomes digital silence, so noise never
commits a provider VAD turn. Shared by whichever agent currently owns a real
microphone — the single-agent `VoiceBridge` and the two-agent `FilterAgent`
(see bridge.py / filter_agent.py) — since both face the same room noise.

A short pre-roll (so a word's onset isn't clipped when speech first crosses
the threshold) and a hangover (so a brief dip mid-word doesn't chop the turn)
keep it usable rather than choppy. The threshold is read fresh every chunk via
`level()`, so a dashboard change takes effect instantly; `level() <= 0`
disables the gate (everything passes).
"""

from __future__ import annotations

from collections import deque
from typing import Callable

from ..listener.detector import rms_level


class LoudnessGate:
    def __init__(
        self,
        level: Callable[[], float],
        preroll_chunks: int = 3,
        hangover_chunks: int = 8,
    ) -> None:
        self._level = level
        self._preroll: deque[bytes] = deque(maxlen=max(1, preroll_chunks))
        self._hangover_max = max(1, hangover_chunks)
        self._hangover = 0

    def process(self, pcm: bytes) -> bytes:
        threshold = self._level()
        if threshold <= 0.0:
            return pcm  # gate disabled: pass the room through untouched
        if rms_level(pcm) >= threshold:
            self._hangover = self._hangover_max
            if self._preroll:  # speech onset: prepend the buffered lead-in once
                lead_in = b"".join(self._preroll)
                self._preroll.clear()
                return lead_in + pcm
            return pcm
        if self._hangover > 0:  # trailing quiet within a word — keep passing
            self._hangover -= 1
            return pcm
        self._preroll.append(pcm)   # below threshold: hold as possible lead-in
        return bytes(len(pcm))       # ...and send silence in its place
