"""Local phrase transcription for wake-word matching (faster-whisper).

Loaded lazily: importing this module is cheap, and the Whisper model (a one-time
download of ~75–150 MB depending on `whisper_model` in harp.yaml) is only pulled
in on the first real transcription. Runs on CPU in int8 — a 2–4 s phrase takes
well under a second, which is fine for deciding whether to wake.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PhraseTranscriber:
    def __init__(self, model_name: str = "base") -> None:
        self._model_name = model_name
        self._model = None

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        """Transcribe one captured phrase (16-bit mono PCM, 16 kHz)."""
        if sample_rate != 16000:
            raise ValueError("Whisper expects 16 kHz audio; got %d" % sample_rate)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self._load().transcribe(
            audio, beam_size=1, condition_on_previous_text=False
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import, deferred

            logger.info("loading whisper model %r (first use downloads it)", self._model_name)
            self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
        return self._model
