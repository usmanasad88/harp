"""Local phrase transcription for wake-word matching (faster-whisper).

Loaded lazily: importing this module is cheap, and the Whisper model (a one-time
download of ~75–150 MB depending on `whisper_model` in harp.yaml) is only pulled
in on the first real transcription. Runs on CPU in int8 — a 2–4 s phrase takes
well under a second, which is fine for deciding whether to wake.

Offline-first load: once the model is cached, we load it with
`local_files_only=True` so startup never touches the network. Without that,
huggingface_hub makes a HEAD request to re-validate the cached files against the
server on every load — which stalls for tens of seconds on a flaky connection
and looks like a hang mid-"transcribing…". Only the genuinely-first use (nothing
cached) falls back to a networked download.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Bias Whisper toward romanized (Latin-script) Urdu. Raw Whisper otherwise emits
# spoken Urdu in Perso-Arabic, which the romanized wake words in harp.yaml
# ("salam", "assalam", ...) could never match. This is an `initial_prompt`, so
# it only nudges the decoder's script/style — best-effort, verify live. It holds
# NO wake word on purpose: if Whisper ever regurgitates the prompt on silence,
# it must not cause a false wake. Override with HARP_WHISPER_PROMPT.
_ROMAN_URDU_PROMPT = (
    "Yeh baat-cheet Roman Urdu aur English mein likhi gayi hai. "
    "Aap kaisay hain? Main theek hoon, shukriya."
)


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
            audio,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=os.getenv("HARP_WHISPER_PROMPT", _ROMAN_URDU_PROMPT),
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import, deferred

            try:
                # Cached already: load straight from disk, no network. Skips the
                # huggingface_hub HEAD re-validation that otherwise runs on every
                # load and can hang for tens of seconds on a flaky connection.
                self._model = WhisperModel(
                    self._model_name, device="cpu", compute_type="int8",
                    local_files_only=True,
                )
            except Exception:
                # Not cached yet (genuinely first use): download it once — after
                # this it's local for good and the offline path above wins.
                logger.info(
                    "downloading whisper model %r (one-time, ~75–150 MB)", self._model_name
                )
                self._model = WhisperModel(
                    self._model_name, device="cpu", compute_type="int8",
                )
        return self._model
