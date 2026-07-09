"""The always-on listener subsystem: mic + detector + bus.

Owns the microphone while HARP is idle. Publishes `WakeRequested` with a
model-facing `context` string so the live session, once opened, knows *why* it
was woken (the orchestrator delivers that context to Gemini Live / OpenAI
Realtime at session start).

Mic sharing: this subsystem listens only while the app is in STANDBY. On
StateChanged to anything else it closes its mic stream (the live voice session
needs the device) and reopens it when STANDBY returns. Run standalone — with no
orchestrator publishing states — it simply listens forever, which is what the
calibration tool (`python -m harp.listener`) relies on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import sounddevice as sd

from ..config import (
    FALLBACK_WAKE_CONTEXT_LOUD_SOUND,
    FALLBACK_WAKE_CONTEXT_WAKE_WORD,
    ListenerSettings,
    format_prompt,
    load_wake_context_loud_sound,
    load_wake_context_wake_word,
)
from ..core.bus import Bus
from ..core.events import PhraseHeard, StateChanged, WakeRequested
from ..voice.audio_io import Microphone
from .detector import LoudSound, Phrase, WakeDetector, match_wake_word
from .transcriber import PhraseTranscriber

logger = logging.getLogger(__name__)

# After requesting a wake, ignore the mic briefly so we don't fire twice before
# the orchestrator reacts (its StateChanged will pause us properly).
_COOLDOWN_SECONDS = 2.0

# The mic can be temporarily unavailable — another app holds it, the OS is
# blocking microphone access, a USB device dropped out. When that happens we
# retry rather than take the whole agent down (PLAN: "narrates its own problems
# and retries"). Backoff between attempts.
_MIC_RETRY_SECONDS = 5.0


class AlwaysOnListener:
    def __init__(
        self,
        bus: Bus,
        settings: ListenerSettings,
        transcribe: Callable[[bytes, int], str] | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self._bus = bus
        self._s = settings
        self._rate = sample_rate
        self._detector = WakeDetector(settings, sample_rate)
        # Injectable for tests; the real Whisper wrapper is created lazily in
        # run() so merely constructing the subsystem stays cheap.
        self._transcribe = transcribe
        self._listening = True

    async def run(self) -> None:
        """Listen on the mic whenever HARP is idle; request wakes on the bus."""
        if self._transcribe is None:
            self._transcribe = PhraseTranscriber(self._s.whisper_model).transcribe
        state_task = asyncio.create_task(self._watch_state())
        try:
            while True:
                if not self._listening:
                    await asyncio.sleep(0.2)
                    continue
                try:
                    async with Microphone(self._rate) as mic:
                        async for chunk in mic.chunks():
                            if not self._listening:
                                break  # release the device for the live session
                            await self._handle(self._detector.feed(chunk))
                except (sd.PortAudioError, OSError) as exc:
                    # The mic wouldn't open (or dropped out mid-listen). Don't let
                    # it kill the whole agent — warn and retry; it recovers on its
                    # own once the device frees up / access is restored. (Common
                    # on Windows: OS microphone-access privacy blocking desktop
                    # apps, or another app holding the device.)
                    logger.warning(
                        "listener mic unavailable (%s) — is another app using it, "
                        "or is OS microphone access blocked for desktop apps? "
                        "retrying in %.0fs",
                        exc, _MIC_RETRY_SECONDS,
                    )
                    await asyncio.sleep(_MIC_RETRY_SECONDS)
        finally:
            state_task.cancel()

    async def _watch_state(self) -> None:
        async for ev in self._bus.subscribe(StateChanged):
            self._listening = ev.new == "standby"

    async def _handle(self, decision: LoudSound | Phrase | None) -> None:
        if decision is None:
            return
        if isinstance(decision, LoudSound):
            # Wording lives in prompts/wake_context_loud_sound.md (see
            # prompts/README.md); `{level}` is filled in here.
            context = format_prompt(
                load_wake_context_loud_sound(),
                FALLBACK_WAKE_CONTEXT_LOUD_SOUND,
                level=f"{decision.level:.2f}",
            )
            await self._wake("loud sound", context)
            return
        # A phrase was captured: transcribe off-thread, then match wake words.
        text = await asyncio.to_thread(self._transcribe, decision.pcm, self._rate)
        word = match_wake_word(text, self._s.wake_words)
        logger.info("heard %r → wake word: %s", text, word)
        if text.strip():
            # Published match or not, so observers (dashboard) can see what the
            # ears picked up and why it did/didn't wake.
            await self._bus.publish(PhraseHeard(text=text, wake_word=word))
        if word:
            # Wording lives in prompts/wake_context_wake_word.md (see
            # prompts/README.md); `{text}` is the transcribed phrase.
            context = format_prompt(
                load_wake_context_wake_word(), FALLBACK_WAKE_CONTEXT_WAKE_WORD, text=text
            )
            await self._wake("wake word", context)

    async def _wake(self, reason: str, context: str) -> None:
        logger.info("requesting wake: %s", reason)
        await self._bus.publish(WakeRequested(reason=reason, context=context))
        await asyncio.sleep(_COOLDOWN_SECONDS)
