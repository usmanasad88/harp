"""The proactive-trigger rule engine.

Subscribes to GestureDetected and turns a wave into a wake request: someone
waving at HARP is an invitation to talk. It publishes `WakeRequested`; the
orchestrator owns the actual session open and only honors it while STANDBY, so
a wave mid-conversation is harmlessly ignored.

De-duplication is deliberately upstream: the gesture recognizer
(vision/gestures.py) already debounces with a hold requirement + cooldown and
emits GestureDetected once per sustained wave, so this engine can stay a thin
translation from "a wave happened" to "please wake." Richer rules
(known-person-with-an-open-follow-up → re-engage) can join here later.
"""

from __future__ import annotations

import logging

from ..config import load_wake_context_wave
from ..core.bus import Bus
from ..core.events import GestureDetected, WakeRequested

logger = logging.getLogger(__name__)

# Wording lives in prompts/wake_context_wave.md (see prompts/README.md).
_WAVE_CONTEXT = load_wake_context_wave()


class TriggerEngine:
    def __init__(self, bus: Bus) -> None:
        self._bus = bus

    async def run(self) -> None:
        """Evaluate proactive rules against bus events and request wakes."""
        async for ev in self._bus.subscribe(GestureDetected):
            if ev.kind == "wave":
                logger.info("wave → requesting wake")
                await self._bus.publish(
                    WakeRequested(reason="wave", context=_WAVE_CONTEXT)
                )
