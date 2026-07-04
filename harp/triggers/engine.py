"""The proactive-trigger rule engine.

Subscribes to GestureDetected + PersonIdentified (and reads memory for open
follow-ups). When a rule fires AND its guard rails allow, it publishes a wake
request that the orchestrator treats as a wake condition. Only meaningful while
STANDBY — it must not pester someone already mid-conversation.

To build:
  - a small rule set (wave → greet; known-person-with-follow-up → re-engage),
  - cooldowns + de-dup so HARP doesn't repeatedly approach the same person,
  - publish `WakeRequested` (core/events.py); the orchestrator owns the actual
    open and only honors it while STANDBY.
"""

from __future__ import annotations

from ..core.bus import Bus


class TriggerEngine:
    def __init__(self, bus: Bus) -> None:
        self._bus = bus

    async def run(self) -> None:
        """Evaluate proactive rules against bus events and request wakes."""
        raise NotImplementedError
