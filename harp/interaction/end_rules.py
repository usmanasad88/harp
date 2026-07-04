"""End-of-interaction detection.

Watches the bus for the signals that mean "the conversation is over" — e.g. the
person left the frame (PresenceChanged / PersonIdentified) AND no one has spoken
(UserSaid) for a while — and tells the orchestrator to close the session. The
timeouts are tunable; err toward NOT cutting people off mid-thought.

To build:
  - track last-speech time and current presence,
  - fire when (absent for T_gone) or (silent for T_silence while alone),
  - publish `EndOfInteractionDetected` (core/events.py) — the signal the
    orchestrator consumes to move ACTIVE → STANDBY.
"""

from __future__ import annotations

from ..core.bus import Bus


class EndOfInteractionMonitor:
    def __init__(self, bus: Bus) -> None:
        self._bus = bus

    async def run(self) -> None:
        """Watch presence + silence and signal when the interaction should end."""
        raise NotImplementedError
