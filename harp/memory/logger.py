"""Append-only log of an interaction, turn by turn.

Subscribes to UserSaid / AgentSaid (and tool events) and writes them to a
per-interaction transcript that the summarizer later reads. Cheap and always-on
during ACTIVE — it's the raw record captured before any summarization.

To build:
  - open a log per interaction (keyed by person_id + timestamp),
  - append each turn as a structured line (who, text, ts),
  - flush / close on InteractionEnded.
"""

from __future__ import annotations

from ..core.bus import Bus


class InteractionLogger:
    def __init__(self, bus: Bus) -> None:
        self._bus = bus

    async def run(self) -> None:
        """Subscribe to conversation events and persist them per interaction."""
        raise NotImplementedError
