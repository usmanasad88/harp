"""End-of-interaction → a per-person memory summary.

On InteractionEnded, read that interaction's log and write a concise summary
(what they asked, what got resolved, and any open follow-up intent such as "was
looking for hall B") into the store, attached to the person. This is what makes
the next meeting contextual and what feeds memory-based proactive triggers.

To build:
  - trigger on InteractionEnded, load the interaction log,
  - summarize (an LLM call) into a compact record + extract follow-up intents,
  - save via memory/store.
"""

from __future__ import annotations


async def summarize_interaction(log, person_id: str) -> str:
    """Turn one interaction's log into a stored memory summary for the person."""
    raise NotImplementedError
