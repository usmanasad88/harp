"""Session-control tools the live model can call on itself.

So far: `end_session` — lets the realtime agent hang up. When the visitor says
goodbye or clearly asks to close the conversation, the model calls it; we
publish `EndOfInteractionDetected`, which the orchestrator handles exactly like
the face-absence end rule (ACTIVE → STANDBY, session torn down, InteractionEnded
published). It is the model-driven counterpart to the automatic end rules.

Shape mirrors knowledge/tools.py: `declarations(provider)` for
SessionConfig.tools, and an async handler that returns a small result for
VoiceConnection.respond_tool. The handler needs the bus (it publishes an
event), so app.py — the composition root — routes `end_session` here and
everything else to knowledge_tools.dispatch.
"""

from __future__ import annotations

from typing import Any

from ..config import load_end_session_description
from ..core.bus import Bus
from ..core.events import EndOfInteractionDetected

TOOL_NAME = "end_session"

# Wording lives in prompts/end_session_tool.md (see prompts/README.md) — this
# is what teaches the model when it's appropriate to hang up on itself.
_DESCRIPTION = load_end_session_description()

_PARAMETERS = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Brief note on why you're ending, e.g. 'visitor said goodbye'.",
        }
    },
    "required": [],
}


def declarations(provider: str) -> list[dict]:
    """Tool declarations for SessionConfig.tools, in `provider`'s format."""
    base = {"name": TOOL_NAME, "description": _DESCRIPTION, "parameters": _PARAMETERS}
    if provider == "openai":
        return [{"type": "function", **base}]
    if provider == "gemini":
        return [{"function_declarations": [base]}]
    raise ValueError(f"unknown provider: {provider!r} (expected 'gemini' or 'openai')")


async def end_session(bus: Bus, arguments: dict) -> Any:
    """Request the orchestrator close the live session; ack back to the model."""
    reason = str(arguments.get("reason", "")).strip() or "no reason given"
    await bus.publish(
        EndOfInteractionDetected(
            reason=f"agent ended the session ({reason})", cause="agent"
        )
    )
    return {"ok": True, "note": "Ending the session now. Goodbye."}
