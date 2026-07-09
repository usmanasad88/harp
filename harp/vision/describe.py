"""describe_scene: the live model looks through the camera, mid-session.

The realtime session doesn't stream video; what it gets at open is the
pre-computed briefing (memory/context). This tool covers the rest of the
conversation: when a visitor asks "what can you see?", "how many of us are
there?", or shows the robot something, the model calls describe_scene and the
parallel Flash Lite agent (memory/agent) describes the CURRENT shared-camera
frame. The call waits for a rate-limiter slot (a model and a visitor are
holding for the answer) but still degrades to an {"error": ...} payload —
never an exception — when the camera or the helper is unavailable.

Shape mirrors knowledge/tools.py: `declarations(provider)` for
SessionConfig.tools, and an async handler returning a payload for
VoiceConnection.respond_tool. app.py wires the frame source (a clean,
overlay-free jpeg_snapshot) and routes the call here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..config import (
    FALLBACK_DESCRIBE_SCENE_PROMPT,
    format_prompt,
    load_describe_scene_prompt,
    load_describe_scene_tool_description,
)
from ..memory.agent import GeminiAgent

logger = logging.getLogger(__name__)

TOOL_NAME = "describe_scene"

# Wording lives in prompts/describe_scene_tool.md (see prompts/README.md).
_DESCRIPTION = load_describe_scene_tool_description()

_PARAMETERS = {
    "type": "object",
    "properties": {
        "focus": {
            "type": "string",
            "description": (
                "Optional: what to look for, e.g. 'how many people', "
                "'what the visitor is holding'."
            ),
        }
    },
    "required": [],
}

# The visitor is audibly waiting on this — don't hold the turn hostage for
# longer than a slow-but-alive call would take.
_CALL_TIMEOUT_S = 15.0


def declarations(provider: str) -> list[dict]:
    """Tool declarations for SessionConfig.tools, in `provider`'s format."""
    base = {"name": TOOL_NAME, "description": _DESCRIPTION, "parameters": _PARAMETERS}
    if provider == "openai":
        return [{"type": "function", **base}]
    if provider == "gemini":
        return [{"function_declarations": [base]}]
    raise ValueError(f"unknown provider: {provider!r} (expected 'gemini' or 'openai')")


async def describe_scene(
    agent: GeminiAgent, frame_jpeg: Callable[[], bytes | None], arguments: dict
) -> Any:
    """Describe the current camera frame for the live model."""
    frame = frame_jpeg()
    if frame is None:
        return {"error": "the camera has no frame right now"}
    focus = str(arguments.get("focus", "")).strip() or "anything notable"
    prompt = format_prompt(
        load_describe_scene_prompt(), FALLBACK_DESCRIBE_SCENE_PROMPT, focus=focus
    )
    text = await agent.generate(
        prompt, image_jpeg=frame, wait=True, timeout=_CALL_TIMEOUT_S
    )
    if text is None:
        return {"error": "the vision helper is unavailable right now"}
    return {"description": text}
