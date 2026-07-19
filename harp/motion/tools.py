"""The base-motor tools: let the live model drive the stall patrol and follow.

Shape mirrors interaction/session_tools.py: `declarations(provider)` /
`follow_declarations(provider)` for SessionConfig.tools, and async handlers
that app.py routes here from its dispatch. Each handler talks to the ONE
controller that owns its behavior — MoveAroundController for the patrol
(shared with the dashboard's button), FollowController for follow mode — so
the model, the dashboard, and the modes themselves can never fight over the
motors: whoever starts first wins, the other is told why.

Both tools take one 'start'/'stop' action: move_around runs the bounded
patrol lap; follow_person drives toward the KNOWN person face-ID sees
(refusing for strangers) until stopped or the person is lost from view. The
wording that teaches the model when to do either lives in
prompts/move_around_tool.md and prompts/follow_tool.md (see prompts/README.md).
"""

from __future__ import annotations

from typing import Any

from ..config import load_follow_tool_description, load_move_around_tool_description
from .controller import MoveAroundController
from .follow import FollowController

TOOL_NAME = "move_around"
FOLLOW_TOOL_NAME = "follow_person"

# Wording lives in prompts/ (see prompts/README.md).
_DESCRIPTION = load_move_around_tool_description()
_FOLLOW_DESCRIPTION = load_follow_tool_description()

_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["start", "stop"],
            "description": (
                "'start' begins the patrol lap (the default); 'stop' halts "
                "the base immediately."
            ),
        }
    },
    "required": [],
}

_FOLLOW_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["start", "stop"],
            "description": (
                "'start' begins following the recognized person in front of "
                "you (the default); 'stop' halts the base immediately."
            ),
        }
    },
    "required": [],
}


def _declare(base: dict, provider: str) -> list[dict]:
    if provider == "openai":
        return [{"type": "function", **base}]
    if provider == "gemini":
        return [{"function_declarations": [base]}]
    raise ValueError(f"unknown provider: {provider!r} (expected 'gemini' or 'openai')")


def declarations(provider: str) -> list[dict]:
    """The move_around tool declaration, in `provider`'s format."""
    return _declare(
        {"name": TOOL_NAME, "description": _DESCRIPTION, "parameters": _PARAMETERS},
        provider,
    )


def follow_declarations(provider: str) -> list[dict]:
    """The follow_person tool declaration, in `provider`'s format."""
    return _declare(
        {
            "name": FOLLOW_TOOL_NAME,
            "description": _FOLLOW_DESCRIPTION,
            "parameters": _FOLLOW_PARAMETERS,
        },
        provider,
    )


async def move_around(controller: MoveAroundController, arguments: dict) -> Any:
    """Start or stop the patrol; return a small result for respond_tool."""
    action = str(arguments.get("action", "start")).strip().lower() or "start"
    if action == "stop":
        return await controller.stop(note="the agent asked to stop")
    if action != "start":
        return {"error": f"unknown action: {action!r} (expected 'start' or 'stop')"}
    return await controller.start()


async def follow_person(controller: FollowController, arguments: dict) -> Any:
    """Start or stop follow mode; return a small result for respond_tool."""
    action = str(arguments.get("action", "start")).strip().lower() or "start"
    if action == "stop":
        return await controller.stop(note="the agent asked to stop following")
    if action != "start":
        return {"error": f"unknown action: {action!r} (expected 'start' or 'stop')"}
    return await controller.start()
