"""session_tools.end_session: the model hanging up on itself.

Calling it publishes EndOfInteractionDetected (the same event the face-absence
end rule uses), which the orchestrator turns into a clean ACTIVE → STANDBY.
"""

from __future__ import annotations

import asyncio

import pytest

from harp.core.bus import Bus
from harp.core.events import EndOfInteractionDetected
from harp.interaction import session_tools


async def test_end_session_publishes_end_event_and_acks():
    bus = Bus()
    ends = bus.subscribe(EndOfInteractionDetected)

    result = await session_tools.end_session(bus, {"reason": "visitor said goodbye"})

    ev = await asyncio.wait_for(anext(ends), 1.0)
    assert "visitor said goodbye" in ev.reason
    assert result["ok"] is True


async def test_end_session_without_reason_still_works():
    bus = Bus()
    ends = bus.subscribe(EndOfInteractionDetected)

    await session_tools.end_session(bus, {})

    ev = await asyncio.wait_for(anext(ends), 1.0)
    assert ev.reason  # a non-empty, human-readable reason is always present


def test_declarations_shaped_per_provider():
    (openai,) = session_tools.declarations("openai")
    assert openai["type"] == "function" and openai["name"] == session_tools.TOOL_NAME

    (gemini,) = session_tools.declarations("gemini")
    assert gemini["function_declarations"][0]["name"] == session_tools.TOOL_NAME

    with pytest.raises(ValueError):
        session_tools.declarations("nope")
