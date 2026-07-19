"""IdlePrompt: the timer must exist exactly while the app is in STANDBY.

Timing/async behavior is the whole subsystem — the failure modes are silent
(a prompt that never plays, keeps playing over a live conversation, or fires
the instant standby starts instead of one interval in), so each is pinned
here with a real event loop and a tiny interval. The StatusVoice is faked;
clip playback itself is covered by test_status_voice.py.
"""

from __future__ import annotations

import asyncio
import contextlib

from harp.core.bus import Bus
from harp.core.events import StateChanged
from harp.interaction.idle_prompt import IdlePrompt

INTERVAL = 0.05


class FakeStatusVoice:
    def __init__(self) -> None:
        self.played: list[str] = []

    async def play(self, line_id: str) -> None:
        self.played.append(line_id)


async def _wait_for(condition, timeout: float = 2.0) -> None:
    """Poll until `condition()` — generous timeout so a loaded machine can't
    flake the test; the assertions on counts do the real checking."""
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


async def test_prompts_only_while_standby():
    bus = Bus()
    status = FakeStatusVoice()
    task = asyncio.create_task(IdlePrompt(bus, status, interval_seconds=INTERVAL).run())
    await asyncio.sleep(0)  # let run() subscribe before events flow

    # Entering standby starts the timer — but the first prompt comes one full
    # interval later, never instantly (boot lines get airtime first).
    await bus.publish(StateChanged(old="starting", new="standby"))
    await asyncio.sleep(INTERVAL / 2)
    assert status.played == []
    await _wait_for(lambda: len(status.played) >= 2)  # and it repeats
    assert set(status.played) == {"hold_green_button"}

    # A wake (ACTIVE) stops the prompting for the whole session. One interval
    # of settling absorbs a prompt already in flight when the wake landed.
    await bus.publish(StateChanged(old="standby", new="active"))
    await asyncio.sleep(INTERVAL)
    quiet_mark = len(status.played)
    await asyncio.sleep(INTERVAL * 3)
    assert len(status.played) == quiet_mark

    # The session ending (back to STANDBY) resumes the invitations.
    await bus.publish(StateChanged(old="active", new="standby"))
    await _wait_for(lambda: len(status.played) > quiet_mark)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
