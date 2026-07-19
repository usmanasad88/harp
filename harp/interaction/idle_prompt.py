"""Intermittent "how to talk to me" prompt for push-to-talk mode.

With push-to-talk armed (harp.yaml `push_to_talk:`) — especially exclusive
mode at a noisy expo — a passer-by has no way to know HARP is voice-driven,
let alone that a button must be held. While HARP sits idle (STANDBY), this
subsystem periodically plays a canned status clip ("Please hold the green
button to talk to me", status_voice id `hold_green_button`), so the robot
keeps inviting interaction without spending a cloud session.

Design notes:
  - Pure bus consumer: it watches StateChanged and runs a repeating timer only
    while the app is in STANDBY. Leaving STANDBY (a wake, an error, shutdown)
    cancels the timer immediately, so the prompt never opens over a live
    conversation. If a press lands mid-clip, the clip finishes (sounddevice
    playback isn't interruptible) — a ~2s overlap at worst.
  - Playback goes through the SAME StatusVoice the orchestrator narrates with;
    its internal lock serializes clips, so the prompt can never talk over a
    boot / error / goodbye line (or vice versa).
  - The first prompt comes one full interval AFTER standby starts, never
    instantly: the "going on standby" / connectivity lines get airtime first,
    and a person whose conversation just ended isn't immediately lectured.
"""

from __future__ import annotations

import asyncio
import contextlib

from ..core.bus import Bus
from ..core.events import StateChanged
from ..core.state import AppState


class IdlePrompt:
    """Replays one status clip every `interval_seconds` while HARP is idle."""

    def __init__(
        self,
        bus: Bus,
        status_voice,
        line_id: str = "hold_green_button",
        interval_seconds: float = 45.0,
    ) -> None:
        self._bus = bus
        self._status = status_voice
        self._line_id = line_id
        self._interval = interval_seconds

    async def run(self) -> None:
        """Track app state until cancelled; a timer exists only while STANDBY."""
        timer: asyncio.Task | None = None
        try:
            async for ev in self._bus.subscribe(StateChanged):
                standby = ev.new == AppState.STANDBY.value
                if standby and timer is None:
                    timer = asyncio.create_task(self._prompt_loop())
                elif not standby and timer is not None:
                    timer.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await timer
                    timer = None
        finally:
            if timer is not None:
                timer.cancel()

    async def _prompt_loop(self) -> None:
        """One prompt per interval. play() returns when the clip ends, so the
        gap is measured from the end of one prompt to the start of the next —
        a slow clip can't make prompts pile up."""
        while True:
            await asyncio.sleep(self._interval)
            await self._status.play(self._line_id)
