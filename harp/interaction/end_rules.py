"""End-of-interaction detection.

Watches the bus for the signal that means "the conversation is over" and tells
the orchestrator to close the session. The rule HARP uses: the person walked
off — no face in frame for a continuous stretch (`absence_timeout`, default
10 s) while a session is open. Face-ID doubles as the presence signal
(vision/face_id.py publishes PresenceChanged), so no separate detector is
needed.

Flow: arm a countdown when presence goes absent during an ACTIVE session;
cancel it the moment a face returns; if it runs out, publish
`EndOfInteractionDetected` — the signal the orchestrator consumes to move
ACTIVE → STANDBY. Only armed between InteractionStarted and InteractionEnded,
so it never fires while HARP is idle.

Presence at session start: the bus doesn't replay history, and face-ID only
publishes PresenceChanged on a *change* — so a session that opens while nobody
is in frame (woken by voice or a loud sound) would never receive a "you're
absent" event and would stay open forever. To close that gap we read the
current presence directly at InteractionStarted via the injected `is_present`
getter (wired to face-ID by app.py), the same way the dashboard seeds mic-mute
state on a fresh connection. Thereafter PresenceChanged drives it.

Err toward NOT cutting people off: the timeout absorbs brief detection dropouts
(a turned head, a bad frame) because a returning face disarms it and a fresh
absence starts the full countdown over.

SilenceMonitor is the second, independent rule: nothing said in either
direction for `silence_timeout` while a session is open also ends it. It
exists because the absence rule alone leaves a hole — with a person standing
in frame who never talks (common in exclusive push-to-talk: a false wake, or
someone who walked up but won't press the button), face presence keeps the
session open forever. Any conversation activity (UserSaid / AgentSaid, even a
streaming fragment, or the talk key changing) restarts the full countdown.
Both rules publish the same EndOfInteractionDetected, distinguished by
`cause` ("walked_off" vs "silence") so the status rule book
(orchestrator/status_rules.py) can narrate each differently.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Callable

from ..core.bus import Bus
from ..core.events import (
    AgentSaid,
    EndOfInteractionDetected,
    InteractionEnded,
    InteractionStarted,
    PresenceChanged,
    TalkKeyChanged,
    UserSaid,
)

logger = logging.getLogger(__name__)

_DEFAULT_ABSENCE_TIMEOUT = 10.0
_DEFAULT_SILENCE_TIMEOUT = 15.0


class EndOfInteractionMonitor:
    def __init__(
        self,
        bus: Bus,
        absence_timeout: float = _DEFAULT_ABSENCE_TIMEOUT,
        is_present: Callable[[], bool] | None = None,
    ) -> None:
        self._bus = bus
        self._absence_timeout = absence_timeout
        # Reads whether a face is in frame right now (face-ID). Used to seed
        # presence at session start, since the bus won't replay the last
        # PresenceChanged. None (no camera this run) = can't judge presence, so
        # we assume present and never auto-close on absence.
        self._is_present = is_present
        # A session is open (armed) only between InteractionStarted and Ended.
        self._active = False
        # Latest presence reading; refreshed from `is_present` at each open.
        self._present = True
        self._countdown: asyncio.Task | None = None

    async def run(self) -> None:
        """Watch presence + interaction lifecycle; signal when the person is gone."""
        events = self._bus.subscribe(
            InteractionStarted, InteractionEnded, PresenceChanged
        )
        try:
            async for ev in events:
                if isinstance(ev, InteractionStarted):
                    self._active = True
                    # Seed from the live reading: a session can open with nobody
                    # in frame, and no PresenceChanged would ever tell us that.
                    if self._is_present is not None:
                        self._present = self._is_present()
                elif isinstance(ev, InteractionEnded):
                    self._active = False
                elif isinstance(ev, PresenceChanged):
                    self._present = ev.present
                self._reevaluate()
        finally:
            self._disarm()

    def _reevaluate(self) -> None:
        """Count down only while a session is open AND nobody is in frame."""
        if self._active and not self._present:
            self._arm()
        else:
            self._disarm()

    def _arm(self) -> None:
        if self._countdown is None or self._countdown.done():
            self._countdown = asyncio.create_task(self._run_countdown())

    def _disarm(self) -> None:
        if self._countdown is not None:
            self._countdown.cancel()
            self._countdown = None

    async def _run_countdown(self) -> None:
        """Absent for the full timeout with no face returning: end the session."""
        await asyncio.sleep(self._absence_timeout)
        reason = f"no face for {self._absence_timeout:.0f}s"
        logger.info("ending interaction: %s", reason)
        await self._bus.publish(
            EndOfInteractionDetected(reason=reason, cause="walked_off")
        )


class SilenceMonitor:
    """Ends a session when NOBODY has said anything for `silence_timeout`.

    The countdown starts the moment a session opens (so a session where nothing
    is ever said still closes) and restarts on every sign of conversation:
    UserSaid / AgentSaid — including streaming fragments, which prove someone
    is mid-utterance — and TalkKeyChanged, so a person who just pressed (or
    released) the talk button isn't cut off before their words are transcribed.
    Like the absence rule, it is armed only between InteractionStarted and
    InteractionEnded; it can never fire while HARP is idle."""

    def __init__(self, bus: Bus, silence_timeout: float = _DEFAULT_SILENCE_TIMEOUT) -> None:
        self._bus = bus
        self._silence_timeout = silence_timeout
        # Armed only while a session is open (InteractionStarted → Ended).
        self._active = False
        self._countdown: asyncio.Task | None = None

    async def run(self) -> None:
        """Watch the conversation; signal when it has gone quiet for too long."""
        events = self._bus.subscribe(
            InteractionStarted, InteractionEnded, UserSaid, AgentSaid, TalkKeyChanged
        )
        try:
            async for ev in events:
                if isinstance(ev, InteractionStarted):
                    self._active = True
                    self._restart()  # quiet-from-the-start sessions still close
                elif isinstance(ev, InteractionEnded):
                    self._active = False
                    self._disarm()
                elif self._active:
                    # UserSaid / AgentSaid / TalkKeyChanged mid-session: the
                    # conversation is alive, the full countdown starts over.
                    self._restart()
        finally:
            self._disarm()

    def _restart(self) -> None:
        self._disarm()
        self._countdown = asyncio.create_task(self._run_countdown())

    def _disarm(self) -> None:
        if self._countdown is not None:
            self._countdown.cancel()
            self._countdown = None

    async def _run_countdown(self) -> None:
        """A full timeout of quiet with a session open: end it."""
        await asyncio.sleep(self._silence_timeout)
        reason = f"no conversation for {self._silence_timeout:.0f}s"
        logger.info("ending interaction: %s", reason)
        await self._bus.publish(EndOfInteractionDetected(reason=reason, cause="silence"))
