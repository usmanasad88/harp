"""HARP's supervisor: the state machine that gates the cloud voice session.

Responsibilities
----------------
  - Boot: play the canned "starting up" line (status_voice), dial the provider,
    then STARTING → STANDBY (or → ERROR if the cloud is unreachable).
  - Idle (STANDBY): wait for a WakeRequested on the bus — published by whoever
    detected a wake condition (speech, a wave, the button, a proactive trigger).
    On wake, open a voice session → ACTIVE.
  - Active (ACTIVE): run the live session (harp/voice), bridging its VoiceEvents
    onto the bus (UserSaid / AgentSaid / ToolRequested). Close it and return to
    STANDBY when EndOfInteractionDetected arrives (harp/interaction).
  - Errors: on ErrorRaised(fatal=False), narrate in plain language and retry with
    backoff (orchestrator/retry.py); on fatal, head to STOPPING.
  - Emit Heartbeat so the watchdog (orchestrator/watchdog.py) knows we're alive.

WAKE POLICY LIVES HERE. The detectors only report facts (present / identified /
gesture / button); the decision to actually spend money on a session is the
orchestrator's alone: it honors WakeRequested only while STANDBY. It imports
nothing from the subsystems — only core.bus, core.events, core.state, and the
voice layer it drives.

Current status: the state machine, wake/end flow, error handling with backoff,
heartbeat, the voice session, AND status narration are real. `_open_session`
runs the injected VoiceBridge (harp/voice/bridge.py) as a task, `_close_session`
cancels it. An injected StatusVoice (orchestrator/status_voice.py) speaks canned
lines at boot / standby / error / shutdown, and an injected connectivity probe
lets boot announce "connection established" vs "no internet". Both the bridge
and the status voice are optional — constructed without them (tests, partial
wiring) the orchestrator drives state and events exactly as before, silently.

An optional `patrol_active_check` (() -> bool) lets an external autonomous-
patrol process (harp/motion/autonomous_patrol.py) suppress voice wakes while it
owns the wheels — checked on every WakeRequested via a tiny local HTTP flag
(harp/motion/patrol_state.py), imports nothing from harp.motion. When it
returns True the wake is simply ignored, exactly like an "ignoring wake while
ACTIVE" no-op; the gimbal (a separate process) keeps tracking faces the whole
time regardless.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path

from ..core.bus import Bus
from ..core.events import (
    EndOfInteractionDetected,
    ErrorRaised,
    Heartbeat,
    InteractionEnded,
    InteractionStarted,
    ShutdownRequested,
    StateChanged,
    WakeRequested,
)
from ..core.state import AppState, can_transition
from .retry import backoff_seconds, should_give_up

logger = logging.getLogger(__name__)


def _error_line(where: str) -> str:
    """Map an ErrorRaised.where to the closest canned status clip."""
    w = where.lower()
    if "mic" in w or "audio" in w:
        return "mic_problem"
    if "voice" in w or "session" in w or "provider" in w or "network" in w:
        return "connection_lost"
    return "error_recoverable"


class Orchestrator:
    def __init__(
        self,
        bus: Bus,
        provider_name: str,
        heartbeat_interval: float = 5.0,
        heartbeat_file: Path | None = None,
        voice_bridge=None,
        status_voice=None,
        connectivity_check=None,
        patrol_active_check=None,
    ) -> None:
        self._bus = bus
        self._provider = provider_name
        self._state = AppState.STARTING
        self._heartbeat_interval = heartbeat_interval
        # The watchdog is a separate process and can't see the bus, so liveness
        # is also written to a file: its mtime = the last heartbeat.
        self._heartbeat_file = heartbeat_file
        # Consecutive non-fatal errors; reset once a session opens fine again.
        self._error_count = 0
        self._first_error_at: float | None = None
        # The real voice session (VoiceBridge, injected by app.py). None keeps
        # the old seam behavior: state + events only, no session — which is
        # what the bus-driven tests and partial wirings rely on.
        self._voice_bridge = voice_bridge
        self._session_task: asyncio.Task | None = None
        # Canned status narration (orchestrator/status_voice.StatusVoice) and an
        # optional boot connectivity probe (() -> bool). Both injected by app.py;
        # both None keeps the old behavior — silent boot, no connection lines —
        # which every existing bus-driven test relies on.
        self._status = status_voice
        self._connectivity_check = connectivity_check
        # Optional (() -> bool) probe: True while an external autonomous-patrol
        # process is driving the wheels. Injected by app.py; None keeps the old
        # behavior (every wake is honored while STANDBY).
        self._patrol_active_check = patrol_active_check

    @property
    def state(self) -> AppState:
        return self._state

    async def run(self) -> None:
        """Main supervisor loop: boot, then react to bus events until shutdown."""
        events = self._bus.subscribe(
            WakeRequested, EndOfInteractionDetected, ErrorRaised, ShutdownRequested
        )
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            await self._boot()
            async for ev in events:
                if isinstance(ev, ShutdownRequested):
                    await self._shutdown(ev.reason or "shutdown requested")
                elif isinstance(ev, WakeRequested):
                    if self._patrol_active_check is not None and self._patrol_active_check():
                        logger.debug(
                            "ignoring wake (%s) — autonomous patrol is active", ev.reason
                        )
                    elif self._state is AppState.STANDBY:
                        await self._open_session(ev.reason, ev.context)
                    else:
                        logger.debug(
                            "ignoring wake (%s) while %s", ev.reason, self._state.value
                        )
                elif isinstance(ev, EndOfInteractionDetected):
                    if self._state is AppState.ACTIVE:
                        await self._close_session(ev.reason)
                elif isinstance(ev, ErrorRaised):
                    await self._handle_error(ev)
                if self._state is AppState.STOPPING:
                    break
        finally:
            heartbeat.cancel()
            await self._stop_session_task()

    # --- state transitions ----------------------------------------------------

    async def _to(self, target: AppState, reason: str = "") -> None:
        """Validate against core.state, perform the move, publish StateChanged."""
        if not can_transition(self._state, target):
            raise RuntimeError(
                f"illegal state transition {self._state.value} → {target.value}"
                f" ({reason})"
            )
        old, self._state = self._state, target
        logger.info("state %s → %s (%s)", old.value, target.value, reason)
        await self._bus.publish(StateChanged(old=old.value, new=target.value))

    async def _boot(self) -> None:
        """STARTING → STANDBY: announce the boot, then (if a connectivity probe
        was injected) report whether we can reach the internet, so a failure to
        dial the cloud is spoken up front rather than only surfacing on the first
        wake. The probe is best-effort — it never blocks reaching STANDBY."""
        await self._say("starting_up")
        if self._connectivity_check is not None:
            reachable = await self._check_connectivity()
            await self._say("connection_established" if reachable else "no_internet")
        await self._to(AppState.STANDBY, "boot complete")

    async def _shutdown(self, reason: str) -> None:
        """Close any open session gracefully, then head to STOPPING."""
        if self._state is AppState.ACTIVE:
            await self._close_session(f"shutting down: {reason}", narrate=False)
        await self._say("shutting_down")
        await self._to(AppState.STOPPING, reason)

    # --- session lifecycle ------------------------------------------------------

    async def _open_session(self, reason: str, context: str = "") -> None:
        """STANDBY → ACTIVE: run the voice bridge, which pumps the live session's
        events onto the bus. `context` is delivered to the model at session
        start so it knows why it was woken (what was said, who waved, ...)."""
        await self._to(AppState.ACTIVE, reason)
        # A successful open proves we've recovered from any earlier errors.
        self._error_count = 0
        self._first_error_at = None
        if self._voice_bridge is not None:
            self._session_task = asyncio.create_task(self._run_session(context))
        await self._bus.publish(InteractionStarted(reason=reason, context=context))

    async def _close_session(self, reason: str, narrate: bool = True) -> None:
        """ACTIVE → STANDBY: tear the session down; publish InteractionEnded so
        the memory summarizer can run. `narrate` plays the "going on standby"
        cue on a normal end; the error/shutdown paths pass narrate=False because
        they speak their own line (an error notice / a goodbye) instead."""
        await self._stop_session_task()
        await self._to(AppState.STANDBY, reason)
        await self._bus.publish(InteractionEnded(reason=reason))
        if narrate:
            await self._say("going_standby")

    async def _run_session(self, context: str) -> None:
        """Body of the session task. It never mutates orchestrator state
        directly — it translates the bridge's fate into bus events, which the
        main loop reacts to like any other subsystem's."""
        try:
            await self._voice_bridge.run(context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("voice session crashed")
            await self._bus.publish(ErrorRaised(where="voice.session", message=str(exc)))
        else:
            # The bridge returned on its own: the provider closed the stream
            # (server-side session limit, network drop that ended cleanly, or
            # a ProviderError it already reported). If nothing else moved us
            # off ACTIVE, treat it as the interaction ending.
            if self._state is AppState.ACTIVE:
                await self._bus.publish(
                    EndOfInteractionDetected(reason="voice session ended")
                )

    async def _stop_session_task(self) -> None:
        task, self._session_task = self._session_task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # --- status narration -------------------------------------------------------

    async def _say(self, line_id: str) -> None:
        """Play a canned status line if a StatusVoice was injected. Fully
        guarded: narration is best-effort and must never break the supervisor."""
        if self._status is None:
            return
        try:
            await self._status.play(line_id)
        except Exception:
            logger.exception("status narration failed for %s", line_id)

    async def _check_connectivity(self) -> bool:
        """Run the injected (blocking) connectivity probe off-thread; any failure
        counts as unreachable rather than crashing boot."""
        try:
            return bool(await asyncio.to_thread(self._connectivity_check))
        except Exception:
            logger.exception("connectivity check failed")
            return False

    # --- errors & liveness ------------------------------------------------------

    async def _handle_error(self, ev: ErrorRaised) -> None:
        """Non-fatal: narrate, back off, resume via STANDBY. Fatal: STOPPING."""
        logger.error("error in %s: %s (fatal=%s)", ev.where, ev.message, ev.fatal)
        if self._state is AppState.ACTIVE:
            await self._close_session(f"error in {ev.where}", narrate=False)
        if ev.fatal:
            await self._say("error_fatal")
            await self._to(AppState.STOPPING, ev.message)
            return
        await self._to(AppState.ERROR, ev.message)
        # Say the problem out loud (status_voice), per PLAN.md: pick the closest
        # canned line for where it failed (mic / network / generic).
        await self._say(_error_line(ev.where))
        self._error_count += 1
        now = time.monotonic()
        if self._first_error_at is None:
            self._first_error_at = now
        if should_give_up(self._error_count, now - self._first_error_at):
            await self._to(AppState.STOPPING, "too many errors, giving up")
            return
        await asyncio.sleep(backoff_seconds(self._error_count))
        await self._to(AppState.STANDBY, "retrying after error")

    async def _heartbeat(self) -> None:
        """Prove liveness forever, two ways: a Heartbeat on the bus for
        in-process observers (dashboard), and a touched file for the watchdog,
        which is a separate process and can't see the bus. A watchdog that
        finds the file's mtime too old concludes we're dead or hung."""
        if self._heartbeat_file is not None:
            self._heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        while True:
            if self._heartbeat_file is not None:
                self._heartbeat_file.touch()
            await self._bus.publish(Heartbeat(ts=time.time()))
            await asyncio.sleep(self._heartbeat_interval)
