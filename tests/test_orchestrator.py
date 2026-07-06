"""Orchestrator tests: drive it purely through the bus.

No real voice, no camera — every test publishes bus events in and asserts the
StateChanged / InteractionStarted / InteractionEnded events that come out,
exactly how the real subsystems will interact with it. The voice-bridge tests
at the bottom inject a fake bridge; without one the orchestrator moves state
and publishes events but runs no session (the pre-bridge behavior the earlier
tests still cover).
"""

from __future__ import annotations

import asyncio

import pytest

from harp.core.bus import Bus
from harp.core.events import (
    EndOfInteractionDetected,
    ErrorRaised,
    InteractionEnded,
    InteractionStarted,
    ShutdownRequested,
    StateChanged,
    WakeRequested,
)
from harp.core.state import AppState
from harp.orchestrator import retry
from harp.orchestrator.orchestrator import Orchestrator, _error_line


async def next_event(stream, timeout: float = 1.0):
    return await asyncio.wait_for(anext(stream), timeout)


async def cancel(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.fixture
def bus() -> Bus:
    return Bus()


def start(bus: Bus) -> tuple[Orchestrator, asyncio.Task]:
    orch = Orchestrator(bus, "gemini")
    return orch, asyncio.create_task(orch.run())


async def test_boots_to_standby(bus):
    states = bus.subscribe(StateChanged)
    orch, task = start(bus)
    try:
        ev = await next_event(states)
        assert (ev.old, ev.new) == ("starting", "standby")
        assert orch.state is AppState.STANDBY
    finally:
        await cancel(task)


async def test_wake_opens_session(bus):
    states = bus.subscribe(StateChanged)
    started = bus.subscribe(InteractionStarted)
    orch, task = start(bus)
    try:
        await next_event(states)  # starting → standby
        await bus.publish(WakeRequested(reason="wave"))
        ev = await next_event(states)
        assert (ev.old, ev.new) == ("standby", "active")
        assert (await next_event(started)).reason == "wave"
    finally:
        await cancel(task)


async def test_wake_ignored_while_active(bus):
    states = bus.subscribe(StateChanged)
    started = bus.subscribe(InteractionStarted)
    orch, task = start(bus)
    try:
        await next_event(states)  # → standby
        await bus.publish(WakeRequested(reason="wave"))
        await next_event(states)  # → active
        await next_event(started)  # the one real open
        await bus.publish(WakeRequested(reason="wave again"))
        with pytest.raises(TimeoutError):
            await next_event(started, timeout=0.2)  # no second open
        assert orch.state is AppState.ACTIVE
    finally:
        await cancel(task)


async def test_end_of_interaction_returns_to_standby(bus):
    states = bus.subscribe(StateChanged)
    ended = bus.subscribe(InteractionEnded)
    orch, task = start(bus)
    try:
        await next_event(states)  # → standby
        await bus.publish(WakeRequested(reason="speech"))
        await next_event(states)  # → active
        await bus.publish(EndOfInteractionDetected(reason="left frame + silent"))
        ev = await next_event(states)
        assert (ev.old, ev.new) == ("active", "standby")
        assert (await next_event(ended)).reason == "left frame + silent"
    finally:
        await cancel(task)


async def test_nonfatal_error_backs_off_then_returns_to_standby(bus, monkeypatch):
    monkeypatch.setattr(
        "harp.orchestrator.orchestrator.backoff_seconds", lambda attempt: 0
    )
    states = bus.subscribe(StateChanged)
    orch, task = start(bus)
    try:
        await next_event(states)  # → standby
        await bus.publish(ErrorRaised(where="voice", message="boom"))
        assert (await next_event(states)).new == "error"
        assert (await next_event(states)).new == "standby"
        assert orch.state is AppState.STANDBY
    finally:
        await cancel(task)


async def test_nonfatal_error_while_active_closes_session_first(bus, monkeypatch):
    monkeypatch.setattr(
        "harp.orchestrator.orchestrator.backoff_seconds", lambda attempt: 0
    )
    states = bus.subscribe(StateChanged)
    ended = bus.subscribe(InteractionEnded)
    orch, task = start(bus)
    try:
        await next_event(states)  # → standby
        await bus.publish(WakeRequested(reason="speech"))
        await next_event(states)  # → active
        await bus.publish(ErrorRaised(where="voice", message="boom"))
        assert (await next_event(states)).new == "standby"  # session closed
        assert "voice" in (await next_event(ended)).reason
        assert (await next_event(states)).new == "error"
        assert (await next_event(states)).new == "standby"  # recovered
    finally:
        await cancel(task)


async def test_fatal_error_stops(bus):
    states = bus.subscribe(StateChanged)
    orch, task = start(bus)
    await next_event(states)  # → standby
    await bus.publish(ErrorRaised(where="core", message="dead", fatal=True))
    assert (await next_event(states)).new == "stopping"
    await asyncio.wait_for(task, 1.0)  # run() actually returns
    assert orch.state is AppState.STOPPING


async def test_gives_up_after_repeated_errors(bus, monkeypatch):
    monkeypatch.setattr(
        "harp.orchestrator.orchestrator.backoff_seconds", lambda attempt: 0
    )
    states = bus.subscribe(StateChanged)
    orch, task = start(bus)
    await next_event(states)  # → standby
    for _ in range(retry.MAX_ATTEMPTS - 1):
        await bus.publish(ErrorRaised(where="voice", message="boom"))
        assert (await next_event(states)).new == "error"
        assert (await next_event(states)).new == "standby"
    await bus.publish(ErrorRaised(where="voice", message="boom"))
    assert (await next_event(states)).new == "error"
    assert (await next_event(states)).new == "stopping"
    await asyncio.wait_for(task, 1.0)


async def test_successful_open_resets_error_budget(bus, monkeypatch):
    monkeypatch.setattr(
        "harp.orchestrator.orchestrator.backoff_seconds", lambda attempt: 0
    )
    states = bus.subscribe(StateChanged)
    orch, task = start(bus)
    try:
        await next_event(states)  # → standby
        # Burn all but one attempt...
        for _ in range(retry.MAX_ATTEMPTS - 1):
            await bus.publish(ErrorRaised(where="voice", message="boom"))
            await next_event(states)  # → error
            await next_event(states)  # → standby
        # ...then a session opens fine, which should reset the budget.
        await bus.publish(WakeRequested(reason="speech"))
        await next_event(states)  # → active
        await bus.publish(EndOfInteractionDetected(reason="done"))
        await next_event(states)  # → standby
        # One more error must retry (budget reset), not give up.
        await bus.publish(ErrorRaised(where="voice", message="boom"))
        assert (await next_event(states)).new == "error"
        assert (await next_event(states)).new == "standby"
    finally:
        await cancel(task)


async def test_wake_context_is_forwarded_to_interaction(bus):
    states = bus.subscribe(StateChanged)
    started = bus.subscribe(InteractionStarted)
    orch, task = start(bus)
    try:
        await next_event(states)  # → standby
        await bus.publish(
            WakeRequested(reason="wake word", context='someone said: "hello"')
        )
        await next_event(states)  # → active
        ev = await next_event(started)
        assert ev.reason == "wake word"
        assert ev.context == 'someone said: "hello"'
    finally:
        await cancel(task)


async def test_heartbeat_touches_liveness_file(bus, tmp_path):
    hb_file = tmp_path / "run" / "heartbeat"
    orch = Orchestrator(
        bus, "gemini", heartbeat_interval=0.01, heartbeat_file=hb_file
    )
    task = asyncio.create_task(orch.run())
    try:
        await asyncio.sleep(0.05)
        assert hb_file.exists()
        first = hb_file.stat().st_mtime_ns
        await asyncio.sleep(0.05)
        assert hb_file.stat().st_mtime_ns > first  # keeps beating, not just once
    finally:
        await cancel(task)


async def test_shutdown_while_active_closes_session_then_stops(bus):
    states = bus.subscribe(StateChanged)
    ended = bus.subscribe(InteractionEnded)
    orch, task = start(bus)
    await next_event(states)  # → standby
    await bus.publish(WakeRequested(reason="speech"))
    await next_event(states)  # → active
    await bus.publish(ShutdownRequested(reason="ctrl-c"))
    assert (await next_event(states)).new == "standby"  # graceful close
    assert "ctrl-c" in (await next_event(ended)).reason
    assert (await next_event(states)).new == "stopping"
    await asyncio.wait_for(task, 1.0)


# --- the voice bridge (injected; a fake stands in for the real session) -------


class FakeBridge:
    """Records how the orchestrator drives it. `behavior` controls run():
    "block" = run until cancelled (a healthy live session), "return" = end
    immediately (provider closed the stream), "raise" = crash."""

    def __init__(self, behavior: str = "block") -> None:
        self.behavior = behavior
        self.contexts: list[str] = []
        self.cancelled = False

    async def run(self, context: str = "") -> None:
        self.contexts.append(context)
        if self.behavior == "raise":
            raise RuntimeError("connect blew up")
        if self.behavior == "return":
            return
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def start_with_bridge(bus: Bus, bridge: FakeBridge) -> tuple[Orchestrator, asyncio.Task]:
    orch = Orchestrator(bus, "gemini", voice_bridge=bridge)
    return orch, asyncio.create_task(orch.run())


async def test_wake_runs_bridge_with_context_and_end_cancels_it(bus):
    states = bus.subscribe(StateChanged)
    bridge = FakeBridge()
    orch, task = start_with_bridge(bus, bridge)
    try:
        await next_event(states)  # → standby
        await bus.publish(WakeRequested(reason="wake word", context="someone said hi"))
        await next_event(states)  # → active
        await asyncio.sleep(0.05)  # let the session task start
        assert bridge.contexts == ["someone said hi"]
        assert not bridge.cancelled

        await bus.publish(EndOfInteractionDetected(reason="done"))
        assert (await next_event(states)).new == "standby"
        assert bridge.cancelled
    finally:
        await cancel(task)


async def test_bridge_crash_surfaces_as_error_and_recovers(bus, monkeypatch):
    monkeypatch.setattr(
        "harp.orchestrator.orchestrator.backoff_seconds", lambda attempt: 0
    )
    states = bus.subscribe(StateChanged)
    orch, task = start_with_bridge(bus, FakeBridge(behavior="raise"))
    try:
        await next_event(states)  # → standby
        await bus.publish(WakeRequested(reason="speech"))
        await next_event(states)  # → active
        # The crash becomes ErrorRaised(voice.session): close, error, recover.
        assert (await next_event(states)).new == "standby"
        assert (await next_event(states)).new == "error"
        assert (await next_event(states)).new == "standby"
    finally:
        await cancel(task)


async def test_bridge_ending_on_its_own_closes_the_interaction(bus):
    states = bus.subscribe(StateChanged)
    ended = bus.subscribe(InteractionEnded)
    orch, task = start_with_bridge(bus, FakeBridge(behavior="return"))
    try:
        await next_event(states)  # → standby
        await bus.publish(WakeRequested(reason="speech"))
        await next_event(states)  # → active
        # Provider closed the stream → EndOfInteractionDetected → standby.
        assert (await next_event(states)).new == "standby"
        assert "voice session ended" in (await next_event(ended)).reason
        assert orch.state is AppState.STANDBY
    finally:
        await cancel(task)


async def test_shutdown_cancels_a_running_bridge(bus):
    states = bus.subscribe(StateChanged)
    bridge = FakeBridge()
    orch, task = start_with_bridge(bus, bridge)
    await next_event(states)  # → standby
    await bus.publish(WakeRequested(reason="speech"))
    await next_event(states)  # → active
    await asyncio.sleep(0.05)
    await bus.publish(ShutdownRequested(reason="ctrl-c"))
    assert (await next_event(states)).new == "standby"
    assert (await next_event(states)).new == "stopping"
    await asyncio.wait_for(task, 1.0)
    assert bridge.cancelled


# --- status narration (injected StatusVoice; a fake records what it played) ----


class FakeStatusVoice:
    """Records the canned lines the orchestrator asked to play, in order."""

    def __init__(self) -> None:
        self.played: list[str] = []

    async def play(self, line_id: str, lang: str | None = None) -> None:
        self.played.append(line_id)


def start_with_status(
    bus: Bus, status: FakeStatusVoice, *, connectivity_check=None
) -> tuple[Orchestrator, asyncio.Task]:
    orch = Orchestrator(
        bus, "gemini", status_voice=status, connectivity_check=connectivity_check
    )
    return orch, asyncio.create_task(orch.run())


def test_error_line_maps_where_to_clip():
    assert _error_line("voice.session") == "connection_lost"
    assert _error_line("dashboard.mic_mute") == "mic_problem"
    assert _error_line("something else") == "error_recoverable"


async def test_boot_narrates_starting_then_connection_established(bus):
    status = FakeStatusVoice()
    states = bus.subscribe(StateChanged)
    _, task = start_with_status(bus, status, connectivity_check=lambda: True)
    try:
        await next_event(states)  # → standby (boot finished)
        assert status.played == ["starting_up", "connection_established"]
    finally:
        await cancel(task)


async def test_boot_narrates_no_internet_when_unreachable(bus):
    status = FakeStatusVoice()
    states = bus.subscribe(StateChanged)
    _, task = start_with_status(bus, status, connectivity_check=lambda: False)
    try:
        await next_event(states)  # → standby
        assert status.played == ["starting_up", "no_internet"]
    finally:
        await cancel(task)


async def test_normal_end_says_going_standby(bus):
    status = FakeStatusVoice()
    states = bus.subscribe(StateChanged)
    _, task = start_with_status(bus, status)
    try:
        await next_event(states)  # → standby
        await bus.publish(WakeRequested(reason="speech"))
        await next_event(states)  # → active
        await bus.publish(EndOfInteractionDetected(reason="left frame"))
        await next_event(states)  # → standby
        await asyncio.sleep(0.05)  # let the post-transition narration run
        assert "going_standby" in status.played
    finally:
        await cancel(task)


async def test_error_close_suppresses_standby_and_narrates_problem(bus, monkeypatch):
    monkeypatch.setattr(
        "harp.orchestrator.orchestrator.backoff_seconds", lambda attempt: 0
    )
    status = FakeStatusVoice()
    states = bus.subscribe(StateChanged)
    _, task = start_with_status(bus, status)
    try:
        await next_event(states)  # → standby
        await bus.publish(WakeRequested(reason="speech"))
        await next_event(states)  # → active
        await bus.publish(ErrorRaised(where="voice", message="boom"))
        await next_event(states)  # → standby (session closed)
        await next_event(states)  # → error
        await next_event(states)  # → standby (recovered)
        await asyncio.sleep(0.05)
        # The error close must NOT double-narrate a normal standby cue; it says
        # the problem line instead (where="voice" → connection_lost).
        assert "going_standby" not in status.played
        assert "connection_lost" in status.played
    finally:
        await cancel(task)


async def test_fatal_error_narrates_error_fatal(bus):
    status = FakeStatusVoice()
    states = bus.subscribe(StateChanged)
    _, task = start_with_status(bus, status)
    await next_event(states)  # → standby
    await bus.publish(ErrorRaised(where="core", message="dead", fatal=True))
    await next_event(states)  # → stopping
    await asyncio.wait_for(task, 1.0)
    assert "error_fatal" in status.played


async def test_shutdown_narrates_shutting_down_without_standby(bus):
    status = FakeStatusVoice()
    states = bus.subscribe(StateChanged)
    _, task = start_with_status(bus, status)
    await next_event(states)  # → standby
    await bus.publish(WakeRequested(reason="speech"))
    await next_event(states)  # → active
    await bus.publish(ShutdownRequested(reason="ctrl-c"))
    await next_event(states)  # → standby (graceful close, no going_standby cue)
    await next_event(states)  # → stopping
    await asyncio.wait_for(task, 1.0)
    assert "shutting_down" in status.played
    assert "going_standby" not in status.played
