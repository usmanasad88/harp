"""EndOfInteractionMonitor closes a session when the person walks off: face-ID
presence goes absent and stays absent for the whole timeout while a session is
active. Driven purely through the bus with a tiny timeout so the tests stay
fast and deterministic."""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from harp.core.bus import Bus
from harp.core.events import (
    AgentSaid,
    EndOfInteractionDetected,
    InteractionEnded,
    InteractionStarted,
    PresenceChanged,
    TalkKeyChanged,
    UserSaid,
)
from harp.interaction.end_rules import EndOfInteractionMonitor, SilenceMonitor


async def _start(monitor) -> asyncio.Task:
    task = asyncio.create_task(monitor.run())
    await asyncio.sleep(0.01)  # let run() register its bus subscription
    return task


async def _stop(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _next(stream, timeout: float = 1.0):
    return await asyncio.wait_for(anext(stream), timeout=timeout)


async def test_absent_for_the_timeout_ends_the_session():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(EndOfInteractionMonitor(bus, absence_timeout=0.05))

    await bus.publish(InteractionStarted(reason="wave"))
    await bus.publish(PresenceChanged(present=False, count=0))

    ev = await _next(stream)
    assert isinstance(ev, EndOfInteractionDetected)
    assert "no face" in ev.reason
    await _stop(task)


async def test_session_opening_with_nobody_in_frame_closes():
    # The gap that kept sessions open forever: woken with no face in frame, no
    # PresenceChanged ever arrives, so presence is seeded from is_present().
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    monitor = EndOfInteractionMonitor(bus, absence_timeout=0.05, is_present=lambda: False)
    task = await _start(monitor)

    await bus.publish(InteractionStarted(reason="wake word"))  # no PresenceChanged

    ev = await _next(stream)
    assert isinstance(ev, EndOfInteractionDetected)
    await _stop(task)


async def test_session_opening_with_a_face_present_stays_open():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    monitor = EndOfInteractionMonitor(bus, absence_timeout=0.05, is_present=lambda: True)
    task = await _start(monitor)

    await bus.publish(InteractionStarted(reason="wave"))  # someone is in frame

    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.15)
    await _stop(task)


async def test_face_returning_before_timeout_keeps_the_session():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(EndOfInteractionMonitor(bus, absence_timeout=0.1))

    await bus.publish(InteractionStarted(reason="wave"))
    await bus.publish(PresenceChanged(present=False, count=0))
    await asyncio.sleep(0.03)  # well under the timeout
    await bus.publish(PresenceChanged(present=True, count=1))  # they came back

    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.15)
    await _stop(task)


async def test_absence_while_idle_does_not_end_anything():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(EndOfInteractionMonitor(bus, absence_timeout=0.05))

    # No session open: an empty frame must not trigger anything.
    await bus.publish(PresenceChanged(present=False, count=0))

    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.15)
    await _stop(task)


async def test_interaction_ending_cancels_a_pending_countdown():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(EndOfInteractionMonitor(bus, absence_timeout=0.1))

    await bus.publish(InteractionStarted(reason="wave"))
    await bus.publish(PresenceChanged(present=False, count=0))  # arm the countdown
    await asyncio.sleep(0.03)
    await bus.publish(InteractionEnded(reason="provider closed"))  # already over

    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.15)
    await _stop(task)


async def test_leaving_again_starts_a_fresh_full_countdown():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(EndOfInteractionMonitor(bus, absence_timeout=0.1))

    await bus.publish(InteractionStarted(reason="wave"))
    await bus.publish(PresenceChanged(present=False, count=0))  # leave
    await asyncio.sleep(0.05)
    await bus.publish(PresenceChanged(present=True, count=1))  # back (disarm)
    await bus.publish(PresenceChanged(present=False, count=0))  # leave again

    ev = await _next(stream, timeout=1.0)
    assert isinstance(ev, EndOfInteractionDetected)
    await _stop(task)


# --- SilenceMonitor: the "nobody is saying anything" rule ----------------------
# The hole it plugs: a person standing in frame who never talks keeps face-
# presence alive forever, so only a quiet-time countdown can close the session.


async def test_a_session_where_nothing_is_said_closes():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(SilenceMonitor(bus, silence_timeout=0.05))

    await bus.publish(InteractionStarted(reason="button"))  # ...and then nothing

    ev = await _next(stream)
    assert ev.cause == "silence"  # the rule book narrates this as a goodbye
    await _stop(task)


async def test_conversation_activity_restarts_the_countdown():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(SilenceMonitor(bus, silence_timeout=0.08))

    await bus.publish(InteractionStarted(reason="button"))
    # Keep the conversation alive past several would-be timeouts, alternating
    # every signal that counts as activity (streaming fragments included).
    for ev in (
        UserSaid(text="hel", final=False),
        AgentSaid(text="hi there", final=True),
        TalkKeyChanged(held=True),
    ):
        await asyncio.sleep(0.05)  # well under the timeout, but sums past it
        await bus.publish(ev)

    # If a countdown had (wrongly) expired mid-conversation, its event is
    # already queued and _next returns instantly. The genuine close can only
    # arrive one FULL timeout after the last activity — so measure it.
    t0 = time.monotonic()
    ev = await _next(stream, timeout=1.0)  # went quiet for good → closes
    assert ev.cause == "silence"
    assert time.monotonic() - t0 >= 0.06  # a fresh countdown, not a stale fire
    await _stop(task)


async def test_interaction_ending_disarms_the_silence_countdown():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(SilenceMonitor(bus, silence_timeout=0.05))

    await bus.publish(InteractionStarted(reason="wave"))
    await bus.publish(InteractionEnded(reason="agent hung up"))  # closed early

    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.15)
    await _stop(task)


async def test_speech_while_idle_never_arms_the_silence_rule():
    bus = Bus()
    stream = bus.subscribe(EndOfInteractionDetected)
    task = await _start(SilenceMonitor(bus, silence_timeout=0.05))

    # No session open: overheard speech / button noise must not start anything.
    await bus.publish(UserSaid(text="just chatting nearby", final=True))
    await bus.publish(TalkKeyChanged(held=True))

    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.15)
    await _stop(task)
