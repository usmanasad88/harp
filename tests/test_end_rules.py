"""EndOfInteractionMonitor closes a session when the person walks off: face-ID
presence goes absent and stays absent for the whole timeout while a session is
active. Driven purely through the bus with a tiny timeout so the tests stay
fast and deterministic."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from harp.core.bus import Bus
from harp.core.events import (
    EndOfInteractionDetected,
    InteractionEnded,
    InteractionStarted,
    PresenceChanged,
)
from harp.interaction.end_rules import EndOfInteractionMonitor


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
