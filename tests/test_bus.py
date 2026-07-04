"""Bus is the spine every subsystem depends on: test it in total isolation,
with no real subsystem involved — just events in, events out."""

from __future__ import annotations

import asyncio

import pytest

from harp.core.bus import Bus
from harp.core.events import Event, Heartbeat, PresenceChanged, UserSaid


async def _next(stream) -> Event:
    """Pull one event with a timeout so a bug can't hang the test suite."""
    return await asyncio.wait_for(anext(stream), timeout=1)


async def test_subscriber_receives_published_event():
    bus = Bus()
    stream = bus.subscribe()

    await bus.publish(PresenceChanged(present=True, count=1))

    event = await _next(stream)
    assert event == PresenceChanged(present=True, count=1)


async def test_subscribe_filters_by_type():
    bus = Bus()
    stream = bus.subscribe(UserSaid)

    await bus.publish(PresenceChanged(present=True))
    await bus.publish(UserSaid(text="hello", final=True))

    event = await _next(stream)
    assert isinstance(event, UserSaid)
    assert event.text == "hello"


async def test_each_subscriber_gets_its_own_copy():
    bus = Bus()
    a = bus.subscribe()
    b = bus.subscribe()

    await bus.publish(Heartbeat(ts=1.0))

    assert await _next(a) == Heartbeat(ts=1.0)
    assert await _next(b) == Heartbeat(ts=1.0)


async def test_publish_never_blocks_on_a_slow_subscriber():
    bus = Bus()
    slow = bus.subscribe()  # never read from

    for i in range(200):  # far past the queue's cap
        await asyncio.wait_for(bus.publish(Heartbeat(ts=float(i))), timeout=1)

    # The slow subscriber lost old events (drop-oldest), but the bus is alive
    # and a fresh subscriber still sees new events normally.
    fresh = bus.subscribe()
    await bus.publish(Heartbeat(ts=999.0))
    assert await _next(fresh) == Heartbeat(ts=999.0)
    del slow


async def test_closing_a_stream_removes_the_subscriber():
    bus = Bus()
    stream = bus.subscribe()
    await bus.publish(Heartbeat(ts=1.0))
    await _next(stream)  # the generator must have actually started for
    # aclose() to run its cleanup — see docstring note in core/bus.py

    await stream.aclose()

    # Publishing with no listeners left must not raise, and the bus must not
    # keep a dangling reference to the closed queue.
    await bus.publish(Heartbeat(ts=2.0))
    assert len(bus._subscribers) == 0
