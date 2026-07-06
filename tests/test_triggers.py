"""TriggerEngine turns a wave (GestureDetected kind='wave') into a WakeRequested
that the orchestrator honors while STANDBY. Other gestures don't wake HARP."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from harp.core.bus import Bus
from harp.core.events import GestureDetected, WakeRequested
from harp.triggers.engine import TriggerEngine


async def _start(bus: Bus) -> asyncio.Task:
    task = asyncio.create_task(TriggerEngine(bus).run())
    await asyncio.sleep(0.01)  # let run() register its bus subscription
    return task


async def _stop(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _next(stream, timeout: float = 1.0):
    return await asyncio.wait_for(anext(stream), timeout=timeout)


async def test_wave_requests_a_wake():
    bus = Bus()
    stream = bus.subscribe(WakeRequested)
    task = await _start(bus)

    await bus.publish(GestureDetected(kind="wave"))

    ev = await _next(stream)
    assert ev.reason == "wave"
    assert ev.context  # a model-facing greeting hint is delivered at wake
    await _stop(task)


async def test_non_wave_gesture_does_not_wake():
    bus = Bus()
    stream = bus.subscribe(WakeRequested)
    task = await _start(bus)

    await bus.publish(GestureDetected(kind="thumbs_up"))

    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.1)
    await _stop(task)
