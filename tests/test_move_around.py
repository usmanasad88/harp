"""MoveAroundController: the async/threading seam between the bus and the
blocking patrol thread. What's under test is lifecycle timing — a patrol that
finishes on its own, a stop that must halt a thread mid-lap quickly, a double
start that must not open the ports twice, and a port failure that must degrade
into an error result instead of a crash. Fake motors + monkeypatched pause
constants keep it hardware-free and fast; the arithmetic of the lap itself is
not retested."""

from __future__ import annotations

import asyncio
import time

import pytest

from harp.config import MotionSettings
from harp.core.bus import Bus
from harp.core.events import ErrorRaised, MoveAroundChanged
from harp.motion import patrol
from harp.motion.controller import MoveAroundController


class FakeMotors:
    def __init__(self) -> None:
        self.commands: list[tuple[int, int]] = []
        self.stopped = False

    def start(self) -> None:
        pass

    def command(self, left: int, right: int) -> None:
        self.commands.append((left, right))

    def stop(self) -> None:
        self.stopped = True


def fast_settings() -> MotionSettings:
    """A lap that takes milliseconds (pauses are shrunk via monkeypatch)."""
    return MotionSettings(
        enabled=True, side_length=0.01, segments=1,
        sec_per_meter=0.01, sec_per_90_turn=0.006, laps=1,
    )


def slow_settings() -> MotionSettings:
    """A lap that would take minutes — for tests that must stop it mid-run."""
    return MotionSettings(enabled=True, side_length=3.0, segments=2, laps=1)


@pytest.fixture
def fast_pauses(monkeypatch):
    monkeypatch.setattr(patrol, "_POLL_INTERVAL", 0.001)
    monkeypatch.setattr(patrol, "_STEP_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(patrol, "_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(patrol, "_LOOK_SECONDS", 0.0)


async def _next_event(stream, timeout: float = 5.0):
    return await asyncio.wait_for(anext(stream), timeout=timeout)


async def test_patrol_finishes_on_its_own_and_releases_the_motors(fast_pauses):
    bus = Bus()
    motors = FakeMotors()
    controller = MoveAroundController(bus, fast_settings(), motors_factory=lambda l, r: motors)
    stream = bus.subscribe(MoveAroundChanged)

    result = await controller.start()
    assert result.get("ok") is True

    started = await _next_event(stream)
    assert started.active is True

    finished = await _next_event(stream)
    assert finished.active is False
    assert "finished" in finished.note
    # The lap ended by itself: motors zeroed + closed, controller idle again.
    assert motors.stopped is True
    assert motors.commands[-1] == (0, 0)
    assert controller.active is False


async def test_stop_halts_a_long_patrol_quickly(fast_pauses):
    bus = Bus()
    motors = FakeMotors()
    controller = MoveAroundController(bus, slow_settings(), motors_factory=lambda l, r: motors)
    stream = bus.subscribe(MoveAroundChanged)

    await controller.start()
    assert (await _next_event(stream)).active is True

    began = time.monotonic()
    result = await controller.stop(note="visitor asked")
    elapsed = time.monotonic() - began
    assert result["ok"] is True
    # The patrol thread checks the stop event at 20 Hz (0.001s here) and the
    # writer join is bounded — a "quick stop" that takes seconds is a safety bug.
    assert elapsed < 2.0

    stopped = await _next_event(stream)
    assert stopped.active is False
    assert stopped.note == "visitor asked"
    assert motors.stopped is True

    # Stopping again while idle is a harmless no-op, not an error.
    again = await controller.stop()
    assert again["note"] == "not moving"


async def test_double_start_does_not_open_the_ports_twice(fast_pauses):
    bus = Bus()
    opened = []

    def factory(left, right):
        opened.append((left, right))
        return FakeMotors()

    controller = MoveAroundController(bus, slow_settings(), motors_factory=factory)
    stream = bus.subscribe(MoveAroundChanged)

    first = await controller.start()
    second = await controller.start()
    assert first.get("ok") and second.get("ok")
    assert "already moving" in second["note"]
    assert len(opened) == 1  # the second start must not touch the hardware

    await controller.stop()
    events = [await _next_event(stream), await _next_event(stream)]
    assert [e.active for e in events] == [True, False]  # no second "started"


async def test_port_failure_degrades_to_an_error_result():
    bus = Bus()

    def factory(left, right):
        raise OSError("could not open port 'COM4'")

    controller = MoveAroundController(bus, fast_settings(), motors_factory=factory)
    stream = bus.subscribe(ErrorRaised)

    result = await controller.start()
    assert "COM4" in result["error"]
    assert controller.active is False  # a failed start leaves it startable

    error = await _next_event(stream)
    assert error.where == "motion.move_around"
