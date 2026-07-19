"""FollowController: the safety-bearing seams of follow mode. What's under
test: the steer() sign conventions (which pair is "forward", which sign turns
which way — a silent bug here drives the robot AWAY from the person) with the
hysteresis band; the known-person gate (follow must refuse strangers and an
empty frame without ever touching the ports); losing the person ending follow
on its own; and a stop that must halt the loop quickly. Fake motors, frames,
and detector keep it hardware-free."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from harp.config import MotionSettings
from harp.core.bus import Bus
from harp.core.events import FollowChanged, PersonIdentified
from harp.motion import follow as follow_mod
from harp.motion.controller import MoveAroundController
from harp.motion.follow import FollowController, steer


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


class FakeDetector:
    """Returns whatever boxes the test currently wants detected."""

    def __init__(self, boxes) -> None:
        self.boxes = boxes

    def detect(self, frame):
        return list(self.boxes)


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
KNOWN = PersonIdentified(person_id="usman", name="Usman", is_known=True)

# Face boxes in the 640x480 frame, named for what steer should do with them.
FAR_CENTERED = [300, 200, 340, 248]      # height 48 = 0.10 < far_frac → drive
NEAR_CENTERED = [280, 160, 360, 288]     # height 128 = 0.27 ≥ near_frac → stop
MID_CENTERED = [300, 180, 340, 266]      # height 86 = 0.18, inside the band
RIGHT_OF_CENTER = [560, 200, 600, 248]   # cx 580 → far right of the frame
LEFT_OF_CENTER = [40, 200, 80, 248]      # cx 60 → far left


def settings(**overrides) -> MotionSettings:
    return MotionSettings(enabled=True, **overrides)


def test_steer_signs_and_hysteresis():
    s = settings()
    # Forward is (+, -): the right wheel is mirror-mounted (see patrol.py).
    assert steer(FAR_CENTERED, 640, 480, s, driving=False) == (
        s.follow_speed, -s.follow_speed, True,
    )
    # Close enough: full stop, and the driving latch clears.
    assert steer(NEAR_CENTERED, 640, 480, s, driving=True) == (0, 0, False)
    # Inside the far/near band the previous decision stands — no hunting.
    assert steer(MID_CENTERED, 640, 480, s, driving=True)[2] is True
    assert steer(MID_CENTERED, 640, 480, s, driving=False) == (0, 0, False)
    # Off-center overrides distance: same-sign spins, (+, +) toward the right.
    assert steer(RIGHT_OF_CENTER, 640, 480, s, driving=True) == (
        s.follow_turn_speed, s.follow_turn_speed, False,
    )
    assert steer(LEFT_OF_CENTER, 640, 480, s, driving=False) == (
        -s.follow_turn_speed, -s.follow_turn_speed, False,
    )


def make_controller(
    bus: Bus,
    motors: FakeMotors,
    *,
    current_person=lambda: KNOWN,
    person_in_front=lambda pid: pid == KNOWN.person_id,
    boxes=None,
    announce=None,
    conflict=None,
    **setting_overrides,
) -> FollowController:
    detector = FakeDetector([FAR_CENTERED] if boxes is None else boxes)
    return FollowController(
        bus,
        settings(**setting_overrides),
        latest_frame=lambda: FRAME,
        current_person=current_person,
        person_in_front=person_in_front,
        motors_factory=lambda left, right: motors,
        detector_factory=lambda: detector,
        announce=announce,
        conflict=conflict,
    )


async def _next_event(stream, timeout: float = 5.0):
    return await asyncio.wait_for(anext(stream), timeout=timeout)


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout=timeout)


@pytest.fixture
def fast_loop(monkeypatch):
    monkeypatch.setattr(follow_mod, "_POLL_INTERVAL", 0.005)


async def test_start_refuses_strangers_and_empty_frames():
    bus = Bus()
    announced: list[str] = []
    opened: list[FakeMotors] = []

    def factory(left, right):  # pragma: no cover - must never run
        opened.append(FakeMotors())
        return opened[-1]

    for person in (None, PersonIdentified(person_id="unknown", is_known=False)):
        controller = FollowController(
            bus,
            settings(),
            latest_frame=lambda: FRAME,
            current_person=lambda: person,
            person_in_front=lambda pid: False,
            motors_factory=factory,
            detector_factory=lambda: FakeDetector([]),
            announce=announced.append,
        )
        result = await controller.start()
        assert "no known person" in result["error"]
        assert controller.active is False

    # Both refusals played the "no known person" clip; the ports were never
    # touched — a refused follow must not even open the motors.
    assert announced == ["no_person", "no_person"]
    assert opened == []


async def test_follows_then_ends_on_its_own_when_the_person_is_lost(fast_loop):
    bus = Bus()
    motors = FakeMotors()
    announced: list[str] = []
    visible = True
    controller = make_controller(
        bus,
        motors,
        person_in_front=lambda pid: visible and pid == KNOWN.person_id,
        announce=announced.append,
        follow_lost_seconds=0.1,
    )
    stream = bus.subscribe(FollowChanged)

    result = await controller.start()
    assert result.get("ok") is True
    started = await _next_event(stream)
    assert started.active is True
    assert started.person == "Usman"
    assert announced == ["started"]

    # A far, centered face while face-ID vouches for the target → forward.
    forward = (settings().follow_speed, -settings().follow_speed)
    await _wait_for(lambda: forward in motors.commands)

    # Face-ID stops seeing Usman: wheels must zero at once (never steer toward
    # an unvouched face), and past follow_lost_seconds follow ends by itself.
    visible = False
    ended = await _next_event(stream)
    assert ended.active is False
    assert "lost sight of Usman" in ended.note
    assert motors.stopped is True
    assert motors.commands[-1] == (0, 0)
    assert controller.active is False
    assert announced == ["started", "stopped"]


async def test_stop_halts_follow_quickly(fast_loop):
    bus = Bus()
    motors = FakeMotors()
    controller = make_controller(bus, motors)
    stream = bus.subscribe(FollowChanged)

    await controller.start()
    assert (await _next_event(stream)).active is True

    began = time.monotonic()
    result = await controller.stop(note="visitor asked")
    elapsed = time.monotonic() - began
    assert result["ok"] is True
    # The loop checks the stop event every poll tick and the motor writer join
    # is bounded — a stop that takes seconds is a safety bug.
    assert elapsed < 2.0

    stopped = await _next_event(stream)
    assert stopped.active is False
    assert stopped.note == "visitor asked"
    assert motors.stopped is True

    # Stopping again while idle is a harmless no-op, not an error.
    again = await controller.stop()
    assert again["note"] == "not following"


async def test_follow_and_patrol_refuse_to_hold_the_motors_together(fast_loop):
    # Both directions of the app.py conflict wiring: the guard is what keeps
    # two independent controllers off the same serial ports.
    bus = Bus()
    follow_motors = FakeMotors()
    patrol_active = False
    controller = make_controller(
        bus,
        follow_motors,
        conflict=lambda: "the patrol is running" if patrol_active else None,
    )

    patrol_active = True
    refused = await controller.start()
    assert "can't follow right now" in refused["error"]

    patrol_active = False
    assert (await controller.start()).get("ok") is True

    patrol = MoveAroundController(
        bus,
        settings(),
        motors_factory=lambda left, right: FakeMotors(),
        conflict=lambda: "follow mode is on" if controller.active else None,
    )
    refused = await patrol.start()
    assert "can't move around right now" in refused["error"]

    await controller.stop()
    assert controller.active is False
