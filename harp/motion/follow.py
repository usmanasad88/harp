"""Bus-wired controller for "follow me" — drive toward a person HARP knows.

The second thing the base motors can do (the first is controller.py's stall
patrol). The live model's follow_person tool (harp/motion/tools.py) lands
here; every start/stop/auto-stop is announced the same way: a FollowChanged
event on the bus, published only by this controller.

Follow ONLY works for a known person: start() refuses unless face-ID
currently recognizes someone enrolled, and the loop keeps requiring face-ID
to vouch for that same person — the moment an unrecognized face is what's in
front (the target left, or a stranger stepped closer), the wheels zero, and
if the target stays unseen past `follow_lost_seconds` follow ends by itself.

The loop itself is deliberately simple, per-frame reactive steering — no
planning, no mapping:
  - A fast YOLOv8n-face pass (face_tracker.FaceDetector, the same model the
    head tracker uses) finds the largest face in the shared camera's latest
    frame at ~5-10 Hz.
  - Turn: if the face center leaves the central box (±follow_center_frac of
    the frame width), spin in place toward it. Same sign convention as
    patrol.py: forward is (+rpm, -rpm), same-sign spins.
  - Drive: the shared camera has no depth stream, so face-box HEIGHT is the
    distance proxy. Below follow_far_frac of the frame → too far, drive
    forward; at/above follow_near_frac → close enough, stop. The wide band
    between the two is hysteresis: keep doing whatever you were doing, so the
    base doesn't hunt back and forth around a single threshold.

Safety layers, outermost first: face-ID must keep vouching for the target or
the wheels zero within one pass; `stop()` (checked every loop tick, zeroes on
the way out); the lost-timeout ending follow on its own; and the motors' own
0.25s deadman if this process dies mid-drive. Ports are opened at start and
released at stop, exactly like the patrol controller.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from ..config import MotionSettings
from ..core.bus import Bus
from ..core.events import ErrorRaised, FollowChanged, PersonIdentified
from .base_motors import BaseMotors

logger = logging.getLogger(__name__)

# Loop cadence: one detector pass + one motor command per tick. Detection is
# the slow part (~50-100 ms of CPU); the motors' 0.25s deadman backstops a
# pass that runs long by briefly zeroing the wheels — a pause, never a runaway.
_POLL_INTERVAL = 0.1


def _default_detector() -> Any:
    """Lazy so importing this module never needs cv2/onnxruntime (tests, and
    machines without the vision stack, inject a fake instead)."""
    from .face_tracker import FaceDetector

    return FaceDetector()


def steer(
    box: list[int] | tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    settings: MotionSettings,
    driving: bool,
) -> tuple[int, int, bool]:
    """One face box → (left_rpm, right_rpm, driving').

    Pure on purpose — this is the whole follow policy, and the sign
    conventions (which way is "turn right", which pair is "forward") are
    exactly the kind of silent bug a test should pin down. `driving` is the
    hysteresis state: whether the previous tick was driving forward.
    """
    x1, y1, x2, y2 = box
    off = ((x1 + x2) / 2 - frame_w / 2) / frame_w  # -0.5 (left) .. +0.5 (right)
    height_frac = (y2 - y1) / frame_h

    # Recenter first: while the face is outside the central box, only turn.
    # patrol.py's convention: same-sign spins in place, (+,+) toward the right.
    if abs(off) > settings.follow_center_frac:
        turn = settings.follow_turn_speed if off > 0 else -settings.follow_turn_speed
        return turn, turn, False

    # Centered: drive by the face-size distance proxy, with a wide hysteresis
    # band (far_frac .. near_frac) where the previous decision stands.
    if height_frac < settings.follow_far_frac:
        driving = True
    elif height_frac >= settings.follow_near_frac:
        driving = False
    if driving:
        return settings.follow_speed, -settings.follow_speed, True
    return 0, 0, False


class FollowController:
    """Start/stop follow mode; publish FollowChanged for each change.

    Vision comes in as callables so this stays testable without hardware:
    `latest_frame` is the shared camera's newest frame, `current_person` is
    face-ID's most prominent identity (used to pick + gate the target), and
    `person_in_front(person_id)` is "does face-ID currently say THIS person is
    the most prominent face". `announce(moment)` fires a canned status clip
    ("started" / "no_person" / "stopped" — resolved through the status rule
    book by the caller); `conflict()` returns a reason to refuse starting
    (the patrol holding the motors) or None.
    """

    def __init__(
        self,
        bus: Bus,
        settings: MotionSettings,
        *,
        latest_frame: Callable[[], np.ndarray | None],
        current_person: Callable[[], PersonIdentified | None],
        person_in_front: Callable[[str], bool],
        motors_factory: Callable[[str, str], Any] = BaseMotors,
        detector_factory: Callable[[], Any] = _default_detector,
        announce: Callable[[str], None] | None = None,
        conflict: Callable[[], str | None] | None = None,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._latest_frame = latest_frame
        self._current_person = current_person
        self._person_in_front = person_in_front
        self._motors_factory = motors_factory
        self._detector_factory = detector_factory
        self._announce = announce or (lambda moment: None)
        self._conflict = conflict
        self._detector: Any = None  # built on first start, reused after
        # Serializes start/stop, same as the patrol controller.
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stop_event: threading.Event | None = None
        self._stop_note = ""
        self._target: PersonIdentified | None = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def snapshot(self) -> dict:
        """Seed for a fresh dashboard connection (the bus never replays)."""
        person = self._target if self.active else None
        return {
            "active": self.active,
            "person": (person.name or person.person_id) if person else "",
        }

    async def start(self) -> dict:
        """Begin following the known person face-ID sees right now. Returns a
        small result dict for the model: {"ok", "note"} or {"error"}."""
        async with self._lock:
            if self.active:
                assert self._target is not None
                return {"ok": True, "note": f"already following {self._label()}"}
            if self._conflict is not None:
                reason = self._conflict()
                if reason:
                    return {"error": f"can't follow right now: {reason}"}
            person = self._current_person()
            if person is None or not person.is_known:
                self._announce("no_person")
                return {
                    "error": (
                        "no known person in frame — following only works for "
                        "people I recognize"
                    )
                }
            try:
                # Serial ports + (first time) the ONNX session both block; off
                # the loop.
                motors = await asyncio.to_thread(self._open_motors_and_detector)
            except Exception as exc:
                message = f"could not start following: {exc}"
                logger.warning("follow: %s", message)
                await self._bus.publish(ErrorRaised(where="motion.follow", message=message))
                return {"error": message}
            self._target = person
            self._stop_note = ""
            self._stop_event = threading.Event()
            self._task = asyncio.create_task(
                self._run(motors, self._stop_event), name="follow-person"
            )
            await self._bus.publish(
                FollowChanged(active=True, person=self._label(), note="following started")
            )
            self._announce("started")
            return {
                "ok": True,
                "note": (
                    f"Now following {self._label()}. I stop when they ask, when "
                    "I lose sight of them, or on action 'stop'."
                ),
            }

    async def stop(self, note: str = "stopped") -> dict:
        """Halt the follow loop and wait for the motors to zero and the ports
        to close. Idempotent — stopping while idle is fine."""
        async with self._lock:
            task, stop_event = self._task, self._stop_event
            if task is None or task.done() or stop_event is None:
                return {"ok": True, "note": "not following"}
            self._stop_note = note
            stop_event.set()
            await task  # _run zeroes the motors, closes ports, publishes
            return {"ok": True, "note": "stopped following"}

    def _label(self) -> str:
        assert self._target is not None
        return self._target.name or self._target.person_id

    def _open_motors_and_detector(self) -> Any:
        if self._detector is None:
            self._detector = self._detector_factory()
        motors = self._motors_factory(self._settings.left_port, self._settings.right_port)
        motors.start()
        return motors

    async def _run(self, motors: Any, stop_event: threading.Event) -> None:
        note = "stopped following"
        try:
            reason = await asyncio.to_thread(self._follow_loop, motors, stop_event)
            if reason == "lost":
                note = f"lost sight of {self._label()} — follow ended"
            elif self._stop_note:
                note = self._stop_note
        except Exception as exc:
            note = f"follow failed: {exc}"
            logger.exception("follow: loop crashed")
            await self._bus.publish(ErrorRaised(where="motion.follow", message=note))
        finally:
            # Runs even if this task is cancelled at app shutdown: stop() joins
            # the writer thread, zeroes both wheels, and closes the ports.
            stop_event.set()
            await asyncio.to_thread(motors.stop)
            self._announce("stopped")
            await self._bus.publish(
                FollowChanged(active=False, person=self._label(), note=note)
            )

    def _follow_loop(self, motors: Any, stop_event: threading.Event) -> str:
        """Blocking; runs in a worker thread. Returns why it ended: "stopped"
        (stop_event tripped) or "lost" (target unseen past the timeout)."""
        assert self._target is not None
        target_id = self._target.person_id
        lost_after = self._settings.follow_lost_seconds
        last_confirmed = time.monotonic()
        driving = False
        try:
            while not stop_event.is_set():
                now = time.monotonic()
                # Face-ID must keep vouching that the person in front IS the
                # target (its own ~1.5s cadence). A stranger stepping closer or
                # the target walking off both drop this to False within a pass.
                confirmed = self._person_in_front(target_id)
                if confirmed:
                    last_confirmed = now
                elif now - last_confirmed > lost_after:
                    motors.command(0, 0)
                    return "lost"

                frame = self._latest_frame()
                box = self._largest_face(frame) if confirmed and frame is not None else None
                if box is None:
                    driving = False
                    motors.command(0, 0)
                else:
                    h, w = frame.shape[:2]
                    left, right, driving = steer(box, w, h, self._settings, driving)
                    motors.command(left, right)
                stop_event.wait(_POLL_INTERVAL)
            return "stopped"
        finally:
            motors.command(0, 0)

    def _largest_face(self, frame: np.ndarray) -> list[int] | None:
        boxes = self._detector.detect(frame)
        if not boxes:
            return None
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
