"""Bus-wired controller for the "move around" stall patrol.

The owner of the base motors while a patrol runs (follow mode, follow.py, is
the other motor user — the two refuse to start over each other via the
`conflict` callbacks app.py wires between them). Both entry points —
the live model's move_around tool (harp/motion/tools.py) and the dashboard's
Move around button (SetMoveAround over /ws) — land on this controller, so a
patrol can never be started twice and everyone learns of changes the same way:
a MoveAroundChanged event on the bus, published only here (start, stop, the
bounded lap finishing on its own, or a failure).

The patrol itself is the blocking drive → look-around → turn lap from
patrol.py, run in a worker thread via asyncio.to_thread. The serial ports are
opened when a patrol starts and released when it ends, so the standalone
motion CLIs (`python -m harp.motion`, autonomous_patrol) can use the same
adapters whenever a patrol isn't running. Safety layers, outermost first: the
bounded lap count, `stop()` (checked at 20 Hz, zeroes on the way out), and the
motors' own 0.25s deadman if this whole process dies mid-lap.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable

from ..config import MotionSettings
from ..core.bus import Bus
from ..core.events import ErrorRaised, MoveAroundChanged
from .base_motors import BaseMotors
from .patrol import PatrolParams, run_patrol

logger = logging.getLogger(__name__)


class MoveAroundController:
    """Start/stop the bounded patrol; publish MoveAroundChanged for each change.

    `motors_factory(left_port, right_port)` exists for tests — production uses
    BaseMotors. All public methods are coroutines called from the event loop;
    only the patrol itself runs off-loop.
    """

    def __init__(
        self,
        bus: Bus,
        settings: MotionSettings,
        motors_factory: Callable[[str, str], Any] = BaseMotors,
        conflict: Callable[[], str | None] | None = None,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._motors_factory = motors_factory
        # Something else holding the motors (follow mode) — a reason string
        # refuses start() cleanly instead of failing on the busy serial ports.
        self._conflict = conflict
        self._params = PatrolParams(
            side_length=settings.side_length,
            segments=settings.segments,
            base_speed=settings.base_speed,
            turn_speed=settings.turn_speed,
            sec_per_meter=settings.sec_per_meter,
            sec_per_90_turn=settings.sec_per_90_turn,
        )
        self._laps = max(1, settings.laps)
        # Serializes start/stop so a tool call and a dashboard click arriving
        # together can't both open the ports.
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stop_event: threading.Event | None = None
        self._stop_note = ""

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def snapshot(self) -> dict:
        """Seed for a fresh dashboard connection (the bus never replays)."""
        return {"active": self.active}

    async def set_active(self, active: bool) -> dict:
        """The dashboard's entry point: True = start, False = stop."""
        if active:
            return await self.start()
        return await self.stop(note="stopped from the dashboard")

    async def start(self) -> dict:
        """Open the motors and launch the bounded patrol. Returns a small
        result dict for the model / dashboard: {"ok", "note"} or {"error"}."""
        async with self._lock:
            if self.active:
                return {"ok": True, "note": "already moving — the patrol is running"}
            if self._conflict is not None:
                reason = self._conflict()
                if reason:
                    return {"error": f"can't move around right now: {reason}"}
            try:
                # Opening two serial ports blocks briefly; off the loop.
                motors = await asyncio.to_thread(self._open_motors)
            except Exception as exc:
                message = f"could not open the base motor ports: {exc}"
                logger.warning("move_around: %s", message)
                await self._bus.publish(
                    ErrorRaised(where="motion.move_around", message=message)
                )
                return {"error": message}
            self._stop_note = ""
            self._stop_event = threading.Event()
            self._task = asyncio.create_task(
                self._run(motors, self._stop_event), name="move-around-patrol"
            )
            await self._bus.publish(MoveAroundChanged(active=True, note="patrol started"))
            estimate = int(self._params.lap_seconds() * self._laps)
            return {
                "ok": True,
                "note": (
                    f"Moving now: {self._laps} patrol lap(s), roughly {estimate} "
                    "seconds, then I stop on my own. Call with action 'stop' to "
                    "stop early."
                ),
            }

    async def stop(self, note: str = "stopped") -> dict:
        """Halt the patrol (checked at 20 Hz) and wait for the motors to zero
        and the ports to close. Idempotent — stopping while idle is fine."""
        async with self._lock:
            task, stop_event = self._task, self._stop_event
            if task is None or task.done() or stop_event is None:
                return {"ok": True, "note": "not moving"}
            self._stop_note = note
            stop_event.set()
            await task  # _run zeroes the motors, closes ports, publishes
            return {"ok": True, "note": "stopped moving"}

    def _open_motors(self) -> Any:
        motors = self._motors_factory(self._settings.left_port, self._settings.right_port)
        motors.start()
        return motors

    async def _run(self, motors: Any, stop_event: threading.Event) -> None:
        note = "finished the patrol"
        try:
            completed = await asyncio.to_thread(
                run_patrol, motors, self._params, stop_event, self._laps
            )
            if not completed:
                note = self._stop_note or "stopped"
        except Exception as exc:
            note = f"patrol failed: {exc}"
            logger.exception("move_around: patrol crashed")
            await self._bus.publish(ErrorRaised(where="motion.move_around", message=note))
        finally:
            # Runs even if this task is cancelled at app shutdown: stop() joins
            # the writer thread, zeroes both wheels, and closes the ports.
            stop_event.set()
            await asyncio.to_thread(motors.stop)
            await self._bus.publish(MoveAroundChanged(active=False, note=note))
