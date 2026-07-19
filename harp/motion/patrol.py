"""Reusable stall-patrol motion: drive → look-around → turn laps over BaseMotors.

Extracted from the standalone autonomous_patrol CLI so the same motion runs two
ways: `python -m harp.motion.autonomous_patrol` (laps forever, PS4 E-STOP) and
the full agent's `move_around` tool / dashboard button (bounded laps via
harp/motion/controller.py). Everything here is blocking and thread-oriented —
run it in a worker thread and trip `stop_event` to bail out. Every primitive
checks the event at 20 Hz and writes an immediate zero-speed command on the way
out; the motors' own deadman (base_motors.py) backstops anything beyond that.

Motion is timed dead reckoning (seconds-per-meter / seconds-per-90°), so laps
drift with battery level and floor surface — calibrate `sec_per_meter` and
`sec_per_90_turn` in place, and keep the boundary comfortably inside the stall.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from .base_motors import BaseMotors

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.05  # 20 Hz command refresh, well inside the 0.25s deadman
# The lifelike pauses between movements. Module constants (not params) — they
# are choreography, not calibration; tests shrink them via monkeypatch.
_STEP_SETTLE_SECONDS = 0.5  # settle after each drive step, before scanning
_SETTLE_SECONDS = 1.0   # settle after the scan recenter and the corner turn
_LOOK_SECONDS = 1.5     # pause-and-look at each end of the scanning swivel


@dataclass(frozen=True)
class PatrolParams:
    """One stall patrol's geometry and calibration (see harp.yaml `motion:`)."""

    side_length: float = 3.0     # boundary side, meters
    segments: int = 2            # intermediate stops per side
    base_speed: int = 350        # forward wheel rpm
    turn_speed: int = 300        # in-place turn rpm
    sec_per_meter: float = 3.5   # seconds to cover one meter at base_speed
    sec_per_90_turn: float = 3.8  # seconds to pivot 90 degrees at turn_speed

    def __post_init__(self) -> None:
        if self.segments < 1:
            raise ValueError("segments must be at least 1")
        if self.side_length <= 0:
            raise ValueError("side_length must be positive")

    @property
    def segment_duration(self) -> float:
        return self.side_length / self.segments * self.sec_per_meter

    @property
    def sec_per_15_deg(self) -> float:
        return self.sec_per_90_turn / 6.0

    def lap_seconds(self) -> float:
        """Rough duration of one full lap, for user/model-facing notes."""
        scan = 4 * self.sec_per_15_deg + 2 * _LOOK_SECONDS + _SETTLE_SECONDS
        side = self.segments * (self.segment_duration + _STEP_SETTLE_SECONDS + scan)
        return 4 * (side + self.sec_per_90_turn + _SETTLE_SECONDS)


def execute_command(
    motors: BaseMotors,
    left_rpm: int,
    right_rpm: int,
    duration: float,
    stop_event: threading.Event,
) -> None:
    """Feed one command to the motors at 20 Hz for `duration`. On stop, write
    an immediate zero instead of coasting until the deadman fires."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        if stop_event.is_set():
            motors.command(0, 0)
            return
        motors.command(left_rpm, right_rpm)
        time.sleep(_POLL_INTERVAL)


def humanoid_scan(motors: BaseMotors, params: PatrolParams, stop_event: threading.Event) -> None:
    """A lifelike left-to-right scanning swivel, ending recentered."""
    turn = params.turn_speed
    step = params.sec_per_15_deg
    logger.info("[Scan] Swiveling left to scan...")
    execute_command(motors, -turn, -turn, step, stop_event)
    execute_command(motors, 0, 0, _LOOK_SECONDS, stop_event)

    logger.info("[Scan] Swiveling right to scan...")
    execute_command(motors, turn, turn, step * 2, stop_event)
    execute_command(motors, 0, 0, _LOOK_SECONDS, stop_event)

    logger.info("[Scan] Recentering base...")
    execute_command(motors, -turn, -turn, step, stop_event)
    execute_command(motors, 0, 0, _SETTLE_SECONDS, stop_event)


def patrol_lap(motors: BaseMotors, params: PatrolParams, stop_event: threading.Event) -> bool:
    """One full lap of the square boundary. Returns True if it completed,
    False if `stop_event` cut it short. The right wheel is mirror-mounted, so
    forward is (+rpm, -rpm) and same-sign commands spin in place."""
    for side in range(1, 5):
        logger.info("--- Patrolling Boundary Wall %d/4 ---", side)
        for segment in range(params.segments):
            if stop_event.is_set():
                return False
            logger.info("Driving forward: Step %d/%d", segment + 1, params.segments)
            execute_command(
                motors, params.base_speed, -params.base_speed,
                params.segment_duration, stop_event,
            )
            execute_command(motors, 0, 0, _STEP_SETTLE_SECONDS, stop_event)
            humanoid_scan(motors, params, stop_event)

        if stop_event.is_set():
            return False
        logger.info("Corner reached. Turning 90 degrees...")
        execute_command(
            motors, params.turn_speed, params.turn_speed,
            params.sec_per_90_turn, stop_event,
        )
        execute_command(motors, 0, 0, _SETTLE_SECONDS, stop_event)
    return not stop_event.is_set()


def run_patrol(
    motors: BaseMotors,
    params: PatrolParams,
    stop_event: threading.Event,
    laps: int = 1,
) -> bool:
    """Run `laps` full laps (laps <= 0 = forever, the CLI's mode). Returns True
    when the requested laps completed, False when stopped early. Always leaves
    the motors on a zero-speed command."""
    lap = 0
    try:
        while not stop_event.is_set():
            lap += 1
            logger.info("--- Patrol lap %d%s ---", lap, f"/{laps}" if laps > 0 else "")
            if not patrol_lap(motors, params, stop_event):
                return False
            if laps > 0 and lap >= laps:
                return True
        return False
    finally:
        motors.command(0, 0)
