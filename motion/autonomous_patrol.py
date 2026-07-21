#!/usr/bin/env python3
"""HARP autonomous perimeter patrol — square-boundary drive + scan, with a PS4
controller as a live pause/resume E-STOP (not a hard kill).

This is a port of the original patrol script with one behavior change: ANY
controller button press used to end the patrol thread permanently (you had to
re-run the script to patrol again). Now it PAUSES — motors zero immediately,
in-progress motion is abandoned safely — and patrol resumes from the exact
step (side, segment) it was on when Triangle is pressed. Any OTHER button
re-pauses instantly (so a stray Cross/Circle/Square/D-pad press during a
resume can't run the robot into someone). Ctrl+C always fully quits.

    uv run python -m harp.motion.autonomous_patrol --left-port COM3 --right-port COM4

Buttons (same DualShock 4 layout as teleop_ps5.py — HIDAPI, no hat, verified
in this repo's own controller testing):
  Any button      -> PAUSE (freeze in place; safe to walk up and interact)
  Triangle        -> RESUME (continues the SAME patrol step, not from scratch)
  Ctrl+C          -> full stop, process exits

While paused OR resumed, `patrol_state.set_active(True)` stays set the whole
time the script is running (voice wakes stay suppressed) — the distinction
between "patrolling" and "paused for a visitor" is a separate flag
(`patrol_state.set_paused`, this module) so a future revision of the voice
side can tell them apart if useful; today the orchestrator only reads
`active`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

# Hide pygame support prompt
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from . import patrol_state
from .base_motors import BaseMotors

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("autonomous_patrol")

_POLL_INTERVAL = 0.05  # 20 Hz (well inside the motors' 0.25s deadman window)

# DualShock 4 over SDL's HIDAPI driver (this repo's default — see
# teleop_ps5.py's _HIDAPI layout): no hat, Cross=0 Circle=1 Square=2,
# Triangle=3. Kept as a local constant rather than importing teleop_ps5's
# dataclass, since this script only needs the one button.
_TRIANGLE = 3


def _pausable_sleep(seconds: float, paused: threading.Event, stop: threading.Event) -> bool:
    """Sleep up to `seconds`, waking early on pause or stop.

    Returns True if it completed the full sleep undisturbed; False if it was
    cut short (caller should re-check `paused`/`stop` before continuing).
    """
    end = time.time() + seconds
    while time.time() < end:
        if stop.is_set() or paused.is_set():
            return False
        time.sleep(min(_POLL_INTERVAL, end - time.time()))
    return True


def execute_command(
    motors: BaseMotors,
    left_rpm: int,
    right_rpm: int,
    duration: float,
    paused: threading.Event,
    stop: threading.Event,
) -> bool:
    """Feed one command at 20 Hz for `duration`, or until paused/stopped.

    Returns True if the full duration ran; False if interrupted (paused or
    stopped) partway — the caller should treat the current step as unfinished
    and re-run it (from the top) once resumed, rather than continue as if it
    completed.
    """
    start_time = time.time()
    while time.time() - start_time < duration:
        if stop.is_set() or paused.is_set():
            motors.command(0, 0)
            return False
        motors.command(left_rpm, right_rpm)
        time.sleep(_POLL_INTERVAL)
    return True


def humanoid_scan(
    motors: BaseMotors,
    turn_speed: int,
    sec_per_15_deg: float,
    paused: threading.Event,
    stop: threading.Event,
) -> bool:
    """Lifelike left-to-right scanning swivel. Returns False if interrupted."""
    logger.info("[Scan] Swiveling left to scan...")
    if not execute_command(motors, -turn_speed, -turn_speed, sec_per_15_deg, paused, stop):
        return False
    if not execute_command(motors, 0, 0, 1.5, paused, stop):
        return False

    logger.info("[Scan] Swiveling right to scan...")
    if not execute_command(motors, turn_speed, turn_speed, sec_per_15_deg * 2, paused, stop):
        return False
    if not execute_command(motors, 0, 0, 1.5, paused, stop):
        return False

    logger.info("[Scan] Recentering base...")
    if not execute_command(motors, -turn_speed, -turn_speed, sec_per_15_deg, paused, stop):
        return False
    return execute_command(motors, 0, 0, 1.0, paused, stop)


def patrol_loop(
    motors: BaseMotors, args: argparse.Namespace, paused: threading.Event, stop: threading.Event
) -> None:
    """The perimeter patrol loop (background thread). Blocks on `paused`
    between steps rather than exiting, so RESUME continues the same side and
    segment instead of restarting the square from side 1."""
    segment_distance = args.side_length / args.segments
    segment_duration = segment_distance * args.sec_per_meter
    sec_per_15_deg = args.sec_per_90_turn / 6.0

    logger.info("--- AUTONOMOUS PATROL STARTED ---")
    logger.info(f"Target Boundary: {args.side_length}m x {args.side_length}m")
    logger.info("Press ANY button to PAUSE. Press TRIANGLE to RESUME. Ctrl+C to quit.")

    try:
        while not stop.is_set():
            for side in range(1, 5):
                if stop.is_set():
                    return
                logger.info(f"--- Patrolling Boundary Wall {side}/4 ---")

                for segment in range(args.segments):
                    if stop.is_set():
                        return
                    _wait_while_paused(paused, stop)
                    if stop.is_set():
                        return

                    logger.info(f"Driving forward: Step {segment + 1}/{args.segments}")
                    if not execute_command(
                        motors, args.base_speed, -args.base_speed, segment_duration, paused, stop
                    ):
                        continue  # paused mid-step — _wait_while_paused re-runs it

                    if not execute_command(motors, 0, 0, 0.5, paused, stop):
                        continue
                    if not humanoid_scan(motors, args.turn_speed, sec_per_15_deg, paused, stop):
                        continue

                if stop.is_set():
                    return
                _wait_while_paused(paused, stop)
                if stop.is_set():
                    return
                logger.info("Corner reached. Turning 90 degrees...")
                if not execute_command(
                    motors, args.turn_speed, args.turn_speed, args.sec_per_90_turn, paused, stop
                ):
                    continue
                execute_command(motors, 0, 0, 1.0, paused, stop)

    except Exception:
        logger.exception("Error during patrol")


def _wait_while_paused(paused: threading.Event, stop: threading.Event) -> None:
    """Block here (motors already zeroed by whoever set `paused`) until
    RESUME clears it, or the script is stopping."""
    if not paused.is_set():
        return
    patrol_state.set_paused(True)
    logger.info("[PAUSED] Motors stopped. Press TRIANGLE to resume.")
    while paused.is_set() and not stop.is_set():
        time.sleep(_POLL_INTERVAL)
    if not stop.is_set():
        patrol_state.set_paused(False)
        logger.info("[RESUMED] Continuing patrol.")


def main() -> None:
    parser = argparse.ArgumentParser(description="HARP autonomous exhibition patrol.")
    parser.add_argument("--left-port", required=True, help="Left motor port")
    parser.add_argument("--right-port", required=True, help="Right motor port")
    parser.add_argument("--base-speed", type=int, default=100, help="Forward speed in RPM (default: 100)")
    parser.add_argument("--turn-speed", type=int, default=100, help="Turning speed in RPM (default: 100)")
    parser.add_argument("--side-length", type=float, default=2.0, help="Stall boundary length in meters (default: 2.0)")
    parser.add_argument("--segments", type=int, default=2, help="Number of intermediate stops per side")
    parser.add_argument("--sec-per-meter", type=float, default=3.5, help="Time in seconds to travel 1 meter")
    parser.add_argument("--sec-per-90-turn", type=float, default=3.8, help="Time in seconds to pivot 90 degrees")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()

    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        logger.info(f"Controller connected: {joystick.get_name()}")
    else:
        logger.warning("No controller found! You must use Ctrl+C to stop.")

    try:
        motors = BaseMotors(args.left_port, args.right_port)
        motors.start()
        patrol_state.start()
        patrol_state.set_active(True)
        patrol_state.set_paused(False)
    except Exception as e:
        logger.error(f"Failed to open ports: {e}")
        pygame.quit()
        sys.exit(1)

    stop_event = threading.Event()
    paused_event = threading.Event()

    patrol_thread = threading.Thread(
        target=patrol_loop, args=(motors, args, paused_event, stop_event), daemon=True
    )
    patrol_thread.start()

    try:
        while patrol_thread.is_alive() and not stop_event.is_set():
            for event in pygame.event.get():
                if event.type != pygame.JOYBUTTONDOWN:
                    continue
                if event.button == _TRIANGLE:
                    if paused_event.is_set():
                        paused_event.clear()  # patrol_loop's _wait_while_paused notices
                    # Already running: Triangle while active does nothing extra.
                else:
                    if not paused_event.is_set():
                        logger.warning(
                            f"!!! PAUSE TRIGGERED VIA CONTROLLER (Button {event.button}) !!!"
                        )
                        paused_event.set()
                        motors.command(0, 0)
            pygame.time.wait(20)  # 50Hz polling

    except KeyboardInterrupt:
        logger.warning("Stopping — Ctrl+C.")
        stop_event.set()
    finally:
        stop_event.set()
        paused_event.clear()  # release _wait_while_paused if it was blocking
        patrol_thread.join(timeout=3.0)
        motors.stop()
        patrol_state.set_active(False)
        patrol_state.set_paused(False)
        pygame.quit()
        logger.info("Robot successfully stopped and powered down.")


if __name__ == "__main__":
    main()
