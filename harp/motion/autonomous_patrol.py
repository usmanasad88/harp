#!/usr/bin/env python3
"""Standalone perimeter-patrol runner with a PS4-controller E-STOP.

    uv run python -m harp.motion.autonomous_patrol --left-port COM4 --right-port COM5 \
        --side-length 3.0 --segments 2 --base-speed 350 --turn-speed 300

The motion itself (drive → look-around → 90° corner laps) lives in patrol.py,
shared with the full agent's move_around tool; this wrapper adds the CLI, the
pygame controller E-STOP (any button press), and Ctrl+C handling, and laps
forever until stopped. Don't run it while the full agent's motion subsystem is
enabled — both want the same two serial ports.
"""

import argparse
import logging
import os
import sys
import threading

# Hide pygame support prompt
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from .base_motors import BaseMotors
from .patrol import PatrolParams, run_patrol

logger = logging.getLogger("autonomous_patrol")


def main():
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    try:
        params = PatrolParams(
            side_length=args.side_length,
            segments=args.segments,
            base_speed=args.base_speed,
            turn_speed=args.turn_speed,
            sec_per_meter=args.sec_per_meter,
            sec_per_90_turn=args.sec_per_90_turn,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Initialize Pygame & Controller Subsystem
    pygame.init()
    pygame.joystick.init()

    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        logger.info(f"E-STOP controller connected: {joystick.get_name()}")
    else:
        logger.warning("No controller found! You must use Ctrl+C to stop.")

    # Initialize serial motors
    try:
        motors = BaseMotors(args.left_port, args.right_port)
        motors.start()
    except Exception as e:
        logger.error(f"Failed to open ports: {e}")
        pygame.quit()
        sys.exit(1)

    stop_event = threading.Event()

    logger.info("--- AUTONOMOUS PATROL STARTED ---")
    logger.info(f"Target Boundary: {params.side_length}m x {params.side_length}m")
    logger.info("READY! Press ANY button on your PS4 controller to E-STOP.")

    # Start the patrol (laps=0: forever) in a background thread
    patrol_thread = threading.Thread(
        target=run_patrol,
        args=(motors, params, stop_event),
        kwargs={"laps": 0},
        daemon=True,
    )
    patrol_thread.start()

    # Main thread polls for Pygame controller events
    try:
        while patrol_thread.is_alive() and not stop_event.is_set():
            for event in pygame.event.get():
                # Trigger E-STOP on any button press
                if event.type == pygame.JOYBUTTONDOWN:
                    logger.warning(f"!!! EMERGENCY STOP TRIGGERED VIA CONTROLLER (Button {event.button}) !!!")
                    stop_event.set()
                    break
            pygame.time.wait(20) # 50Hz polling

    except KeyboardInterrupt:
        logger.warning("Emergency Stop triggered via Terminal (Ctrl+C).")
        stop_event.set()
    finally:
        stop_event.set()
        patrol_thread.join(timeout=3.0)
        motors.stop()
        pygame.quit()
        logger.info("Robot successfully stopped and powered down.")

if __name__ == "__main__":
    main()
