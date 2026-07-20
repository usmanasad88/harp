#!/usr/bin/env python3
import argparse
import logging
import sys
import threading
import time
import os

# Hide pygame support prompt
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from .base_motors import BaseMotors
from . import patrol_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("autonomous_patrol")

# Control loop frequency
_POLL_INTERVAL = 0.05  # 20 Hz (well inside the motors' 0.25s deadman window)

def execute_command(motors: BaseMotors, left_rpm: int, right_rpm: int, duration: float, stop_event: threading.Event):
    """Feeds commands to the motor at 20 Hz for a specified duration."""
    start_time = time.time()
    while time.time() - start_time < duration:
        if stop_event.is_set():
            break
        motors.command(left_rpm, right_rpm)
        time.sleep(_POLL_INTERVAL)

def humanoid_scan(motors: BaseMotors, turn_speed: int, sec_per_15_deg: float, stop_event: threading.Event):
    """Executes a lifelike left-to-right scanning swivel."""
    logger.info("[Scan] Swiveling left to scan...")
    execute_command(motors, -turn_speed, -turn_speed, sec_per_15_deg, stop_event)
    execute_command(motors, 0, 0, 1.5, stop_event)  # Pause and look
    
    logger.info("[Scan] Swiveling right to scan...")
    execute_command(motors, turn_speed, turn_speed, sec_per_15_deg * 2, stop_event)
    execute_command(motors, 0, 0, 1.5, stop_event)  # Pause and look
    
    logger.info("[Scan] Recentering base...")
    execute_command(motors, -turn_speed, -turn_speed, sec_per_15_deg, stop_event)
    execute_command(motors, 0, 0, 1.0, stop_event)  # Final pause

def patrol_stall(motors: BaseMotors, args: argparse.Namespace, stop_event: threading.Event):
    """Executes the perimeter patrol loop in a separate thread."""
    segment_distance = args.side_length / args.segments
    segment_duration = segment_distance * args.sec_per_meter
    sec_per_15_deg = args.sec_per_90_turn / 6.0

    logger.info("--- AUTONOMOUS PATROL STARTED ---")
    logger.info(f"Target Boundary: {args.side_length}m x {args.side_length}m")
    logger.info("READY! Press ANY button on your PS4 controller to E-STOP.")

    try:
        while not stop_event.is_set():
            for side in range(1, 5):
                logger.info(f"--- Patrolling Boundary Wall {side}/4 ---")
                
                for segment in range(args.segments):
                    if stop_event.is_set():
                        return
                    
                    # 1. Drive one step forward
                    logger.info(f"Driving forward: Step {segment + 1}/{args.segments}")
                    execute_command(motors, args.base_speed, -args.base_speed, segment_duration, stop_event)
                    
                    # 2. Settle and scan
                    execute_command(motors, 0, 0, 0.5, stop_event)
                    humanoid_scan(motors, args.turn_speed, sec_per_15_deg, stop_event)

                # 3. Corner reached -> Turn 90 degrees
                if stop_event.is_set():
                    return
                logger.info("Corner reached. Turning 90 degrees...")
                execute_command(motors, args.turn_speed, args.turn_speed, args.sec_per_90_turn, stop_event)
                execute_command(motors, 0, 0, 1.0, stop_event)

    except Exception as e:
        logger.exception(f"Error during patrol: {e}")

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
        patrol_state.start()
        patrol_state.set_active(True)
    except Exception as e:
        logger.error(f"Failed to open ports: {e}")
        pygame.quit()
        sys.exit(1)

    stop_event = threading.Event()

    # Start the patrol script in a background thread
    patrol_thread = threading.Thread(
        target=patrol_stall, 
        args=(motors, args, stop_event), 
        daemon=True
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
        patrol_state.set_active(False)
        pygame.quit()
        logger.info("Robot successfully stopped and powered down.")

if __name__ == "__main__":
    main()