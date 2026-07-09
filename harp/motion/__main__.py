"""Standalone robot-body runner — face tracking + PS5 teleop, no ROS.

    uv run python -m harp.motion --list-ports
    uv run python -m harp.motion --gimbal-port COM5 --left-port COM6 --right-port COM7
    uv run python -m harp.motion --gimbal-port COM5 --preview      # head only
    uv run python -m harp.motion --left-port COM6 --right-port COM7  # drive only
    uv run python -m harp.motion --test-controller                 # verify buttons

This is phase 1 of PLAN.md's "Motion / robot body": one process, no event
bus — the camera→detector→gimbal loop runs on a worker thread, the PS5
teleop loop runs on the main thread feeding the base motors' deadman-guarded
writer. Every piece of hardware is optional; whatever isn't found is skipped
with a warning and the rest runs.

Serial ports are given explicitly (Windows COM letters aren't stable across
replug — check with --list-ports, which also shows VID:PID and serial number
for telling the ESP32 apart from the two motor adapters).
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import cv2
from serial.tools import list_ports

from .base_motors import BaseMotors
from .face_tracker import FaceDetector, open_camera, pick_face
from .gimbal import Gimbal
from .teleop_ps5 import PS5Teleop, run_test

logger = logging.getLogger("harp.motion")

_TICK_INTERVAL = 0.1  # gimbal idle logic cadence when no frame arrives


def print_ports() -> None:
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    if not ports:
        print("No serial ports found.")
        return
    for p in ports:
        vid_pid = (
            f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None and p.pid is not None else "-"
        )
        print(f"{p.device:8} {vid_pid:10} sn={p.serial_number or '-':16} {p.description}")


def run_tracker(camera, detector, gimbal, preview: bool, stop_event: threading.Event) -> None:
    """Camera → detect → gimbal loop (worker thread). Owns the preview window."""
    try:
        while not stop_event.is_set():
            frame, depth = camera.read()
            if frame is None:
                if gimbal:
                    gimbal.tick()
                stop_event.wait(_TICK_INTERVAL)
                continue

            boxes = detector.detect(frame)
            target = pick_face(boxes, depth)
            if target and gimbal:
                cx, cy, _ = target
                gimbal.track(cx, cy, frame.shape[1], frame.shape[0])
            if gimbal:
                gimbal.tick()

            if preview:
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if target:
                    cx, cy, dist = target
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    if dist is not None:
                        cv2.putText(
                            frame, f"{dist:.2f} m", (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                        )
                cv2.imshow("harp.motion face tracking", frame)
                cv2.waitKey(1)
    except Exception:
        logger.exception("face tracker crashed")
    finally:
        camera.close()
        if preview:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m harp.motion",
        description="Standalone gimbal face tracking + PS5 base-motor teleop (no ROS).",
    )
    parser.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    parser.add_argument("--test-controller", action="store_true",
                        help="print controller button/hat events and exit")
    parser.add_argument("--gimbal-port", help="ESP32 head serial port (e.g. COM5)")
    parser.add_argument("--left-port", help="left RMD-X8 motor serial port")
    parser.add_argument("--right-port", help="right RMD-X8 motor serial port")
    parser.add_argument("--camera", choices=["auto", "realsense", "webcam", "none"],
                        default="auto", help="camera backend (default: auto)")
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--preview", action="store_true", help="show the detection window")
    parser.add_argument("--base-speed", type=int, default=500, help="drive rpm (default 500)")
    parser.add_argument("--turn-speed", type=int, default=500, help="turn rpm (default 500)")
    parser.add_argument("--deadman", type=float, default=0.25,
                        help="seconds without a teleop refresh before motors zero (default 0.25)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_ports:
        print_ports()
        return
    if args.test_controller:
        run_test()
        return

    # --- Hardware, each piece optional -----------------------------------
    gimbal = None
    if args.gimbal_port:
        try:
            gimbal = Gimbal(args.gimbal_port)
        except Exception as exc:
            logger.warning("gimbal disabled: %s", exc)

    motors = None
    if args.left_port and args.right_port:
        try:
            motors = BaseMotors(
                args.left_port, args.right_port, deadman_seconds=args.deadman
            )
        except Exception as exc:
            logger.warning("base motors disabled: %s", exc)
    elif args.left_port or args.right_port:
        logger.warning("base motors need BOTH --left-port and --right-port — disabled")

    camera = None
    detector = None
    if args.camera != "none" and (gimbal or args.preview):
        camera = open_camera(args.camera, args.webcam_index)
        if camera is not None:
            try:
                detector = FaceDetector()
            except Exception as exc:
                logger.warning("face tracking disabled: %s", exc)
                camera.close()
                camera = None
    elif gimbal is None and args.camera != "none":
        logger.info("no gimbal and no --preview — camera not opened")

    if gimbal and camera is None:
        logger.warning("gimbal has no camera — head will hold still")

    if not (gimbal or motors or (camera and args.preview)):
        parser.error(
            "nothing to run — give --gimbal-port / --left-port + --right-port "
            "(see --list-ports), or --preview for detection only"
        )

    # --- Run --------------------------------------------------------------
    stop_event = threading.Event()
    tracker_thread = None
    if camera is not None and detector is not None:
        tracker_thread = threading.Thread(
            target=run_tracker,
            args=(camera, detector, gimbal, args.preview, stop_event),
            name="harp-face-tracker",
            daemon=True,
        )
        tracker_thread.start()

    if motors:
        motors.start()

    logger.info("running — Ctrl+C to stop")
    try:
        if motors:
            # Teleop on the main thread (pygame prefers it); feeds the deadman.
            PS5Teleop(motors.command, args.base_speed, args.turn_speed).run(stop_event)
        else:
            while not stop_event.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if tracker_thread is not None:
            tracker_thread.join(timeout=5)
        if motors:
            motors.stop()
        if gimbal:
            gimbal.close()
        logger.info("stopped")


if __name__ == "__main__":
    main()
