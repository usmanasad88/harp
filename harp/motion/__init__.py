"""Robot body — servo gimbal head-tracking + RMD-X8 base motors (no ROS).

Ported from the harpcontrol repo (Raspberry Pi 5 / ROS 2 Jazzy) as plain
Python, per PLAN.md "Motion / robot body": pyserial talks to the hardware
directly, pygame replaces the Linux-only evdev controller reader, and
in-process wiring replaces the ROS 2 topics.

Two ways in. The standalone entry point — `uv run python -m harp.motion` —
runs face tracking (RealSense or webcam) driving the gimbal, and PS5 teleop
driving the base motors, without touching the rest of HARP; keep it for depth-
based nearest-face tracking and hands-on teleop. The in-app wiring came after:
the wheeled base's move_around + follow tools (controller.py / follow.py), and
head_tracker.py — the gimbal head tracking reading the app's SHARED camera
(no second RealSense), started by `python -m harp` itself and configured in
harp.yaml (`motion:`).

Every piece of hardware is optional: a missing serial port, camera, or
controller disables just that part with a warning, never a crash.
"""
