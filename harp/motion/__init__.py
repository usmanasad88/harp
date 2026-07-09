"""Robot body — servo gimbal head-tracking + RMD-X8 base motors (no ROS).

Ported from the harpcontrol repo (Raspberry Pi 5 / ROS 2 Jazzy) as plain
Python, per PLAN.md "Motion / robot body": pyserial talks to the hardware
directly, pygame replaces the Linux-only evdev controller reader, and
in-process wiring replaces the ROS 2 topics.

Phase 1 is a standalone entry point — `uv run python -m harp.motion` — that
runs face tracking (RealSense or webcam) driving the gimbal, and PS5 teleop
driving the base motors, without touching the rest of HARP. Wiring these
subsystems onto the app's event bus (voice teleop tools, harp.yaml config)
is the next phase.

Every piece of hardware is optional: a missing serial port, camera, or
controller disables just that part with a warning, never a crash.
"""
