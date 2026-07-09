"""Servo gimbal head — PID face tracking over ESP32 serial.

Port of harpcontrol's facetracking.py (HeadControlNode) minus ROS: the same
PID gains, servo limits, rest position, and look-around sweep, driven by
direct method calls instead of a /target_face subscription. The ESP32
expects ASCII ``Y{yaw}P{pitch}\\n`` at 9600 baud and moves the two servos.

`track(cx, cy)` feeds a detected face center (pixel coords); `tick()` must
be called periodically (~10 Hz) to run the idle logic — return to rest after
2 s without a face, start the look-around sweep after 3 s.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import serial

logger = logging.getLogger(__name__)

# Servo limits (degrees) — from the ESP32 sketch's mechanical range.
YAW_MIN = 20
YAW_MAX = 90
PITCH_MIN = 40
PITCH_MAX = 70

REST_YAW = 55
REST_PITCH = 40

FRAME_W = 640
FRAME_H = 480

BAUD = 9600
SEND_INTERVAL = 0.05  # 20 Hz max serial rate

# PID gains — tuned on the real head in harpcontrol; don't retune blind.
KP_YAW = 0.070
KI_YAW = 0.0004
KD_YAW = 0.012
KP_PITCH = 0.070
KI_PITCH = 0.0004
KD_PITCH = 0.011
I_MAX = 200

_FACE_LOST_AFTER = 0.5  # s without a face before "face lost"
_REST_AFTER = 2.0  # s without a face before returning to rest
_LOOKAROUND_AFTER = 3.0  # s without a face before sweeping


def _clamp(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, value)))


class PIDAxis:
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def compute(self, error: float) -> float:
        now = time.time()
        dt = max(now - self.prev_time, 1e-6)
        self.prev_time = now

        self.integral += error * dt
        self.integral = max(-I_MAX, min(I_MAX, self.integral))

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class Gimbal:
    """One serial connection to the ESP32 head; tracking + idle behavior."""

    def __init__(
        self,
        port: str,
        baud: int = BAUD,
        on_face_change: Callable[[bool], None] | None = None,
        serial_factory: Callable[..., serial.Serial] = serial.Serial,
    ) -> None:
        self._ser = serial_factory(port, baud, timeout=1)
        self._on_face_change = on_face_change

        self._pid_yaw = PIDAxis(KP_YAW, KI_YAW, KD_YAW)
        self._pid_pitch = PIDAxis(KP_PITCH, KI_PITCH, KD_PITCH)

        self.yaw = float(REST_YAW)
        self.pitch = float(REST_PITCH)

        self._last_send = 0.0
        self._last_face_time = time.time()
        self._face_seen = False

        # Look-around sweep state (same stage machine as facetracking.py).
        self._lookaround_active = False
        self._lookaround_stage = 0
        self._target_look_yaw = REST_YAW
        self._next_stage_time = time.time()
        self._sweep_speed = 1.0

        logger.info("gimbal ready on %s (rest Y%d P%d)", port, REST_YAW, REST_PITCH)

    def track(
        self, cx: float, cy: float, frame_w: int = FRAME_W, frame_h: int = FRAME_H
    ) -> None:
        """Steer toward a face center given in pixel coordinates."""
        self._lookaround_active = False
        self._last_face_time = time.time()

        if not self._face_seen:
            self._face_seen = True
            logger.info("gimbal: face acquired")
            if self._on_face_change:
                self._on_face_change(True)

        err_x = (cx - frame_w / 2) / (frame_w / 2)
        err_y = (cy - frame_h / 2) / (frame_h / 2)

        self.yaw -= self._pid_yaw.compute(err_x) * 40
        self.pitch += self._pid_pitch.compute(err_y) * 40

        self.yaw = _clamp(self.yaw, YAW_MIN, YAW_MAX)
        self.pitch = _clamp(self.pitch, PITCH_MIN, PITCH_MAX)

        self._send()

    def tick(self) -> None:
        """Idle logic — call ~10 Hz. Rest after 2 s faceless, sweep after 3 s."""
        now = time.time()
        no_face_for = now - self._last_face_time

        if no_face_for > _FACE_LOST_AFTER and self._face_seen:
            self._face_seen = False
            logger.info("gimbal: face lost")
            if self._on_face_change:
                self._on_face_change(False)

        if no_face_for < _FACE_LOST_AFTER:
            self._lookaround_active = False
            return

        if no_face_for > _REST_AFTER and not self._lookaround_active:
            self.yaw = float(REST_YAW)
            self.pitch = float(REST_PITCH)
            self._send()

        if no_face_for > _LOOKAROUND_AFTER and not self._lookaround_active:
            self._lookaround_active = True
            self._lookaround_stage = 0
            self._target_look_yaw = YAW_MAX
            self._next_stage_time = now

        if self._lookaround_active:
            self._lookaround_step(now)

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass

    def _lookaround_step(self, now: float) -> None:
        if abs(self.yaw - self._target_look_yaw) > 1:
            direction = 1 if self._target_look_yaw > self.yaw else -1
            self.yaw = _clamp(self.yaw + direction * self._sweep_speed, YAW_MIN, YAW_MAX)
            self.pitch = float(REST_PITCH)
            self._send()
            return

        if self._lookaround_stage == 0:
            self._target_look_yaw = YAW_MAX
            self._lookaround_stage = 1
        elif self._lookaround_stage == 1:
            self._target_look_yaw = YAW_MIN
            self._lookaround_stage = 2
        elif self._lookaround_stage == 2:
            self._target_look_yaw = REST_YAW
            self._lookaround_stage = 3
        elif self._lookaround_stage == 3:
            self._next_stage_time = now + 5
            self._lookaround_stage = 4
        elif self._lookaround_stage == 4:
            if now >= self._next_stage_time:
                self._target_look_yaw = YAW_MAX
                self._lookaround_stage = 1

    def _send(self) -> None:
        now = time.time()
        if now - self._last_send < SEND_INTERVAL:
            return
        self._last_send = now

        servo_yaw = _clamp(self.yaw, YAW_MIN, YAW_MAX)
        # The +15 pitch bias compensates the head's mounting angle (from
        # facetracking.py's send_servo) — the clamp still bounds the servo.
        servo_pitch = _clamp(self.pitch + 15, PITCH_MIN, PITCH_MAX)

        cmd = f"Y{servo_yaw}P{servo_pitch}\n"
        try:
            self._ser.write(cmd.encode())
        except Exception:
            logger.warning("gimbal: serial write failed", exc_info=True)
        logger.debug("gimbal sent: %s", cmd.strip())
