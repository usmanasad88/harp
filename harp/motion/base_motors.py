"""RMD-X8 base motors — differential drive over two plain serial ports.

Port of harpcontrol's rmd2.py motor path (raw RMD-X8 speed frames at 115200
baud, one port per wheel) plus the safety the original lacked: a **deadman
timeout**. A dedicated writer thread re-sends the current command at 20 Hz;
if nobody has refreshed the command within `deadman_seconds` (teleop thread
died, script hung), or the subsystem is shutting down, it writes zero-speed
frames to both motors. Teleop must therefore call `command()` continuously
(hold-to-drive), not latch a speed and walk away.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import serial

logger = logging.getLogger(__name__)

BAUD = 115200
WRITE_INTERVAL = 0.05  # 20 Hz refresh
DEADMAN_SECONDS = 0.25


def make_rmdx8_speed_command(rpm: int) -> bytes:
    """Build one RMD-X8 speed-control frame (protocol proven in harpcontrol)."""
    header = bytes.fromhex("3E A2 01 04 E5")
    speed_val = int(rpm * 100)
    speed_bytes = speed_val.to_bytes(4, byteorder="little", signed=True)
    checksum = (sum(header + speed_bytes) + 0x36) & 0xFF
    return header + speed_bytes + bytes([checksum])


class BaseMotors:
    """Owns both wheel ports; refreshes the last command, zeroes when stale."""

    def __init__(
        self,
        left_port: str,
        right_port: str,
        baud: int = BAUD,
        deadman_seconds: float = DEADMAN_SECONDS,
        serial_factory: Callable[..., serial.Serial] = serial.Serial,
    ) -> None:
        self._left = serial_factory(left_port, baud, timeout=0.05)
        self._right = serial_factory(right_port, baud, timeout=0.05)
        self._deadman = deadman_seconds

        self._lock = threading.Lock()
        self._left_rpm = 0
        self._right_rpm = 0
        self._refreshed_at = 0.0  # epoch of the last command(); 0 = never
        self._was_stale = True

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        logger.info("base motors ready (left=%s right=%s)", left_port, right_port)

    def start(self) -> None:
        self._write_both(0, 0)
        self._thread = threading.Thread(
            target=self._writer_loop, name="harp-base-motors", daemon=True
        )
        self._thread.start()

    def command(self, left_rpm: int, right_rpm: int) -> None:
        """Set wheel speeds; must be called again within the deadman window."""
        with self._lock:
            self._left_rpm = left_rpm
            self._right_rpm = right_rpm
            self._refreshed_at = time.time()

    def stop(self) -> None:
        """Halt the writer, zero both motors, close the ports."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        try:
            self._write_both(0, 0)
        finally:
            for port in (self._left, self._right):
                try:
                    port.close()
                except Exception:
                    pass
        logger.info("base motors stopped, ports closed")

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick(time.time())
            self._stop_event.wait(WRITE_INTERVAL)
        # Belt and braces: the writer's last act is a zero frame even if
        # stop() never runs (e.g. the main thread died).
        try:
            self._write_both(0, 0)
        except Exception:
            pass

    def _tick(self, now: float) -> None:
        """One writer pass: re-send the current command, or zeros if stale."""
        with self._lock:
            stale = now - self._refreshed_at > self._deadman
            left, right = (0, 0) if stale else (self._left_rpm, self._right_rpm)
        if stale and not self._was_stale:
            logger.warning("base motors: deadman timeout — zeroing")
        self._was_stale = stale
        self._write_both(left, right)

    def _write_both(self, left_rpm: int, right_rpm: int) -> None:
        try:
            self._left.write(make_rmdx8_speed_command(left_rpm))
            self._right.write(make_rmdx8_speed_command(right_rpm))
        except Exception:
            logger.warning("base motors: serial write failed", exc_info=True)
