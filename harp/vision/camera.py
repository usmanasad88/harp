"""Shared camera capture — one open device, many consumers.

Presence, face-ID, gestures, and the voice layer's image Q&A all need frames.
Opening the webcam several times fails on most hardware, so this is the single
owner of the device: it captures frames and lets every subsystem pull the latest
one.

Capture runs on a background thread, not the asyncio loop: `cv2.VideoCapture.read()`
blocks on real hardware, and stalling the event loop would stall every other
subsystem (voice, bus dispatch, ...) sharing it. The thread just keeps the
newest frame around under a lock — nobody needs every frame, only the freshest.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# How long to wait between reopen attempts once the device drops out.
_RECONNECT_DELAY = 1.0


class Camera:
    def __init__(
        self,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    async def start(self) -> None:
        """Open the device and begin capturing frames on a background thread."""
        loop = asyncio.get_running_loop()
        self._cap = await loop.run_in_executor(None, self._open)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="harp-camera", daemon=True
        )
        self._thread.start()

    def latest(self) -> np.ndarray | None:
        """Return the most recent frame (or None if none yet). Shared by all."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    async def stop(self) -> None:
        """Release the device."""
        self._stop_event.set()
        if self._thread is not None:
            await asyncio.get_running_loop().run_in_executor(None, self._thread.join)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera device {self._device!r}")
        return cap

    def _capture_loop(self) -> None:
        """Runs on the background thread until stop(); reopens the device on failure."""
        while not self._stop_event.is_set():
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("camera %r read failed, reconnecting", self._device)
                self._reconnect()
                continue
            with self._lock:
                self._frame = frame

    def _reconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
        while not self._stop_event.is_set():
            try:
                self._cap = self._open()
            except RuntimeError:
                self._stop_event.wait(_RECONNECT_DELAY)
            else:
                logger.info("camera %r reconnected", self._device)
                return
