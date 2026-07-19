"""Shared camera capture — one open device, many consumers.

Presence, face-ID, gestures, and the voice layer's image Q&A all need frames.
Opening the webcam several times fails on most hardware, so this is the single
owner of the device: it captures frames and lets every subsystem pull the latest
one.

Two backends, chosen at open (harp.yaml `camera.backend`): `auto` (the default)
uses the Intel RealSense's color stream when one is plugged in, else the plain
cv2 webcam. Only the color stream is enabled — depth belongs to the standalone
motion process (harp/motion), and a RealSense can only be owned by ONE process
at a time: if you run `python -m harp.motion` alongside the full agent, pin
this camera to `backend: webcam` so motion keeps the RealSense.

pyrealsense2 is imported lazily so webcam mode works even where the RealSense
SDK is missing or broken.

Capture runs on a background thread, not the asyncio loop: reading a frame
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

BACKEND_CHOICES = ("auto", "realsense", "webcam")


class _RealSenseBackend:
    """RealSense color stream via pyrealsense2. Construction raises RuntimeError
    when no device is connected (that IS the detection `auto` relies on) or no
    stream profile resolves; `read()` returns None once frames stop arriving."""

    name = "realsense"

    def __init__(self, width: int, height: int, fps: int) -> None:
        import pyrealsense2 as rs  # lazy: webcam mode must not need the SDK

        if len(rs.context().query_devices()) == 0:
            raise RuntimeError("no RealSense device connected")
        self._pipeline = rs.pipeline()
        last_error: Exception | None = None
        # Full rate first; the half-rate profile keeps frames coming on a USB 2
        # link (hub/cable), same trade-off as harp/motion's face tracker.
        for profile_fps in (fps, 15):
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, profile_fps)
            try:
                self._pipeline.start(cfg)
            except RuntimeError as exc:
                last_error = exc
                continue
            if profile_fps != fps:
                logger.warning(
                    "camera: RealSense color only resolved at %d fps — likely a "
                    "USB 2 link; plug it into a USB 3 port for full rate",
                    profile_fps,
                )
            return
        raise RuntimeError(f"no RealSense color profile resolved: {last_error}")

    def read(self) -> np.ndarray | None:
        try:
            frames = self._pipeline.wait_for_frames()
        except RuntimeError:  # timed out / device dropped — reconnect path
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        # Copy out of librealsense's frame pool: the buffer behind get_data()
        # is recycled on a later wait_for_frames(), but we keep this frame
        # around as `latest()` indefinitely.
        return np.asanyarray(color.get_data()).copy()

    def release(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


class _WebcamBackend:
    """Plain cv2.VideoCapture webcam."""

    name = "webcam"

    def __init__(self, device: int | str, width: int, height: int, fps: int) -> None:
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera device {device!r}")
        self._cap = cap

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


class Camera:
    def __init__(
        self,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: str = "auto",
    ) -> None:
        if backend not in BACKEND_CHOICES:
            logger.warning(
                "camera: backend %r not one of %s; using 'auto'",
                backend, BACKEND_CHOICES,
            )
            backend = "auto"
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._backend_choice = backend
        self._backend: _RealSenseBackend | _WebcamBackend | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # A source switch requested from another thread (the dashboard's
        # camera dropdown) — picked up at the top of the next capture
        # iteration so the device is only ever opened/closed from the thread
        # that owns it, never touched concurrently from two threads.
        self._pending_switch: tuple[str, int | str | None] | None = None
        self._switch_lock = threading.Lock()

    async def start(self) -> None:
        """Open the device and begin capturing frames on a background thread."""
        loop = asyncio.get_running_loop()
        self._backend = await loop.run_in_executor(None, self._open)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="harp-camera", daemon=True
        )
        self._thread.start()

    def latest(self) -> np.ndarray | None:
        """Return the most recent frame (or None if none yet). Shared by all."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def active_backend(self) -> str | None:
        """Name of whichever backend is actually driving frames right now
        ('realsense' | 'webcam'), or None if not opened yet. Informational
        (e.g. for the dashboard to show what `auto` actually resolved to);
        safe to read from any thread since it's a single reference read."""
        backend = self._backend
        return backend.name if backend is not None else None

    def request_switch(self, backend: str, device: int | str | None = None) -> None:
        """Ask the capture thread to reopen with a new source — the dashboard
        camera dropdown, or any other caller. Applied at the top of the next
        capture iteration, not here, so the device is only ever opened/closed
        from the thread that already owns it. `device` is ignored by the
        realsense backend; pass None to keep the current webcam device."""
        if backend not in BACKEND_CHOICES:
            logger.warning(
                "camera: switch requested with backend %r not one of %s; ignoring",
                backend, BACKEND_CHOICES,
            )
            return
        with self._switch_lock:
            self._pending_switch = (backend, device)

    async def stop(self) -> None:
        """Release the device."""
        self._stop_event.set()
        if self._thread is not None:
            await asyncio.get_running_loop().run_in_executor(None, self._thread.join)
            self._thread = None
        if self._backend is not None:
            self._backend.release()
            self._backend = None

    def _open(self) -> _RealSenseBackend | _WebcamBackend:
        """Open the configured backend; `auto` prefers RealSense, then webcam.
        Reconnection re-runs this, so a RealSense unplugged mid-run hands `auto`
        over to the webcam (and back once it reappears)."""
        if self._backend_choice in ("auto", "realsense"):
            try:
                backend = _RealSenseBackend(self._width, self._height, self._fps)
            except Exception as exc:
                if self._backend_choice == "realsense":
                    raise RuntimeError(f"could not open RealSense camera: {exc}") from exc
                logger.info("camera: RealSense not detected (%s) — trying webcam", exc)
            else:
                logger.info("camera: RealSense color stream")
                return backend
        backend = _WebcamBackend(self._device, self._width, self._height, self._fps)
        logger.info("camera: webcam %r", self._device)
        return backend

    def _capture_loop(self) -> None:
        """Runs on the background thread until stop(); reopens the device on
        failure, or on a source switch requested from another thread."""
        while not self._stop_event.is_set():
            if self._pending_switch is not None:
                self._reconnect()
                continue
            assert self._backend is not None
            frame = self._backend.read()
            if frame is None:
                logger.warning("camera %s read failed, reconnecting", self._backend.name)
                self._reconnect()
                continue
            with self._lock:
                self._frame = frame

    def _take_pending_switch(self) -> tuple[str, int | str | None] | None:
        with self._switch_lock:
            pending, self._pending_switch = self._pending_switch, None
        return pending

    def _reconnect(self) -> None:
        """Reopen the device — after a read failure, or a requested source
        switch (self._pending_switch), consumed here so it's only ever
        applied on the thread that already owns the device. Checked on every
        retry, not just the first, so a switch requested while a previous one
        is still failing to open takes over immediately instead of waiting
        out the failed one's retry delay."""
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        while not self._stop_event.is_set():
            pending = self._take_pending_switch()
            if pending is not None:
                backend, device = pending
                logger.info(
                    "camera: switching source to %s%s",
                    backend, f" (device {device!r})" if device is not None else "",
                )
                self._backend_choice = backend
                if device is not None:
                    self._device = device
                with self._lock:
                    self._frame = None
            try:
                self._backend = self._open()
            except RuntimeError:
                self._stop_event.wait(_RECONNECT_DELAY)
            else:
                logger.info("camera reconnected (%s)", self._backend.name)
                return
