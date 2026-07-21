"""Head gimbal face tracking, wired onto the SHARED camera.

The same camera→detect→gimbal loop as the standalone `python -m harp.motion`
runner (see __main__.py's run_tracker), but sourcing frames from the one shared
camera (harp/vision/camera) the whole app already reads, instead of opening a
second RealSense of its own. That second open was the whole problem: a
RealSense can only be owned by ONE process at a time, so running the standalone
gimbal alongside `python -m harp` starved the app's camera. Here the head lives
INSIDE the app and shares its frames — one device, many consumers.

Like every other shared-camera consumer (gestures, face-ID, follow), this only
sees the color stream — there is no depth here — so face selection falls back
to the LARGEST face (_largest_face_center below), the same tradeoff
FollowController makes. Depth-based nearest-face selection (face_tracker's
pick_face + RealSense depth) still lives in the standalone runner for when you
want it.

The loop runs on a worker thread (detection blocks and would stall the event
loop); `run()` is the app-facing runner that starts it and, on shutdown, stops
the thread and closes the serial port. A gimbal that can't be opened (no ESP32
on the port) disables head tracking for the run with a warning — it never
takes the voice side down.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from typing import Any, Callable

import numpy as np

from ..config import MotionSettings
from . import face_server
from .gimbal import Gimbal

logger = logging.getLogger(__name__)

# Loop cadence cap. Detection (~50-100 ms on CPU) self-paces the loop well
# under the camera's ~30 fps; this small wait just keeps a fast/empty pass from
# busy-spinning. The gimbal rate-limits its own serial writes (SEND_INTERVAL).
_LOOP_INTERVAL = 0.03


def _find_msedge() -> str | None:
    """Locate msedge.exe. `start msedge` works in a .bat because cmd resolves it
    via the App Paths registry, which isn't on PATH — so shutil.which usually
    misses it and we fall back to the standard install locations."""
    found = shutil.which("msedge")
    if found:
        return found
    for var in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if not base:
            continue
        path = os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")
        if os.path.isfile(path):
            return path
    return None


# How far inside a monitor's top-left we aim the kiosk window. Not zero: right
# on the shared edge, DPI rounding can bleep the window onto the neighbor, which
# is exactly the "opened on the laptop, not the HDMI" failure.
_KIOSK_INSET = 100


def _monitor_rects() -> list[tuple[tuple[int, int, int, int], bool]]:
    """Each display's ((left, top, right, bottom), is_primary), primary first
    then the rest ordered left-to-right / top-to-bottom. Coordinates are in
    PHYSICAL pixels — we enumerate under a per-monitor-DPI-aware context so they
    match what Edge expects for --window-position (a DPI-unaware read can return
    scaled coords that land the window on the wrong screen). Windows-only;
    returns [] on any non-Windows/API error so callers fall back to the default
    monitor."""
    if not sys.platform.startswith("win"):
        return []
    try:
        import ctypes
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        MONITORINFOF_PRIMARY = 0x1
        user32 = ctypes.windll.user32
        found: list[tuple[tuple[int, int, int, int], bool]] = []

        # Read monitor geometry in physical pixels for this thread only, then
        # restore. DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4. Missing on
        # Windows < 1607, where we just proceed with whatever the default is.
        set_ctx = getattr(user32, "SetThreadDpiAwarenessContext", None)
        prev_ctx = None
        if set_ctx is not None:
            set_ctx.restype = ctypes.c_void_p
            set_ctx.argtypes = [ctypes.c_void_p]
            prev_ctx = set_ctx(ctypes.c_void_p(-4))

        # BOOL cb(HMONITOR, HDC, LPRECT, LPARAM) — pointer-sized args as c_void_p.
        proc_type = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(RECT),
            ctypes.c_void_p,
        )

        def _collect(hmon, _hdc, _rect, _data):
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(MONITORINFO)
            if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                r = info.rcMonitor
                found.append(
                    (
                        (int(r.left), int(r.top), int(r.right), int(r.bottom)),
                        bool(info.dwFlags & MONITORINFOF_PRIMARY),
                    )
                )
            return 1

        try:
            if not user32.EnumDisplayMonitors(0, 0, proc_type(_collect), 0):
                return []
        finally:
            if set_ctx is not None and prev_ctx:
                set_ctx(prev_ctx)  # restore this thread's DPI context
    except Exception as exc:  # any ctypes/API surprise → let the caller default
        logger.debug("face kiosk: monitor enumeration failed (%s)", exc)
        return []
    # Primary first, then the remaining displays left-to-right, top-to-bottom.
    found.sort(key=lambda m: (not m[1], m[0][0], m[0][1]))
    return found


def _kiosk_position(monitor: int) -> tuple[int, int] | None:
    """Top-left (x, y) to seed the kiosk window on the requested 1-based display
    (1 = primary), or None to leave Edge on its default monitor. monitor <= 1
    keeps the historical no-positioning behavior; a monitor that isn't there
    warns and falls back to the primary. The point sits a little INSIDE the
    target display (see _KIOSK_INSET) so edge-of-screen rounding can't push the
    fullscreen window onto the neighboring monitor."""
    if monitor <= 1:
        return None
    rects = _monitor_rects()
    if monitor <= len(rects):
        (left, top, right, bottom), _ = rects[monitor - 1]
        x = left + min(_KIOSK_INSET, max(0, (right - left) // 2))
        y = top + min(_KIOSK_INSET, max(0, (bottom - top) // 2))
        return (x, y)
    logger.warning(
        "face kiosk: monitor %d requested but only %d display(s) found — "
        "opening on the primary instead",
        monitor,
        len(rects),
    )
    return None


def open_face_kiosk(port: int, monitor: int = 1) -> None:
    """Open the animated face page fullscreen in Edge kiosk mode — what
    start_harp.bat's `start msedge --kiosk ... --edge-kiosk-type=fullscreen`
    line did, now done by the app itself. `monitor` (1-based, 1 = primary) picks
    which display it fullscreens on: Edge goes fullscreen on the monitor holding
    the window's top-left, so we seed --window-position on the target display.

    Targeting a non-primary monitor also forces a dedicated --user-data-dir: if
    Edge is ALREADY running, a plain `msedge ...` just hands the URL to the
    existing process and silently drops the window flags (that's why the kiosk
    kept landing on the laptop) — a separate profile makes this a fresh process
    that honors them.

    Best-effort: if Edge can't be found or launched, fall back to the default
    browser (which can't be steered to a monitor); the page is still reachable
    by URL regardless. Call it only once the face server is listening (it binds
    synchronously in face_server.start, so right after that is safe)."""
    url = f"http://127.0.0.1:{port}/face.html"
    edge = _find_msedge()
    if edge is not None:
        args = [edge, "--kiosk", url, "--edge-kiosk-type=fullscreen"]
        position = _kiosk_position(monitor)
        if position is not None:
            profile = os.path.join(tempfile.gettempdir(), "harp-face-kiosk")
            # Inserted before --kiosk so they apply as the window is created.
            # --no-first-run / --disable-session-crashed-bubble keep the fresh
            # profile from marring the face screen with a first-run or "restore
            # pages" bar.
            args[1:1] = [
                f"--user-data-dir={profile}",
                f"--window-position={position[0]},{position[1]}",
                "--no-first-run",
                "--disable-session-crashed-bubble",
            ]
        try:
            subprocess.Popen(args)
            where = f" on monitor {monitor}" if position is not None else ""
            logger.info("face kiosk: opened %s in Edge fullscreen%s", url, where)
            return
        except OSError as exc:
            logger.warning("face kiosk: could not launch Edge (%s) — using default browser", exc)
    else:
        logger.info("face kiosk: Edge not found — opening %s in the default browser", url)
    try:
        webbrowser.open(url)
    except Exception as exc:  # webbrowser can raise assorted platform errors
        logger.warning("face kiosk: could not open %s (%s)", url, exc)


def _default_detector() -> Any:
    """Lazy so importing this module never needs cv2/onnxruntime — tests inject
    a fake, and a run without head tracking never loads the ONNX stack."""
    from .face_tracker import FaceDetector

    return FaceDetector()


def _largest_face_center(boxes: list[list[int]]) -> tuple[int, int] | None:
    """The color-only face pick: the CENTER of the biggest box, or None. The
    shared camera has no depth, so — like FollowController — we track the
    largest face rather than face_tracker.pick_face's depth-based nearest one
    (kept out of the import path so this module needs no cv2/onnxruntime)."""
    if not boxes:
        return None
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return (x1 + x2) // 2, (y1 + y2) // 2


class HeadTracker:
    """Steer the servo head to keep a face centered, off the shared camera.

    Vision comes in as `latest_frame` (the shared camera's newest frame, same
    callable follow/gestures use) so this stays testable without hardware. The
    gimbal and detector are injected via factories for the same reason.
    """

    def __init__(
        self,
        gimbal: Any,
        detector: Any,
        *,
        latest_frame: Callable[[], np.ndarray | None],
        loop_interval: float = _LOOP_INTERVAL,
    ) -> None:
        self._gimbal = gimbal
        self._detector = detector
        self._latest_frame = latest_frame
        self._loop_interval = loop_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    async def create(
        cls,
        settings: MotionSettings,
        *,
        latest_frame: Callable[[], np.ndarray | None],
        gimbal_factory: Callable[..., Any] = Gimbal,
        detector_factory: Callable[[], Any] = _default_detector,
        face_present: Callable[[bool], None] = face_server.set_face_present,
        start_server: bool = True,
    ) -> "HeadTracker":
        """Open the gimbal + detector off the event loop and (optionally) start
        the animated-face server. Raises if the ESP32 port can't be opened —
        run_app turns that into a "head tracking disabled" warning."""

        def _build() -> tuple[Any, Any]:
            # Serial open + ONNX session load both block; keep them off the loop.
            gimbal = gimbal_factory(settings.gimbal_port, on_face_change=face_present)
            detector = detector_factory()
            return gimbal, detector

        gimbal, detector = await asyncio.to_thread(_build)
        if start_server:
            try:
                face_server.start(port=settings.face_server_port)
            except OSError:
                # Don't leak the serial port if the face page can't bind.
                gimbal.close()
                raise
        return cls(gimbal, detector, latest_frame=latest_frame)

    async def run(self) -> None:
        """App-facing runner: start the tracking thread and stay alive until the
        app cancels this task, then stop the thread and release the port. Never
        returns on its own — a tracking hiccup must not trip the app's
        first-task-done shutdown (the loop below swallows per-frame errors)."""
        self._thread = threading.Thread(
            target=self._loop, name="harp-head-tracker", daemon=True
        )
        self._thread.start()
        try:
            await asyncio.Event().wait()  # runs until cancelled at shutdown
        finally:
            self._stop_event.set()
            await asyncio.to_thread(self._thread.join, 5)
            self._gimbal.close()

    def _loop(self) -> None:
        """Blocking; runs on the worker thread until stop_event. Each pass:
        detect faces in the latest shared frame, steer the head at the chosen
        one, and run the gimbal's idle logic (rest / look-around when no face)."""
        while not self._stop_event.is_set():
            frame = self._latest_frame()
            if frame is None:
                # No frame yet (camera reconnecting, or none produced): keep the
                # idle behavior alive so the head still rests / sweeps.
                self._gimbal.tick()
                self._stop_event.wait(self._loop_interval)
                continue
            try:
                boxes = self._detector.detect(frame)
                center = _largest_face_center(boxes)
                if center is not None:
                    cx, cy = center
                    h, w = frame.shape[:2]
                    self._gimbal.track(cx, cy, w, h)
                self._gimbal.tick()
            except Exception:
                # One bad pass must not kill tracking — log and carry on, same
                # spirit as the shared camera's reconnect loop.
                logger.exception("head tracker: pass failed")
            self._stop_event.wait(self._loop_interval)
