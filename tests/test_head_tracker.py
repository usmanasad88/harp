"""HeadTracker: the seams that matter for the servo head reading the SHARED
camera. What's under test: it steers at the LARGEST detected face (the
color-only pick, since the shared camera has no depth); it keeps the gimbal's
idle logic ticking when no frame is available yet; a bad detection pass never
kills tracking; and run() shuts down cleanly — the thread stops and the serial
port closes. Fake gimbal + detector keep it hardware-free (no ESP32, no ONNX)."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from harp.motion import head_tracker as head_tracker_mod
from harp.motion.head_tracker import HeadTracker, _largest_face_center, open_face_kiosk

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

# Two faces in the 640x480 frame; the bigger box is the one the head must pick.
SMALL_LEFT = [10, 10, 30, 30]        # area 400
BIG_CENTER = [300, 200, 420, 360]    # area 19200, center (360, 280)


class FakeGimbal:
    def __init__(self) -> None:
        self.tracks: list[tuple[int, int, int, int]] = []
        self.ticks = 0
        self.closed = False
        self._lock = threading.Lock()

    def track(self, cx, cy, frame_w, frame_h) -> None:
        with self._lock:
            self.tracks.append((cx, cy, frame_w, frame_h))

    def tick(self) -> None:
        with self._lock:
            self.ticks += 1

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    """Returns fixed boxes, or raises to exercise the resilience path."""

    def __init__(self, boxes, raises: bool = False) -> None:
        self.boxes = boxes
        self.raises = raises

    def detect(self, frame):
        if self.raises:
            raise RuntimeError("detector blew up")
        return list(self.boxes)


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_largest_face_center_picks_the_biggest_box():
    assert _largest_face_center([]) is None
    assert _largest_face_center([SMALL_LEFT, BIG_CENTER]) == (360, 280)


async def test_tracks_largest_face_then_shuts_down_cleanly():
    gimbal = FakeGimbal()
    detector = FakeDetector([SMALL_LEFT, BIG_CENTER])
    tracker = HeadTracker(
        gimbal, detector, latest_frame=lambda: FRAME, loop_interval=0.005
    )
    task = asyncio.create_task(tracker.run())

    await _wait_for(lambda: gimbal.tracks)
    cx, cy, w, h = gimbal.tracks[0]
    # The bigger face's center, with the real frame dims handed to the gimbal.
    assert (cx, cy) == (360, 280)
    assert (w, h) == (640, 480)

    await _cancel(task)
    # Shutdown released the serial port.
    assert gimbal.closed is True


async def test_no_frame_keeps_ticking_but_never_tracks():
    gimbal = FakeGimbal()
    detector = FakeDetector([BIG_CENTER])  # would track — but no frame arrives
    tracker = HeadTracker(
        gimbal, detector, latest_frame=lambda: None, loop_interval=0.005
    )
    task = asyncio.create_task(tracker.run())

    # The idle logic (rest / look-around) must keep running with no frame.
    await _wait_for(lambda: gimbal.ticks > 3)
    assert gimbal.tracks == []

    await _cancel(task)
    assert gimbal.closed is True


def test_face_kiosk_launches_edge_fullscreen(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(head_tracker_mod, "_find_msedge", lambda: r"C:\edge\msedge.exe")
    monkeypatch.setattr(head_tracker_mod.subprocess, "Popen", lambda args, **kw: calls.append(args))
    # Edge is available → we launch it with the kiosk flags, no browser fallback.
    monkeypatch.setattr(
        head_tracker_mod.webbrowser, "open",
        lambda url: pytest.fail("should not fall back to webbrowser when Edge is found"),
    )
    open_face_kiosk(8788)
    assert calls == [
        [r"C:\edge\msedge.exe", "--kiosk",
         "http://127.0.0.1:8788/face.html", "--edge-kiosk-type=fullscreen"]
    ]


def test_face_kiosk_opens_on_second_monitor(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(head_tracker_mod, "_find_msedge", lambda: r"C:\edge\msedge.exe")
    monkeypatch.setattr(head_tracker_mod.subprocess, "Popen", lambda args, **kw: calls.append(args))
    # Primary at (0, 0), the extended screen to its right at (1920, 0).
    monkeypatch.setattr(head_tracker_mod, "_monitor_origins", lambda: [(0, 0), (1920, 0)])
    open_face_kiosk(8788, monitor=2)
    # The kiosk is seeded on the second display so Edge fullscreens it there.
    assert calls == [
        [r"C:\edge\msedge.exe", "--window-position=1920,0", "--kiosk",
         "http://127.0.0.1:8788/face.html", "--edge-kiosk-type=fullscreen"]
    ]


def test_face_kiosk_missing_monitor_falls_back_to_primary(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(head_tracker_mod, "_find_msedge", lambda: r"C:\edge\msedge.exe")
    monkeypatch.setattr(head_tracker_mod.subprocess, "Popen", lambda args, **kw: calls.append(args))
    # Only one display present, but monitor 2 was asked for → no positioning.
    monkeypatch.setattr(head_tracker_mod, "_monitor_origins", lambda: [(0, 0)])
    open_face_kiosk(8788, monitor=2)
    assert calls == [
        [r"C:\edge\msedge.exe", "--kiosk",
         "http://127.0.0.1:8788/face.html", "--edge-kiosk-type=fullscreen"]
    ]


def test_face_kiosk_falls_back_to_default_browser_without_edge(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(head_tracker_mod, "_find_msedge", lambda: None)
    monkeypatch.setattr(head_tracker_mod.webbrowser, "open", opened.append)
    open_face_kiosk(8788)
    # No Edge → the page still opens somewhere, at the same URL.
    assert opened == ["http://127.0.0.1:8788/face.html"]


async def test_a_bad_detection_pass_does_not_kill_tracking():
    gimbal = FakeGimbal()
    detector = FakeDetector([BIG_CENTER], raises=True)
    tracker = HeadTracker(
        gimbal, detector, latest_frame=lambda: FRAME, loop_interval=0.005
    )
    task = asyncio.create_task(tracker.run())

    # Several passes have raised inside the loop; the runner is still alive.
    await asyncio.sleep(0.1)
    assert task.done() is False
    assert gimbal.tracks == []

    await _cancel(task)
    assert gimbal.closed is True
