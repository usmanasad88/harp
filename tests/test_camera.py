"""Camera owns the one video device: test it against a fake cv2.VideoCapture
and a fake RealSense backend so the suite doesn't need real hardware."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from harp.vision.camera import Camera


class _FakeCapture:
    """Stands in for cv2.VideoCapture. `reads` is a queue of (ok, frame) pairs
    consumed in order; once exhausted, keeps returning the last entry."""

    instances: list["_FakeCapture"] = []

    def __init__(self, device) -> None:
        self.device = device
        self.reads: list[tuple[bool, np.ndarray | None]] = [
            (True, np.full((2, 2, 3), 1, dtype=np.uint8))
        ]
        self._opened = True
        self.released = False
        self.props: dict[int, float] = {}
        _FakeCapture.instances.append(self)

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop, value) -> None:
        self.props[prop] = value

    def read(self):
        if not self.reads:
            return False, None
        item = self.reads.pop(0) if len(self.reads) > 1 else self.reads[0]
        return item

    def release(self) -> None:
        self.released = True


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    async def _poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)


class _NoRealSense:
    """Stands in for _RealSenseBackend on a machine with none plugged in."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("no RealSense in tests")


@pytest.fixture(autouse=True)
def fake_cv2(monkeypatch):
    _FakeCapture.instances.clear()
    monkeypatch.setattr("harp.vision.camera.cv2.VideoCapture", _FakeCapture)
    # 'auto' must fall through to the fake webcam even on a dev machine that
    # has a real RealSense attached.
    monkeypatch.setattr("harp.vision.camera._RealSenseBackend", _NoRealSense)
    yield _FakeCapture
    _FakeCapture.instances.clear()


async def test_start_populates_latest_frame():
    cam = Camera(device=0)
    await cam.start()
    try:
        await _wait_until(lambda: cam.latest() is not None)
        frame = cam.latest()
        assert frame.shape == (2, 2, 3)
    finally:
        await cam.stop()


async def test_latest_returns_none_before_any_frame():
    cam = Camera(device=0)
    assert cam.latest() is None


async def test_latest_returns_a_copy_not_the_shared_buffer():
    cam = Camera(device=0)
    await cam.start()
    try:
        await _wait_until(lambda: cam.latest() is not None)
        frame = cam.latest()
        frame[0, 0, 0] = 255
        assert cam.latest()[0, 0, 0] != 255
    finally:
        await cam.stop()


async def test_stop_releases_the_device():
    cam = Camera(device=0)
    await cam.start()
    fake = _FakeCapture.instances[0]
    await cam.stop()
    assert fake.released
    assert threading.active_count() >= 1  # sanity: didn't crash the process


async def test_auto_prefers_realsense_when_detected(monkeypatch, fake_cv2):
    class FakeRealSense:
        name = "realsense"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def read(self):
            return np.full((2, 2, 3), 7, dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr("harp.vision.camera._RealSenseBackend", FakeRealSense)
    cam = Camera(device=0)
    await cam.start()
    try:
        await _wait_until(lambda: cam.latest() is not None)
        assert cam.latest()[0, 0, 0] == 7  # frames come from the RealSense...
        assert not fake_cv2.instances      # ...and the webcam was never opened
    finally:
        await cam.stop()


async def test_realsense_dropout_falls_back_to_webcam(monkeypatch, fake_cv2):
    class FlakyRealSense:
        name = "realsense"
        alive = True

        def __init__(self, *args, **kwargs) -> None:
            if not FlakyRealSense.alive:
                raise RuntimeError("unplugged")

        def read(self):
            if not FlakyRealSense.alive:
                return None
            return np.full((2, 2, 3), 7, dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr("harp.vision.camera._RealSenseBackend", FlakyRealSense)
    cam = Camera(device=0)
    await cam.start()
    try:
        await _wait_until(lambda: cam.latest() is not None)
        # Unplug: reads start failing and the RealSense won't reopen, so the
        # reconnect loop must hand `auto` over to the webcam.
        FlakyRealSense.alive = False
        await _wait_until(lambda: len(fake_cv2.instances) >= 1)
        await _wait_until(
            lambda: cam.latest() is not None and cam.latest()[0, 0, 0] == 1
        )
    finally:
        await cam.stop()


async def test_read_failure_triggers_reconnect(fake_cv2):
    cam = Camera(device=0)
    await cam.start()
    try:
        await _wait_until(lambda: cam.latest() is not None)
        first = _FakeCapture.instances[0]
        # Simulate the device dropping out.
        first.reads = [(False, None)]

        await _wait_until(lambda: len(_FakeCapture.instances) >= 2)
        second = _FakeCapture.instances[1]
        await _wait_until(lambda: second in _FakeCapture.instances and first.released)
        assert first.released
    finally:
        await cam.stop()
