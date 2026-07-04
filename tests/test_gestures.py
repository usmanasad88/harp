"""GestureRecognizer.process_frame wires MediaPipe's pretrained GestureRecognizer
into a hold+cooldown debounce and the bus. MediaPipe is faked here — tests need
neither the model nor a webcam, just the debounce/mapping logic."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import harp.vision.gestures as gestures_module
from harp.core.bus import Bus
from harp.core.events import GestureDetected


def _category(name: str) -> SimpleNamespace:
    return SimpleNamespace(category_name=name)


def _landmark(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


# Where the fake hand sits in the frame, normalized (matches _hand_box output).
_FAKE_HAND_BOX = (0.2, 0.3, 0.6, 0.8)


class _FakeResult(SimpleNamespace):
    pass


class _FakeGestureRecognizer:
    instances: list["_FakeGestureRecognizer"] = []

    def __init__(self) -> None:
        self.next_gesture: str | None = None
        _FakeGestureRecognizer.instances.append(self)

    @classmethod
    def create_from_options(cls, options) -> "_FakeGestureRecognizer":
        return cls()

    def recognize(self, mp_image):
        if self.next_gesture is None:
            return _FakeResult(gestures=[], hand_landmarks=[])
        x1, y1, x2, y2 = _FAKE_HAND_BOX
        return _FakeResult(
            gestures=[[_category(self.next_gesture)]],
            hand_landmarks=[[_landmark(x1, y1), _landmark(x2, y2), _landmark(x1, y2)]],
        )


class _FakeCamera:
    pass  # process_frame takes the frame directly; camera is unused here


@pytest.fixture(autouse=True)
def fake_mediapipe(monkeypatch):
    _FakeGestureRecognizer.instances.clear()
    monkeypatch.setattr(gestures_module.vision, "GestureRecognizer", _FakeGestureRecognizer)
    monkeypatch.setattr(gestures_module, "_ensure_model", lambda *a, **k: "/dev/null")
    yield
    _FakeGestureRecognizer.instances.clear()


async def _next(stream, timeout: float = 1.0):
    return await asyncio.wait_for(anext(stream), timeout=timeout)


def _recognizer(bus: Bus):
    return gestures_module.GestureRecognizer(bus, _FakeCamera())


def _frame() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


async def _feed(rec, gesture: str | None, n: int, start_t: float = 0.0) -> list[bool]:
    rec._recognizer.next_gesture = gesture
    return [await rec.process_frame(_frame(), now=start_t + i * 0.1) for i in range(n)]


async def test_no_hand_never_fires():
    bus = Bus()
    rec = _recognizer(bus)
    fired = await _feed(rec, None, 10)
    assert not any(fired)


async def test_non_greeting_gesture_never_fires():
    bus = Bus()
    rec = _recognizer(bus)
    fired = await _feed(rec, "Thumb_Up", 10)
    assert not any(fired)


async def test_brief_open_palm_below_hold_threshold_does_not_fire():
    bus = Bus()
    rec = _recognizer(bus)
    fired = await _feed(rec, "Open_Palm", gestures_module._HOLD_FRAMES - 1)
    assert not any(fired)


async def test_sustained_open_palm_fires_exactly_once_and_publishes():
    bus = Bus()
    stream = bus.subscribe(GestureDetected)
    rec = _recognizer(bus)

    fired = await _feed(rec, "Open_Palm", gestures_module._HOLD_FRAMES + 10)

    assert fired.count(True) == 1
    event = await _next(stream)
    assert event == GestureDetected(kind="wave")


async def test_continuing_to_hold_does_not_refire():
    bus = Bus()
    rec = _recognizer(bus)
    await _feed(rec, "Open_Palm", gestures_module._HOLD_FRAMES + 5)

    more = await _feed(
        rec, "Open_Palm", 20, start_t=(gestures_module._HOLD_FRAMES + 5) * 0.1
    )
    assert not any(more)


async def test_overlay_reflects_the_last_sighting():
    bus = Bus()
    rec = _recognizer(bus)
    assert rec.overlay(now=0.0) is None  # nothing processed yet

    await _feed(rec, "Thumb_Up", 1)
    overlay = rec.overlay(now=0.1)
    assert overlay is not None
    assert overlay.label == "Thumb_Up"
    assert overlay.box == _FAKE_HAND_BOX


async def test_overlay_goes_stale_without_fresh_frames():
    bus = Bus()
    rec = _recognizer(bus)
    await _feed(rec, "Open_Palm", 1)
    assert rec.overlay(now=gestures_module._OVERLAY_TTL_S + 0.1) is None


async def test_overlay_clears_when_the_hand_leaves():
    bus = Bus()
    rec = _recognizer(bus)
    await _feed(rec, "Open_Palm", 1)
    await _feed(rec, None, 1, start_t=0.1)
    assert rec.overlay(now=0.1) is None


async def test_release_then_re_hold_after_cooldown_fires_again():
    bus = Bus()
    rec = _recognizer(bus)
    t = 0.0

    first = await _feed(rec, "Open_Palm", gestures_module._HOLD_FRAMES + 1, start_t=t)
    assert first.count(True) == 1
    t += (gestures_module._HOLD_FRAMES + 1) * 0.1

    # Gesture drops away, then well past both the hold requirement and cooldown.
    released = await _feed(rec, None, 3, start_t=t)
    assert not any(released)
    t += 3 * 0.1 + gestures_module._COOLDOWN_S

    second = await _feed(rec, "Open_Palm", gestures_module._HOLD_FRAMES + 1, start_t=t)
    assert second.count(True) == 1
