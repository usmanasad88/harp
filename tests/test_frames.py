"""jpeg_snapshot: the dashboard's read-only view of the shared camera.
No real hardware — a fake camera returning an in-memory frame."""

from __future__ import annotations

import numpy as np

from harp.vision.frames import Overlay, jpeg_snapshot


class FakeCamera:
    def __init__(self, frame: np.ndarray | None) -> None:
        self._frame = frame

    def latest(self) -> np.ndarray | None:
        return self._frame


def test_snapshot_encodes_latest_frame_as_jpeg():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 2] = 200  # non-trivial content so the encoder has work to do
    jpeg = jpeg_snapshot(FakeCamera(frame))
    assert jpeg is not None
    assert jpeg[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_snapshot_is_none_before_first_frame():
    assert jpeg_snapshot(FakeCamera(None)) is None


def _frame() -> np.ndarray:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 2] = 200
    return frame


def test_overlay_is_drawn_on_the_frame():
    # Separate frame copies: the real Camera.latest() hands out copies, but
    # this fake returns the same array, and drawing mutates it.
    plain = jpeg_snapshot(FakeCamera(_frame()))
    boxed = jpeg_snapshot(
        FakeCamera(_frame()),
        overlays=(lambda: Overlay(label="Open_Palm", box=(0.2, 0.2, 0.8, 0.8)),),
    )
    assert boxed is not None
    assert boxed != plain


def test_none_overlay_draws_nothing():
    plain = jpeg_snapshot(FakeCamera(_frame()))
    untouched = jpeg_snapshot(FakeCamera(_frame()), overlays=(lambda: None,))
    assert untouched == plain
