"""JPEG snapshots of the shared camera — the dashboard's read-only view.

The dashboard must stay decoupled from the vision layer, so it takes a plain
`snapshot() -> bytes | None` callable (wired by app.py, the composition root)
rather than a Camera. This module provides that callable's implementation.

Overlays: services processing frames (gestures today; presence and face-ID
later) contribute what they saw drawn over this view, as callables returning
an `Overlay` — a label plus a box in normalized coordinates, so a provider
never needs to know the snapshot frame's pixel size. The compositing happens
here; app.py decides which providers are wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from .camera import Camera

_JPEG_QUALITY = 75
_BOX_COLOR = (80, 220, 80)  # BGR


@dataclass(frozen=True)
class Overlay:
    """One thing a vision service saw, drawable over the camera view: a text
    label and a bounding box in normalized [0, 1] frame coordinates."""

    label: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


OverlayFn = Callable[[], "Overlay | None"]


def jpeg_snapshot(camera: Camera, overlays: Iterable[OverlayFn] = ()) -> bytes | None:
    """The freshest frame as JPEG bytes, or None if no frame has arrived yet.

    Each overlay provider is asked what it currently sees; None means "nothing
    right now" and draws nothing. Drawing happens on the copy `Camera.latest()`
    returns, never the capture buffer itself.
    """
    frame = camera.latest()
    if frame is None:
        return None
    for provider in overlays:
        overlay = provider()
        if overlay is not None:
            _draw(frame, overlay)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _draw(frame: np.ndarray, overlay: Overlay) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = overlay.box
    top_left = (int(x1 * width), int(y1 * height))
    bottom_right = (int(x2 * width), int(y2 * height))
    cv2.rectangle(frame, top_left, bottom_right, _BOX_COLOR, 2)
    cv2.putText(
        frame,
        overlay.label,
        (top_left[0], max(16, top_left[1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _BOX_COLOR,
        2,
    )
