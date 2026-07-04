"""Webcam presence detector → publishes PresenceChanged.

Loop: pull frames from the shared camera (harp/vision/camera.py), run a light
human check (motion / background-subtraction / a small person detector), and
publish PresenceChanged only on a *transition* (present ↔ absent), debounced so a
one-frame flicker doesn't thrash the state machine.

To build:
  - pick a cheap detector (MediaPipe / a tiny YOLO / OpenCV HOG),
  - debounce with hysteresis (a stricter threshold to enter than to leave),
  - share the ONE camera with vision/face-ID — never open the device twice.
"""

from __future__ import annotations

from ..core.bus import Bus


class PresenceDetector:
    def __init__(self, bus: Bus) -> None:
        self._bus = bus

    async def run(self) -> None:
        """Watch the camera and publish PresenceChanged on transitions."""
        raise NotImplementedError
