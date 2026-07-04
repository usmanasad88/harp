"""Gesture recognition → GestureDetected (a proactive cue).

While idle, a raised open palm should be enough to start a conversation — no
wake word needed (PLAN.md calls this "palm-gesture recognition"). Built on
MediaPipe's pretrained GestureRecognizer — the same canned classifier aura's
gesture_monitor.py uses (Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down,
Thumb_Up, Victory, ILoveYou) — rather than tracking raw hand-landmark motion
ourselves: there's no literal "wave" category, but a held-up Open_Palm is a
natural, reliably-classified stand-in, and the pretrained model is far more
robust than a hand-rolled heuristic on top of raw landmarks would be.

Debounced two ways: the gesture must hold for several consecutive frames
before it counts (ignores single-frame misclassifications), and once it fires
it won't fire again for the same continuous hold — only after the gesture
changes away and a cooldown passes — so one physical gesture produces one
event, not a pulse every frame it's held.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.request

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ..core.bus import Bus
from ..core.events import GestureDetected
from .camera import Camera
from .frames import Overlay

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
_MODEL_PATH = os.path.expanduser("~/.cache/mediapipe/gesture_recognizer.task")

# Canned MediaPipe categories that count as a proactive greeting cue.
_GREETING_GESTURES = {"Open_Palm"}

_HOLD_FRAMES = 4  # consecutive matching frames required before firing
_COOLDOWN_S = 2.0  # minimum gap between fires, in case of a flicker reset

# How long the last sighting stays drawable on the dashboard camera view.
# Past this the overlay goes blank — the recognizer stopped, not the hand.
_OVERLAY_TTL_S = 1.0


def _ensure_model(path: str = _MODEL_PATH) -> str:
    """Return a local path to gesture_recognizer.task, downloading it if needed."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info("downloading gesture recognizer model to %s", path)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


class GestureRecognizer:
    def __init__(self, bus: Bus, camera: Camera, poll_hz: float = 10.0) -> None:
        self._bus = bus
        self._camera = camera
        self._poll_interval = 1.0 / poll_hz
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model()),
            num_hands=1,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

        self._holding_gesture: str | None = None
        self._hold_count = 0
        self._fired_for_hold = False
        self._last_fired = float("-inf")

        self._last_sighting: Overlay | None = None
        self._last_sighting_at = float("-inf")

    async def run(self) -> None:
        """Watch frames and publish GestureDetected for recognized gestures."""
        while True:
            frame = self._camera.latest()
            if frame is not None:
                await self.process_frame(frame, time.monotonic())
            await asyncio.sleep(self._poll_interval)

    async def process_frame(self, frame: np.ndarray, now: float) -> bool:
        """Run one recognition pass; publishes + returns True on a confirmed cue."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._recognizer.recognize(mp_image)
        gesture = self._top_gesture(result)
        self._note_sighting(result, gesture, now)

        if gesture == self._holding_gesture:
            self._hold_count += 1
        else:
            self._holding_gesture = gesture
            self._hold_count = 1
            self._fired_for_hold = False

        ready = (
            gesture in _GREETING_GESTURES
            and self._hold_count >= _HOLD_FRAMES
            and not self._fired_for_hold
            and now - self._last_fired >= _COOLDOWN_S
        )
        if not ready:
            return False

        self._fired_for_hold = True
        self._last_fired = now
        logger.info("gesture cue: %s", gesture)
        await self._bus.publish(GestureDetected(kind="wave"))
        return True

    def overlay(self, now: float | None = None) -> Overlay | None:
        """What the recognizer saw in the last processed frame, for the
        dashboard camera view; None when there's no hand or the sighting is
        stale (same time base as run(): time.monotonic)."""
        if now is None:
            now = time.monotonic()
        if self._last_sighting is None or now - self._last_sighting_at > _OVERLAY_TTL_S:
            return None
        return self._last_sighting

    def _note_sighting(self, result, gesture: str | None, now: float) -> None:
        box = self._hand_box(result)
        if box is None:
            self._last_sighting = None
            return
        self._last_sighting = Overlay(label=gesture or "hand", box=box)
        self._last_sighting_at = now

    @staticmethod
    def _hand_box(result) -> tuple[float, float, float, float] | None:
        """Bounding box of the first detected hand in normalized coordinates,
        from MediaPipe's 21 landmarks; None when no hand is in frame."""
        hands = getattr(result, "hand_landmarks", None)
        if not hands:
            return None
        xs = [lm.x for lm in hands[0]]
        ys = [lm.y for lm in hands[0]]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _top_gesture(result) -> str | None:
        if not result.gestures or not result.gestures[0]:
            return None
        return result.gestures[0][0].category_name
