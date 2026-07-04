"""Face recognition → PersonIdentified (a continuous slow loop).

Watches the shared camera on a slow cadence (~1 pass / 1.5 s — InsightFace on
CPU costs a few hundred ms per pass, so this is deliberately much slower than
gestures) and publishes PersonIdentified whenever WHO is in frame changes:
someone appears, a different face becomes the most prominent, or nobody was
there and now someone is. Between changes it stays quiet — subscribers care
about identity changes, not a 1.5-second drumbeat of the same answer.

Detection + embedding lives here (InsightFace's buffalo_l model, CPU by
default). Identity resolution is delegated to memory/matcher + memory/store —
this module never decides who someone is, only recognizes a face and asks
memory to look it up. Unknown faces are REPORTED (person_id "unknown") but
never stored: auto-remembering strangers waits until real conversation
memories exist (decision 2026-07-02, see DEVLOG). Enrollment is the folder
convention + scripts/enroll_people.py.

Like gestures.py, each pass also records what it saw (best-match label + face
box, normalized coords) for the dashboard camera overlay, via overlay().
`current` exposes the latest identification so the voice bridge can tell the
model who it's talking to at session open.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
from insightface.app import FaceAnalysis

from ..core.bus import Bus
from ..core.events import PersonIdentified
from ..memory import matcher
from ..memory.store import MemoryStore
from .camera import Camera
from .frames import Overlay

logger = logging.getLogger(__name__)

# Below this detector confidence, treat it as noise rather than a face.
_DET_THRESH = 0.5

# The person_id reported for a face that matches nobody in the store.
UNKNOWN_ID = "unknown"

# How long the last sighting stays drawable on the dashboard camera view.
# Must comfortably exceed the poll interval or the box flickers between
# passes; past this the overlay goes blank — the loop stopped, not the face.
_OVERLAY_TTL_S = 4.0


class FaceID:
    def __init__(
        self,
        bus: Bus,
        camera: Camera,
        store: MemoryStore,
        ctx_id: int = -1,
        poll_interval: float = 1.5,
    ) -> None:
        self._bus = bus
        self._camera = camera
        self._store = store
        self._poll_interval = poll_interval
        self._app = FaceAnalysis(name="buffalo_l")
        self._app.prepare(ctx_id=ctx_id, det_thresh=_DET_THRESH)

        # The latest identification; None when nobody is in frame.
        self._current: PersonIdentified | None = None
        self._last_sighting: Overlay | None = None
        self._last_sighting_at = float("-inf")

    @property
    def current(self) -> PersonIdentified | None:
        """Who is in frame right now (as of the last pass); None when nobody."""
        return self._current

    async def run(self) -> None:
        """Watch frames slowly, forever; publish when who-is-in-frame changes."""
        while True:
            frame = self._camera.latest()
            if frame is not None:
                await self.process_frame(frame, time.monotonic())
            await asyncio.sleep(self._poll_interval)

    async def process_frame(self, frame: np.ndarray, now: float) -> PersonIdentified | None:
        """One recognition pass; publishes + returns an event only on a change.

        Detection runs off-thread (it's a blocking few-hundred-ms model call
        and must not stall the shared event loop).
        """
        face = await asyncio.to_thread(self._largest_face, frame)
        if face is None:
            # Nobody in frame: forget the current identity so the same person
            # coming back counts as a fresh appearance and publishes again.
            self._current = None
            self._last_sighting = None
            return None

        person_id, is_known, confidence = matcher.match(face.normed_embedding, self._store)
        name = None
        if is_known:
            assert person_id is not None, "matcher.match must return an id for a known match"
            name = getattr(self._store.get(person_id), "name", None)
        else:
            person_id = UNKNOWN_ID
        self._note_sighting(frame, face, name or UNKNOWN_ID, confidence, now)

        if self._current is not None and self._current.person_id == person_id:
            return None  # same answer as the last pass — stay quiet

        event = PersonIdentified(
            person_id=person_id,
            name=name,
            is_known=is_known,
            confidence=confidence,
        )
        self._current = event
        logger.info("face-id: %s (known=%s, %.2f)", name or person_id, is_known, confidence)
        await self._bus.publish(event)
        return event

    def overlay(self, now: float | None = None) -> Overlay | None:
        """What face-ID saw in the last processed frame, for the dashboard
        camera view; None when there's no face or the sighting is stale
        (same time base as run(): time.monotonic)."""
        if now is None:
            now = time.monotonic()
        if self._last_sighting is None or now - self._last_sighting_at > _OVERLAY_TTL_S:
            return None
        return self._last_sighting

    def _note_sighting(
        self, frame: np.ndarray, face, label: str, confidence: float, now: float
    ) -> None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        box = (
            max(0.0, x1 / width),
            max(0.0, y1 / height),
            min(1.0, x2 / width),
            min(1.0, y2 / height),
        )
        self._last_sighting = Overlay(label=f"{label} {confidence:.2f}", box=box)
        self._last_sighting_at = now

    def _largest_face(self, frame: np.ndarray):
        """Return the most prominent detected face (by bbox area), or None."""
        faces = self._app.get(frame)
        if not faces:
            return None
        return max(faces, key=self._bbox_area)

    @staticmethod
    def _bbox_area(face) -> float:
        x1, y1, x2, y2 = face.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
