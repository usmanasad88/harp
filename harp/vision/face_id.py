"""Face recognition → PersonIdentified (a continuous slow loop).

Watches the shared camera on a slow cadence (~1 pass / 1.5 s — InsightFace on
CPU costs a few hundred ms per pass, so this is deliberately much slower than
gestures) and identifies EVERYONE in frame, not just one person. It publishes
PersonIdentified whenever WHO is in frame changes: a new person appears (known
or unknown), or someone was there and now someone new joins them. Between
changes it stays quiet — subscribers care about identity changes, not a
1.5-second drumbeat of the same faces.

Detection + embedding lives here (InsightFace's buffalo_l model, CPU by
default). Identity resolution is delegated to memory/matcher + memory/store —
this module never decides who someone is, only recognizes a face and asks
memory to look it up. Unknown faces are REPORTED (person_id "unknown") but
never stored: auto-remembering strangers waits until real conversation
memories exist (decision 2026-07-02, see DEVLOG). Enrollment is the folder
convention + scripts/enroll_people.py.

Face-ID also serves as HARP's presence signal: each pass publishes
PresenceChanged whenever the "is anyone here, and how many" answer changes, so
the interaction end-rules can close a session when the person walks off (no
face for a while). A separate dedicated presence detector isn't needed while
the camera is watching for faces anyway.

Like gestures.py, each pass also records what it saw (a labelled box per face,
normalized coords) for the dashboard camera overlay, via overlays().
`current` exposes the most prominent person (largest face) so the voice bridge
can tell the model who it's mainly talking to at session open.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
from insightface.app import FaceAnalysis

from ..core.bus import Bus
from ..core.events import PersonIdentified, PresenceChanged
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

        # The most prominent identity (largest face); None when nobody's here.
        self._current: PersonIdentified | None = None
        # Who was in frame on the last pass: person_id -> their identity. Used
        # to publish only newcomers, not everyone, every pass. Two unknown faces
        # share the one "unknown" key — they're one bucket for event purposes,
        # but each still gets its own overlay box below.
        self._present: dict[str, PersonIdentified] = {}
        # One box+label per detected face, for the dashboard camera view.
        self._sightings: list[Overlay] = []
        self._last_sighting_at = float("-inf")
        # Last (present, count) we announced. Start "absent, 0" so an empty
        # frame at boot (nobody's here yet) is not a spurious transition — we
        # only publish PresenceChanged once someone actually appears or leaves.
        self._presence: tuple[bool, int] = (False, 0)

    @property
    def current(self) -> PersonIdentified | None:
        """The most prominent person in frame right now (largest face, as of
        the last pass); None when nobody is."""
        return self._current

    async def run(self) -> None:
        """Watch frames slowly, forever; publish when who-is-in-frame changes."""
        while True:
            frame = self._camera.latest()
            if frame is not None:
                await self.process_frame(frame, time.monotonic())
            await asyncio.sleep(self._poll_interval)

    async def process_frame(self, frame: np.ndarray, now: float) -> PersonIdentified | None:
        """One recognition pass over every face in frame.

        Publishes a PersonIdentified for each person who newly appeared since
        the last pass (known or unknown). Returns the most prominent person's
        event when THAT identity changed this pass (so direct callers/tests get
        the headline change), else None — the bus is the full output.

        Detection runs off-thread (it's a blocking few-hundred-ms model call and
        must not stall the shared event loop); the light per-face matching that
        follows stays on the loop.
        """
        faces = await asyncio.to_thread(self._detect, frame)
        if not faces:
            # Nobody in frame: forget who was here so a returning face counts as
            # a fresh appearance and publishes again.
            self._current = None
            self._present = {}
            self._sightings = []
            await self._publish_presence(present=False, count=0)
            return None

        # faces come back largest-first, so faces[0] is the most prominent (the
        # `primary`) and the first-seen identity for any given person_id wins
        # the bucket.
        identities: dict[str, PersonIdentified] = {}
        sightings: list[Overlay] = []
        primary: PersonIdentified | None = None
        for face in faces:
            event = self._identify(face)
            if primary is None:
                primary = event
            sightings.append(self._overlay_for(frame, face, event))
            identities.setdefault(event.person_id, event)

        assert primary is not None  # faces is non-empty here
        self._current = primary
        self._sightings = sightings
        self._last_sighting_at = now
        await self._publish_presence(present=True, count=len(faces))

        newcomers = [pid for pid in identities if pid not in self._present]
        for pid in newcomers:
            event = identities[pid]
            logger.info(
                "face-id: %s (known=%s, %.2f)",
                event.name or event.person_id, event.is_known, event.confidence,
            )
            await self._bus.publish(event)
        self._present = identities

        return primary if primary.person_id in newcomers else None

    def overlays(self, now: float | None = None) -> list[Overlay]:
        """A box+label for every face seen in the last processed frame, for the
        dashboard camera view; empty when there's no face or the sighting is
        stale (same time base as run(): time.monotonic)."""
        if now is None:
            now = time.monotonic()
        if now - self._last_sighting_at > _OVERLAY_TTL_S:
            return []
        return self._sightings

    async def _publish_presence(self, present: bool, count: int) -> None:
        """Announce presence only when the (present, count) answer changes, so
        subscribers (the end-rules) see a clean transition, not a 1.5 s drumbeat."""
        if (present, count) == self._presence:
            return
        self._presence = (present, count)
        await self._bus.publish(PresenceChanged(present=present, count=count))

    def _identify(self, face) -> PersonIdentified:
        """Resolve one detected face to an identity via memory, reporting (but
        never storing) strangers as UNKNOWN_ID."""
        person_id, is_known, confidence = matcher.match(face.normed_embedding, self._store)
        name = None
        if is_known:
            assert person_id is not None, "matcher.match must return an id for a known match"
            name = getattr(self._store.get(person_id), "name", None)
        else:
            person_id = UNKNOWN_ID
        return PersonIdentified(
            person_id=person_id, name=name, is_known=is_known, confidence=confidence
        )

    def _overlay_for(self, frame: np.ndarray, face, event: PersonIdentified) -> Overlay:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        box = (
            max(0.0, x1 / width),
            max(0.0, y1 / height),
            min(1.0, x2 / width),
            min(1.0, y2 / height),
        )
        label = event.name or event.person_id
        return Overlay(label=f"{label} {event.confidence:.2f}", box=box)

    def _detect(self, frame: np.ndarray) -> list:
        """Every detected face, largest first (by bbox area)."""
        faces = self._app.get(frame)
        return sorted(faces, key=self._bbox_area, reverse=True)

    @staticmethod
    def _bbox_area(face) -> float:
        x1, y1, x2, y2 = face.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
