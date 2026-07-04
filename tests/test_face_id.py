"""FaceID's slow loop does three independently-testable things: pick the most
prominent face out of a frame, publish PersonIdentified only when who-is-in-
frame changes (unknowns reported, never stored), and record an overlay for the
dashboard camera view. Real InsightFace detection and the real memory matcher
are both faked here — this tests the orchestration, not the model."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import harp.vision.face_id as face_id_module
from harp.core.bus import Bus
from harp.core.events import PersonIdentified
from harp.vision.face_id import UNKNOWN_ID, FaceID


def _face(bbox, embedding=None) -> SimpleNamespace:
    return SimpleNamespace(
        bbox=bbox,
        normed_embedding=embedding if embedding is not None else np.zeros(512, dtype=np.float32),
    )


_FRAME = np.zeros((100, 200, 3), dtype=np.uint8)  # height 100, width 200


class _FakeFaceAnalysis:
    """Stands in for insightface.app.FaceAnalysis."""

    instances: list["_FakeFaceAnalysis"] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self.faces: list[SimpleNamespace] = []
        self.prepared_with: dict | None = None
        _FakeFaceAnalysis.instances.append(self)

    def prepare(self, ctx_id: int, det_thresh: float) -> None:
        self.prepared_with = {"ctx_id": ctx_id, "det_thresh": det_thresh}

    def get(self, frame):
        return self.faces


class _FakeCamera:
    def __init__(self, frame=None) -> None:
        self._frame = frame

    def latest(self):
        return self._frame


class _FakeStore:
    def __init__(self) -> None:
        self.upserted: list[object] = []
        self.people: dict[str, SimpleNamespace] = {}

    def upsert_person(self, person) -> str:  # pragma: no cover - must NOT be called
        self.upserted.append(person)
        return "should-not-happen"

    def get(self, person_id: str):
        return self.people[person_id]


@pytest.fixture(autouse=True)
def fake_face_analysis(monkeypatch):
    _FakeFaceAnalysis.instances.clear()
    monkeypatch.setattr(face_id_module, "FaceAnalysis", _FakeFaceAnalysis)
    yield _FakeFaceAnalysis
    _FakeFaceAnalysis.instances.clear()


async def _next(stream, timeout: float = 1.0):
    return await asyncio.wait_for(anext(stream), timeout=timeout)


async def test_no_face_clears_current_and_publishes_nothing():
    bus = Bus()
    stream = bus.subscribe()
    fid = FaceID(bus, _FakeCamera(), _FakeStore())
    _FakeFaceAnalysis.instances[0].faces = []

    result = await fid.process_frame(_FRAME, now=0.0)

    assert result is None
    assert fid.current is None
    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.05)


async def test_picks_the_largest_face_by_bbox_area(monkeypatch):
    bus = Bus()
    fid = FaceID(bus, _FakeCamera(), _FakeStore())

    small = _face(bbox=(0, 0, 10, 10), embedding=np.full(512, 1.0, dtype=np.float32))
    large = _face(bbox=(0, 0, 100, 100), embedding=np.full(512, 2.0, dtype=np.float32))
    _FakeFaceAnalysis.instances[0].faces = [small, large]

    seen: list[np.ndarray] = []

    def fake_match(embedding, store):
        seen.append(embedding)
        return (None, False, 0.0)

    monkeypatch.setattr(face_id_module.matcher, "match", fake_match)

    await fid.process_frame(_FRAME, now=0.0)

    assert seen[0] is large.normed_embedding


async def test_known_person_resolves_name_and_does_not_upsert(monkeypatch):
    bus = Bus()
    stream = bus.subscribe(PersonIdentified)
    store = _FakeStore()
    store.people["p1"] = SimpleNamespace(name="Ada")
    fid = FaceID(bus, _FakeCamera(), store)
    _FakeFaceAnalysis.instances[0].faces = [_face(bbox=(0, 0, 10, 10))]

    monkeypatch.setattr(face_id_module.matcher, "match", lambda embedding, store: ("p1", True, 0.92))

    event = await fid.process_frame(_FRAME, now=0.0)

    assert event == PersonIdentified(person_id="p1", name="Ada", is_known=True, confidence=0.92)
    assert store.upserted == []
    assert fid.current == event
    assert await _next(stream) == event


async def test_unknown_face_is_reported_but_never_stored(monkeypatch):
    bus = Bus()
    store = _FakeStore()
    fid = FaceID(bus, _FakeCamera(), store)
    _FakeFaceAnalysis.instances[0].faces = [_face(bbox=(0, 0, 10, 10))]

    monkeypatch.setattr(face_id_module.matcher, "match", lambda embedding, store: (None, False, 0.1))

    event = await fid.process_frame(_FRAME, now=0.0)

    assert event.person_id == UNKNOWN_ID
    assert event.is_known is False
    assert event.name is None
    assert store.upserted == []  # report-only: strangers are never persisted


async def test_same_person_across_passes_publishes_once(monkeypatch):
    bus = Bus()
    stream = bus.subscribe(PersonIdentified)
    store = _FakeStore()
    store.people["p1"] = SimpleNamespace(name="Ada")
    fid = FaceID(bus, _FakeCamera(), store)
    _FakeFaceAnalysis.instances[0].faces = [_face(bbox=(0, 0, 10, 10))]
    monkeypatch.setattr(face_id_module.matcher, "match", lambda embedding, store: ("p1", True, 0.9))

    first = await fid.process_frame(_FRAME, now=0.0)
    second = await fid.process_frame(_FRAME, now=1.5)

    assert first is not None
    assert second is None  # unchanged identity: quiet
    await _next(stream)
    with pytest.raises(asyncio.TimeoutError):
        await _next(stream, timeout=0.05)


async def test_reappearing_after_absence_publishes_again(monkeypatch):
    bus = Bus()
    stream = bus.subscribe(PersonIdentified)
    store = _FakeStore()
    store.people["p1"] = SimpleNamespace(name="Ada")
    fid = FaceID(bus, _FakeCamera(), store)
    fake = _FakeFaceAnalysis.instances[0]
    monkeypatch.setattr(face_id_module.matcher, "match", lambda embedding, store: ("p1", True, 0.9))

    fake.faces = [_face(bbox=(0, 0, 10, 10))]
    assert await fid.process_frame(_FRAME, now=0.0) is not None
    fake.faces = []  # they leave the frame...
    assert await fid.process_frame(_FRAME, now=1.5) is None
    fake.faces = [_face(bbox=(0, 0, 10, 10))]  # ...and come back
    assert await fid.process_frame(_FRAME, now=3.0) is not None

    assert (await _next(stream)).person_id == "p1"
    assert (await _next(stream)).person_id == "p1"


async def test_identity_change_publishes_the_new_person(monkeypatch):
    bus = Bus()
    stream = bus.subscribe(PersonIdentified)
    store = _FakeStore()
    store.people["p1"] = SimpleNamespace(name="Ada")
    store.people["p2"] = SimpleNamespace(name="Grace")
    fid = FaceID(bus, _FakeCamera(), store)
    _FakeFaceAnalysis.instances[0].faces = [_face(bbox=(0, 0, 10, 10))]

    answers = iter([("p1", True, 0.9), ("p2", True, 0.8)])
    monkeypatch.setattr(face_id_module.matcher, "match", lambda embedding, store: next(answers))

    await fid.process_frame(_FRAME, now=0.0)
    event = await fid.process_frame(_FRAME, now=1.5)

    assert event == PersonIdentified(person_id="p2", name="Grace", is_known=True, confidence=0.8)
    assert (await _next(stream)).person_id == "p1"
    assert (await _next(stream)).person_id == "p2"


async def test_overlay_reflects_sighting_then_goes_stale(monkeypatch):
    bus = Bus()
    store = _FakeStore()
    store.people["p1"] = SimpleNamespace(name="Ada")
    fid = FaceID(bus, _FakeCamera(), store)
    # Frame is 200x100 (w x h): a bbox of (20, 10, 100, 60) normalizes to
    # (0.1, 0.1, 0.5, 0.6).
    _FakeFaceAnalysis.instances[0].faces = [_face(bbox=(20, 10, 100, 60))]
    monkeypatch.setattr(face_id_module.matcher, "match", lambda embedding, store: ("p1", True, 0.9))

    await fid.process_frame(_FRAME, now=100.0)

    overlay = fid.overlay(now=100.5)
    assert overlay is not None
    assert overlay.label == "Ada 0.90"
    assert overlay.box == pytest.approx((0.1, 0.1, 0.5, 0.6))
    assert fid.overlay(now=100.0 + 60.0) is None  # stale: loop stopped, not the face


async def test_overlay_clears_when_face_leaves(monkeypatch):
    bus = Bus()
    store = _FakeStore()
    store.people["p1"] = SimpleNamespace(name="Ada")
    fid = FaceID(bus, _FakeCamera(), store)
    fake = _FakeFaceAnalysis.instances[0]
    monkeypatch.setattr(face_id_module.matcher, "match", lambda embedding, store: ("p1", True, 0.9))

    fake.faces = [_face(bbox=(0, 0, 10, 10))]
    await fid.process_frame(_FRAME, now=0.0)
    assert fid.overlay(now=0.1) is not None
    fake.faces = []
    await fid.process_frame(_FRAME, now=1.5)
    assert fid.overlay(now=1.6) is None
