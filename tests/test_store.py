"""MemoryStore: JSON-per-person persistence. All against tmp_path — the store
is deliberately light enough to test for real, nothing to fake."""

from __future__ import annotations

import json

import numpy as np

from harp.memory.store import MemoryStore


def _embedding(fill: float = 1.0) -> np.ndarray:
    return np.full(4, fill, dtype=np.float32)


def test_upsert_writes_a_json_file_and_returns_the_given_id(tmp_path):
    store = MemoryStore(tmp_path)
    person_id = store.upsert_person(
        {"person_id": "usman-asad", "name": "Usman Asad", "role": "organizer"}
    )
    assert person_id == "usman-asad"
    data = json.loads((tmp_path / "usman-asad.json").read_text())
    assert data["name"] == "Usman Asad"
    assert data["role"] == "organizer"


def test_generated_ids_slug_the_name_and_avoid_collisions(tmp_path):
    store = MemoryStore(tmp_path)
    first = store.upsert_person({"name": "Ali Khan"})
    second = store.upsert_person({"name": "Ali Khan"})  # a different Ali Khan
    assert first == "ali-khan"
    assert second == "ali-khan-2"


def test_single_embedding_shorthand(tmp_path):
    store = MemoryStore(tmp_path)
    person_id = store.upsert_person({"name": "Ada", "embedding": _embedding(2.0)})
    record = store.get(person_id)
    assert len(record.embeddings) == 1
    np.testing.assert_array_equal(record.embeddings[0], _embedding(2.0))


def test_records_survive_a_reload(tmp_path):
    person_id = MemoryStore(tmp_path).upsert_person(
        {"name": "Ada", "notes": "likes tea", "embeddings": [_embedding()]}
    )
    record = MemoryStore(tmp_path).get(person_id)
    assert record.name == "Ada"
    assert record.notes == "likes tea"
    assert record.embeddings[0].dtype == np.float32
    np.testing.assert_array_equal(record.embeddings[0], _embedding())


def test_re_enrollment_replaces_fields_but_keeps_summaries(tmp_path):
    store = MemoryStore(tmp_path)
    person_id = store.upsert_person({"name": "Ada", "embeddings": [_embedding(1.0)]})
    store.add_summary(person_id, "asked about hall B")

    store.upsert_person(
        {"person_id": person_id, "name": "Ada L.", "embeddings": [_embedding(9.0)]}
    )

    record = MemoryStore(tmp_path).get(person_id)  # reload: check persistence too
    assert record.name == "Ada L."
    np.testing.assert_array_equal(record.embeddings[0], _embedding(9.0))
    assert [s["text"] for s in record.summaries] == ["asked about hall B"]
    assert record.summaries[0]["ts"]


def test_people_lists_everyone(tmp_path):
    store = MemoryStore(tmp_path)
    store.upsert_person({"name": "Ada"})
    store.upsert_person({"name": "Grace"})
    assert sorted(p.person_id for p in store.people()) == ["ada", "grace"]
