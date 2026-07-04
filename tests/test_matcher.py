"""matcher.match: cosine similarity over the store with a threshold. Uses a
real MemoryStore on tmp_path — the matcher is a pure lookup over it."""

from __future__ import annotations

import numpy as np

from harp.memory import matcher
from harp.memory.store import MemoryStore


def _unit(direction: int) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    v[direction] = 1.0
    return v


def test_empty_store_is_unknown(tmp_path):
    store = MemoryStore(tmp_path)
    assert matcher.match(_unit(0), store) == (None, False, 0.0)


def test_close_embedding_matches_with_confidence(tmp_path):
    store = MemoryStore(tmp_path)
    store.upsert_person({"person_id": "ada", "name": "Ada", "embeddings": [_unit(0)]})

    person_id, is_known, confidence = matcher.match(_unit(0), store)

    assert (person_id, is_known) == ("ada", True)
    assert confidence > 0.99


def test_below_threshold_is_unknown_but_reports_similarity(tmp_path):
    store = MemoryStore(tmp_path)
    store.upsert_person({"person_id": "ada", "embeddings": [_unit(0)]})

    # ~0.3 cosine to the stored embedding: a stranger-ish score.
    query = 0.3 * _unit(0) + np.sqrt(1 - 0.09) * _unit(1)
    person_id, is_known, confidence = matcher.match(query, store)

    assert (person_id, is_known) == (None, False)
    assert 0.25 < confidence < 0.35


def test_best_of_several_people_and_embeddings_wins(tmp_path):
    store = MemoryStore(tmp_path)
    store.upsert_person({"person_id": "ada", "embeddings": [_unit(0), _unit(1)]})
    store.upsert_person({"person_id": "grace", "embeddings": [_unit(2)]})

    person_id, is_known, _ = matcher.match(_unit(2), store)
    assert (person_id, is_known) == ("grace", True)

    person_id, is_known, _ = matcher.match(_unit(1), store)
    assert (person_id, is_known) == ("ada", True)


def test_unnormalized_query_still_matches(tmp_path):
    store = MemoryStore(tmp_path)
    store.upsert_person({"person_id": "ada", "embeddings": [_unit(0)]})

    _, is_known, confidence = matcher.match(5.0 * _unit(0), store)

    assert is_known
    assert confidence > 0.99
