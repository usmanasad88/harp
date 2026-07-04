"""Match a detected face against known people.

Given a face embedding from vision/face_id, find the closest stored person (if
any) so HARP knows whether this is a first meeting or a returning visitor, and
can pull that person's summaries + open follow-ups. Pure lookup over memory/store.

Cosine similarity against every stored embedding, best one wins if it clears
the threshold. InsightFace's `normed_embedding`s are unit-length, so the dot
product IS the cosine; the query is re-normalized defensively anyway. Brute
force on purpose: at HARP's scale (dozens of people, a handful of embeddings
each) this is microseconds — a vector index would be pure overhead.

Threshold: 0.4 is the usual verification cut-off for buffalo_l (ArcFace)
embeddings — same-person pairs typically land around 0.5–0.8, different people
below ~0.3. If live matching misbehaves, calibrate against the real webcam
with scripts/preview_face_id.py (it prints the similarity per face).
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.4


def match(embedding, store, threshold: float = DEFAULT_THRESHOLD) -> tuple[str | None, bool, float]:
    """Return (person_id, is_known, confidence) for a face `embedding`.

    `confidence` is the best cosine similarity found (floored at 0.0, and 0.0
    for an empty store). person_id is None when nothing clears the threshold —
    a face HARP doesn't know.
    """
    query = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if norm > 0:
        query = query / norm

    best_id: str | None = None
    best_sim = 0.0
    for person in store.people():
        for stored in person.embeddings:
            sim = float(np.dot(query, stored))
            if sim > best_sim:
                best_id, best_sim = person.person_id, sim

    if best_id is None or best_sim < threshold:
        return (None, False, best_sim)
    return (best_id, True, best_sim)
