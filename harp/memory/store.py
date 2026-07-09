"""Persistence for people and their memories — the who-has-HARP-met store.

Holds, per person: a stable id, face embedding(s), a name if known, memory
summaries, and any open follow-up intents. This is the single source of truth
that face-ID and the summarizer write to, and that matcher and triggers read
from.

Format: one JSON file per person in a directory — human-readable and
hand-editable on purpose (enrollment info is something the user curates), and
at HARP's scale (dozens of people, not millions) a database would be pure
overhead. Embeddings are stored inline as float lists. Writes are atomic
(temp file + rename) so a crash mid-write can't corrupt a record.

Two kinds of data with different lifecycles live in one record:
  - enrollment fields (name/role/notes/embeddings) — owned by
    scripts/enroll_people.py, replaced wholesale on re-enrollment;
  - summaries — accumulated interaction history, appended by the summarizer,
    preserved across re-enrollment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Fields upsert_person replaces when given; summaries are deliberately absent.
_ENROLLMENT_FIELDS = ("name", "role", "notes", "embeddings")


@dataclass
class PersonRecord:
    person_id: str
    name: str | None = None
    role: str | None = None            # e.g. organizer / speaker / guest
    notes: str = ""                    # model-facing, free text
    embeddings: list[np.ndarray] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)  # {"ts", "text"}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "person"


class MemoryStore:
    def __init__(self, path) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._people: dict[str, PersonRecord] = {}
        for file in sorted(self._path.glob("*.json")):
            record = self._load(file)
            self._people[record.person_id] = record

    def upsert_person(self, person: dict) -> str:
        """Create or update a person; return their stable person_id.

        Recognized keys: `person_id` (generated from the name if missing),
        the enrollment fields (`name`, `role`, `notes`, `embeddings`), and
        `embedding` as shorthand for a single-element `embeddings`. Provided
        enrollment fields replace the stored ones; summaries always survive.
        """
        person = dict(person)
        if "embedding" in person:
            person.setdefault("embeddings", []).append(person.pop("embedding"))
        person_id = person.pop("person_id", None) or self._new_id(person.get("name"))
        record = self._people.get(person_id) or PersonRecord(person_id=person_id)
        for key in _ENROLLMENT_FIELDS:
            if key in person:
                setattr(record, key, person[key])
        record.embeddings = [np.asarray(e, dtype=np.float32) for e in record.embeddings]
        self._people[person_id] = record
        self._save(record)
        return person_id

    def add_summary(self, person_id: str, summary: str, **extra: object) -> None:
        """Attach a memory summary to a person. `extra` fields (e.g. the
        summarizer's `follow_up` / `person_facts`) are stored alongside the
        text; readers must treat any key beyond ts/text as optional."""
        record = self.get(person_id)
        entry: dict = {"ts": datetime.now().isoformat(timespec="seconds"), "text": summary}
        entry.update({k: v for k, v in extra.items() if v})
        record.summaries.append(entry)
        self._save(record)

    def get(self, person_id: str) -> PersonRecord:
        """Load a person's record; KeyError for an id nobody handed out."""
        return self._people[person_id]

    def people(self) -> list[PersonRecord]:
        """Every stored person — what the matcher scans."""
        return list(self._people.values())

    def _new_id(self, name: str | None) -> str:
        base = _slugify(name) if name else "guest"
        person_id, n = base, 1
        while person_id in self._people:
            n += 1
            person_id = f"{base}-{n}"
        return person_id

    def _save(self, record: PersonRecord) -> None:
        payload = {
            "person_id": record.person_id,
            "name": record.name,
            "role": record.role,
            "notes": record.notes,
            "embeddings": [np.asarray(e, dtype=np.float32).tolist() for e in record.embeddings],
            "summaries": record.summaries,
        }
        tmp = self._path / f".{record.person_id}.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(self._path / f"{record.person_id}.json")

    @staticmethod
    def _load(file: Path) -> PersonRecord:
        data = json.loads(file.read_text())
        return PersonRecord(
            person_id=data["person_id"],
            name=data.get("name"),
            role=data.get("role"),
            notes=data.get("notes", ""),
            embeddings=[np.asarray(e, dtype=np.float32) for e in data.get("embeddings", [])],
            summaries=data.get("summaries", []),
        )
