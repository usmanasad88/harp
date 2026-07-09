"""The live model's memory/vision tools. search_memory must rank a name hit
above body mentions and see all three corpora (summaries, enrollment notes,
guestbook); describe_scene must degrade to an {"error": ...} payload — never
an exception — when the camera or the helper is down (a raise here would kill
the live session mid-turn)."""

from __future__ import annotations

import json
from pathlib import Path

from harp.memory import tools as memory_tools
from harp.memory.agent import GeminiAgent, RateLimiter
from harp.memory.store import MemoryStore
from harp.vision import describe


def _store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "people")
    store.upsert_person(
        {"person_id": "usman", "name": "Usman", "role": "organizer", "notes": "likes robots"}
    )
    store.add_summary(
        "usman", "Asked where hall B is.", follow_up="did he find hall B",
        person_facts="from NUST",
    )
    store.upsert_person({"person_id": "sara", "name": "Sara"})
    store.add_summary("sara", "Talked about the drone demo schedule.")
    return store


async def test_search_memory_covers_people_notes_and_guestbook(tmp_path: Path):
    store = _store(tmp_path)
    guestbook = tmp_path / "guestbook.jsonl"
    guestbook.write_text(
        json.dumps({"ts": "2026-07-08", "summary": "A child asked about the dolphins."}) + "\n",
        encoding="utf-8",
    )

    hits = await memory_tools.search_memory(store, guestbook, {"query": "Usman"})
    assert hits[0]["person"] == "Usman"  # name match outranks body mentions

    hits = await memory_tools.search_memory(store, guestbook, {"query": "drone schedule"})
    assert hits[0]["person"] == "Sara"

    hits = await memory_tools.search_memory(store, guestbook, {"query": "NUST"})
    assert hits[0]["person"] == "Usman"  # person_facts are searchable too

    hits = await memory_tools.search_memory(store, guestbook, {"query": "dolphins"})
    assert hits[0]["person"] == "unknown visitor(s)"

    assert await memory_tools.search_memory(store, guestbook, {"query": "zzzz qqqq"}) == {
        "note": "no matches found"
    }
    assert await memory_tools.search_memory(store, guestbook, {"query": ""}) == {
        "note": "no matches found"
    }


async def test_search_memory_works_without_a_guestbook(tmp_path: Path):
    hits = await memory_tools.search_memory(
        _store(tmp_path), tmp_path / "never-written.jsonl", {"query": "hall"}
    )
    assert hits[0]["person"] == "Usman"


async def test_describe_scene_happy_path_passes_frame_and_focus():
    captured = {}

    async def caller(prompt, image, json_response):
        captured["prompt"], captured["image"] = prompt, image
        return "Two people, one waving."

    agent = GeminiAgent("m", RateLimiter(10), caller=caller)
    out = await describe.describe_scene(
        agent, lambda: b"jpeg", {"focus": "how many people"}
    )
    assert out == {"description": "Two people, one waving."}
    assert captured["image"] == b"jpeg" and "how many people" in captured["prompt"]


async def test_describe_scene_degrades_to_error_payloads():
    async def down(prompt, image, json_response):
        raise RuntimeError("quota")

    agent = GeminiAgent("m", RateLimiter(10), caller=down)
    assert "error" in await describe.describe_scene(agent, lambda: None, {})  # no frame
    assert "error" in await describe.describe_scene(agent, lambda: b"jpeg", {})  # helper down
