"""search_memory: the live model looks HARP's own past up, by itself.

The mid-session counterpart to the pre-computed wake briefing: when a visitor
asks "do you remember me?" or refers to an earlier conversation, the model
calls this instead of guessing. It searches everything long-term memory
holds — every enrolled person's notes and interaction summaries (memory/store)
plus the guestbook of unknown-visitor interactions (memory/summarizer) — and
returns the best entries with who/when attached.

Ranking is plain query-token overlap (knowledge/retriever's tokenizer, so
both search tools agree on what a word is) with a name-match bonus; at this
corpus size (dozens of people, one entry per conversation) BM25 would be
ceremony. The corpus is rebuilt per call — it's tiny, and it means a summary
written seconds ago is immediately findable.

Shape mirrors knowledge/tools.py: `declarations(provider)` for
SessionConfig.tools, and an async handler returning a payload for
VoiceConnection.respond_tool ({"error"/"note": ...} instead of raising, so a
bad call degrades into the model apologizing).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import load_search_memory_tool_description
from ..knowledge.retriever import tokenize
from .store import MemoryStore

logger = logging.getLogger(__name__)

TOOL_NAME = "search_memory"

# Wording lives in prompts/search_memory_tool.md (see prompts/README.md).
_DESCRIPTION = load_search_memory_tool_description()

_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A few English keywords: a person's name, a topic, an event.",
        }
    },
    "required": ["query"],
}

# When a query token appears in the person's name it should outrank a passing
# mention in some summary body — a name is what visitors search each other by.
_NAME_BONUS = 2.0


def declarations(provider: str) -> list[dict]:
    """Tool declarations for SessionConfig.tools, in `provider`'s format."""
    base = {"name": TOOL_NAME, "description": _DESCRIPTION, "parameters": _PARAMETERS}
    if provider == "openai":
        return [{"type": "function", **base}]
    if provider == "gemini":
        return [{"function_declarations": [base]}]
    raise ValueError(f"unknown provider: {provider!r} (expected 'gemini' or 'openai')")


async def search_memory(
    store: MemoryStore, guestbook: Path, arguments: dict, k: int = 5
) -> Any:
    """Run the search and return up to `k` entries, best first, each as
    {"person", "when", "text"}."""
    terms = set(tokenize(str(arguments.get("query", ""))))
    if not terms:
        return {"note": "no matches found"}
    scored: list[tuple[float, dict]] = []
    for entry in _entries(store, guestbook):
        tokens = set(tokenize(entry["text"]))
        name_tokens = set(tokenize(entry["person"]))
        score = len(terms & tokens) + _NAME_BONUS * len(terms & name_tokens)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = [entry for _, entry in scored[:k]]
    return results if results else {"note": "no matches found"}


def _entries(store: MemoryStore, guestbook: Path) -> list[dict]:
    """The searchable corpus: one entry per stored summary, per person's
    enrollment notes, and per guestbook line."""
    entries: list[dict] = []
    for person in store.people():
        name = person.name or person.person_id
        if person.notes:
            text = f"Enrollment notes: {person.notes}"
            if person.role:
                text = f"Role: {person.role}. {text}"
            entries.append({"person": name, "when": "enrollment", "text": text})
        for summary in person.summaries:
            entries.append(
                {
                    "person": name,
                    "when": summary.get("ts", "unknown"),
                    "text": _summary_text(summary),
                }
            )
    for entry in _guestbook_lines(guestbook):
        entries.append(
            {
                "person": "unknown visitor(s)",
                "when": entry.get("ts", "unknown"),
                "text": _summary_text(
                    {**entry, "text": entry.get("summary", "")}
                ),
            }
        )
    return entries


def _summary_text(summary: dict) -> str:
    """One summary entry flattened to searchable, model-readable text."""
    parts = [str(summary.get("text", ""))]
    if summary.get("person_facts"):
        parts.append(f"About them: {summary['person_facts']}")
    if summary.get("follow_up"):
        parts.append(f"Open follow-up: {summary['follow_up']}")
    return " ".join(p for p in parts if p.strip())


def _guestbook_lines(guestbook: Path) -> list[dict]:
    try:
        raw = guestbook.read_text(encoding="utf-8")
    except OSError:
        return []  # no guestbook yet
    lines = []
    for line in raw.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            lines.append(entry)
    return lines
