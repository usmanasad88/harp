"""Deterministic digest of one interaction transcript — the pre-LLM parse.

Before the summarizer spends a (rate-limited) model call, everything that can
be extracted mechanically from the transcript IS extracted mechanically: who
was there, when, why it started and ended, what the visitor asked the
knowledge base about, how much was actually said. The model then only does
what needs a model — summarizing content and spotting follow-up intents — and
the prompt hands it these facts instead of making it infer them.

Owns the record vocabulary of the transcript files memory/logger writes
(kind: start / person / turn / tool / end), so logger and summarizer agree by
importing this module, not each other.
"""

from __future__ import annotations

from typing import Iterable

# The `kind` values a transcript file may contain (see module docstring).
KIND_START = "start"
KIND_PERSON = "person"
KIND_TURN = "turn"
KIND_TOOL = "tool"
KIND_END = "end"


def digest(records: Iterable[dict]) -> dict:
    """Fold a transcript's records into the mechanical facts of the
    interaction. Tolerates a crashed recording (no end record) and unknown
    record kinds (skipped — the vocabulary may grow)."""
    started = ended = None
    started_t = ended_t = None
    wake_reason = end_reason = ""
    participants: list[dict] = []
    seen_ids: set[str] = set()
    turns: list[dict] = []
    tool_queries: list[str] = []

    for record in records:
        kind = record.get("kind")
        if kind == KIND_START:
            started = record.get("ts")
            started_t = record.get("t")
            wake_reason = record.get("reason", "")
        elif kind == KIND_PERSON:
            person_id = record.get("person_id", "")
            if person_id and person_id not in seen_ids:
                seen_ids.add(person_id)
                participants.append(
                    {
                        "person_id": person_id,
                        "name": record.get("name"),
                        "is_known": bool(record.get("is_known")),
                    }
                )
        elif kind == KIND_TURN:
            text = str(record.get("text", "")).strip()
            if text:
                turns.append({"who": record.get("who", "user"), "text": text})
        elif kind == KIND_TOOL:
            query = str((record.get("arguments") or {}).get("query", "")).strip()
            if query:
                tool_queries.append(query)
        elif kind == KIND_END:
            ended = record.get("ts")
            ended_t = record.get("t")
            end_reason = record.get("reason", "")

    duration = None
    if isinstance(started_t, (int, float)) and isinstance(ended_t, (int, float)):
        duration = max(0.0, round(ended_t - started_t, 1))

    return {
        "started": started,
        "ended": ended,
        "duration_seconds": duration,
        "wake_reason": wake_reason,
        "end_reason": end_reason,
        "participants": participants,
        "turns": turns,
        "user_turns": sum(1 for t in turns if t["who"] == "user"),
        "tool_queries": tool_queries,
    }


def render_facts(d: dict) -> str:
    """The digest as the prompt's facts block — one deterministic line per
    fact, so the summarizer model never has to infer them from the transcript."""
    lines = [f"- started: {d['started'] or 'unknown'}"]
    if d["duration_seconds"] is not None:
        lines.append(f"- lasted: {d['duration_seconds']} seconds")
    if d["wake_reason"]:
        lines.append(f"- the conversation began because of: {d['wake_reason']}")
    lines.append(
        f"- it ended because of: {d['end_reason']}"
        if d["end_reason"]
        else "- it did not end cleanly (the assistant stopped or crashed mid-conversation)"
    )
    if d["participants"]:
        people = ", ".join(
            f"{p['name'] or p['person_id']} ({'enrolled' if p['is_known'] else 'not recognized'})"
            for p in d["participants"]
        )
        lines.append(f"- people seen on camera: {people}")
    else:
        lines.append("- nobody was seen on camera during the conversation")
    lines.append(f"- the visitor spoke {d['user_turns']} time(s)")
    if d["tool_queries"]:
        lines.append(
            "- the assistant looked these up in its knowledge base: "
            + "; ".join(d["tool_queries"])
        )
    return "\n".join(lines)


def render_transcript(d: dict) -> str:
    """The digest's turns as a plain 'who: text' transcript block."""
    labels = {"user": "Visitor", "agent": "Assistant"}
    return "\n".join(f"{labels.get(t['who'], t['who'])}: {t['text']}" for t in d["turns"])
