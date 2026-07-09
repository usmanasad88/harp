"""The deterministic digest (harp/memory/parse): participants must dedupe,
blank turns drop, tool queries surface, and — the crash contract — a
transcript with no end record must still digest into usable facts."""

from __future__ import annotations

from harp.memory import parse


def _rec(kind: str, **kw) -> dict:
    return {"t": kw.pop("t", 0.0), "ts": kw.pop("ts", "2026-07-09T10:00:00"), "kind": kind, **kw}


def test_digest_of_a_realistic_interaction():
    records = [
        _rec("start", t=100.0, reason="wake word", context="(someone said salam)"),
        _rec("person", person_id="usman", name="Usman", is_known=True),
        _rec("person", person_id="unknown", name=None, is_known=False),
        _rec("turn", who="user", text="salam, where is hall B?"),
        _rec("tool", name="search_knowledge", arguments={"query": "hall B location"}),
        _rec("turn", who="agent", text="Hall B is to your left."),
        _rec("person", person_id="usman", name="Usman", is_known=True),  # re-sighted
        _rec("turn", who="user", text="   "),  # blank: never a memory
        _rec("end", t=190.5, reason="left frame"),
    ]
    d = parse.digest(records)
    assert d["duration_seconds"] == 90.5
    assert d["wake_reason"] == "wake word"
    assert d["end_reason"] == "left frame"
    assert [p["person_id"] for p in d["participants"]] == ["usman", "unknown"]
    assert d["user_turns"] == 1 and len(d["turns"]) == 2
    assert d["tool_queries"] == ["hall B location"]

    facts = parse.render_facts(d)
    assert "Usman (enrolled)" in facts and "hall B location" in facts
    assert parse.render_transcript(d).splitlines() == [
        "Visitor: salam, where is hall B?",
        "Assistant: Hall B is to your left.",
    ]


def test_digest_of_a_crashed_recording():
    """No end record (the app died mid-conversation) must still produce a
    summarizable digest that says so."""
    records = [
        _rec("start", t=100.0, reason="wave"),
        _rec("turn", who="user", text="hello?"),
    ]
    d = parse.digest(records)
    assert d["end_reason"] == "" and d["duration_seconds"] is None
    assert d["user_turns"] == 1
    assert "did not end cleanly" in parse.render_facts(d)
