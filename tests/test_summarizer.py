"""End-of-interaction memory (harp/memory/summarizer). The contracts that
keep memories from silently vanishing: every KNOWN participant gets the
summary, unknown-only interactions land in the guestbook (with the transcript
filename for later re-attachment), a wake where nobody spoke never spends a
rate-limited call, a model failure leaves the transcript pending for a later
sweep, and the boot sweep rescues a conversation the app crashed in."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from harp.core.bus import Bus
from harp.core.events import InteractionEnded, MemoryWritten
from harp.memory.logger import ACTIVE_SUFFIX
from harp.memory.store import MemoryStore
from harp.memory.summarizer import MemorySummarizer


class FakeAgent:
    def __init__(self, reply: str | None):
        self.reply = reply
        self.calls = 0

    async def generate(self, prompt: str, **kwargs) -> str | None:
        self.calls += 1
        return self.reply


def _transcript_lines(participants, turns) -> str:
    records = [{"t": 100.0, "ts": "2026-07-09T10:00:00", "kind": "start", "reason": "wave"}]
    for pid, name, known in participants:
        records.append({"kind": "person", "person_id": pid, "name": name, "is_known": known})
    for who, text in turns:
        records.append({"kind": "turn", "who": who, "text": text})
    records.append({"t": 160.0, "kind": "end", "reason": "left frame"})
    return "\n".join(json.dumps(r) for r in records) + "\n"


def _setup(tmp_path: Path, agent: FakeAgent):
    bus = Bus()
    store = MemoryStore(tmp_path / "people")
    interactions = tmp_path / "interactions"
    interactions.mkdir()
    guestbook = tmp_path / "guestbook.jsonl"
    summarizer = MemorySummarizer(
        bus, store, agent, interactions, guestbook, settle_seconds=0
    )
    return bus, store, interactions, guestbook, summarizer


async def test_every_known_participant_gets_the_memory(tmp_path: Path):
    reply = json.dumps(
        {"summary": "Asked about hall B.", "follow_up": "did they find hall B", "person_facts": ""}
    )
    agent = FakeAgent(reply)
    bus, store, interactions, guestbook, summarizer = _setup(tmp_path, agent)
    store.upsert_person({"person_id": "usman", "name": "Usman"})
    store.upsert_person({"person_id": "sara", "name": "Sara"})
    (interactions / "interaction-1.jsonl").write_text(
        _transcript_lines(
            [("usman", "Usman", True), ("sara", "Sara", True), ("unknown", None, False)],
            [("user", "where is hall B"), ("agent", "to your left")],
        ),
        encoding="utf-8",
    )
    events = bus.subscribe(MemoryWritten)

    await summarizer.sweep()

    for pid in ("usman", "sara"):
        entry = store.get(pid).summaries[-1]
        assert entry["text"] == "Asked about hall B."
        assert entry["follow_up"] == "did they find hall B"
    assert not guestbook.exists()  # known people got it; no guestbook duplicate
    assert list(interactions.glob("*.jsonl.done")) and not list(interactions.glob("*.jsonl"))
    event = await asyncio.wait_for(anext(events), 1)
    assert sorted(event.person_ids) == ["sara", "usman"]


async def test_unknown_visitors_land_in_the_guestbook(tmp_path: Path):
    # Reply is NOT valid JSON — the whole text must become the summary
    # (a rough memory beats a lost one).
    agent = FakeAgent("They asked about the robot dolphins and left happy.")
    bus, store, interactions, guestbook, summarizer = _setup(tmp_path, agent)
    (interactions / "interaction-1.jsonl").write_text(
        _transcript_lines([("unknown", None, False)], [("user", "what are you?")]),
        encoding="utf-8",
    )

    await summarizer.sweep()

    entries = [json.loads(l) for l in guestbook.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["summary"] == "They asked about the robot dolphins and left happy."
    assert entries[0]["transcript"] == "interaction-1.jsonl.done"
    assert (interactions / entries[0]["transcript"]).exists()


async def test_wake_with_no_conversation_skips_without_a_model_call(tmp_path: Path):
    agent = FakeAgent('{"summary": "should never be asked"}')
    bus, store, interactions, guestbook, summarizer = _setup(tmp_path, agent)
    (interactions / "interaction-1.jsonl").write_text(
        # The agent greeted, nobody answered: no user turn, no memory.
        _transcript_lines([("unknown", None, False)], [("agent", "hello?")]),
        encoding="utf-8",
    )

    await summarizer.sweep()

    assert agent.calls == 0
    assert list(interactions.glob("*.jsonl.skipped"))
    assert not list(interactions.glob("*.jsonl")) and not guestbook.exists()


async def test_model_failure_leaves_it_pending_for_the_next_sweep(tmp_path: Path):
    agent = FakeAgent(None)  # quota exhausted / network down
    bus, store, interactions, guestbook, summarizer = _setup(tmp_path, agent)
    store.upsert_person({"person_id": "usman", "name": "Usman"})
    (interactions / "interaction-1.jsonl").write_text(
        _transcript_lines([("usman", "Usman", True)], [("user", "salam")]),
        encoding="utf-8",
    )

    await summarizer.sweep()
    assert list(interactions.glob("*.jsonl"))  # still pending, nothing lost

    agent.reply = '{"summary": "Greeted Usman."}'
    await summarizer.sweep()
    assert list(interactions.glob("*.jsonl.done"))
    assert store.get("usman").summaries[-1]["text"] == "Greeted Usman."


async def test_run_rescues_a_crashed_transcript_and_sweeps_on_end(tmp_path: Path):
    agent = FakeAgent('{"summary": "ok"}')
    bus, store, interactions, guestbook, summarizer = _setup(tmp_path, agent)
    # A .part file the previous run died holding open (its last line even got
    # truncated mid-write) — boot must promote and summarize it.
    crashed = _transcript_lines([("unknown", None, False)], [("user", "hello")])
    (interactions / f"interaction-crashed{ACTIVE_SUFFIX}").write_text(
        crashed + '{"kind": "turn", "who": "us', encoding="utf-8"
    )
    task = asyncio.create_task(summarizer.run())
    await asyncio.sleep(0.05)
    assert (interactions / "interaction-crashed.jsonl.done").exists()

    # And a fresh interaction finishing while it runs is swept on the event.
    (interactions / "interaction-2.jsonl").write_text(
        _transcript_lines([("unknown", None, False)], [("user", "bye")]), encoding="utf-8"
    )
    await bus.publish(InteractionEnded(reason="agent ended"))
    await asyncio.sleep(0.05)
    assert (interactions / "interaction-2.jsonl.done").exists()

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
