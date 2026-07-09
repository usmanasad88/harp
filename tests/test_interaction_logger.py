"""The per-interaction transcript (harp/memory/logger): turns are STREAMED —
final=False deltas carry the words and the final=True marker is EMPTY on the
OpenAI path (observed live 2026-07-09; the first live test lost every turn to
the assumption that finals carry text) — so delta accumulation is the
regression here. Plus: participants seeded at open (the bus never replays a
sighting from before the wake), a shutdown mid-conversation flushing the
in-flight turn and finalizing the file, and the crash-rescue path promoting a
stale .part so the conversation isn't lost."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from harp.core.bus import Bus
from harp.core.events import (
    AgentSaid,
    InteractionEnded,
    InteractionStarted,
    PersonIdentified,
    ToolRequested,
    UserSaid,
)
from harp.memory.logger import (
    ACTIVE_SUFFIX,
    PENDING_SUFFIX,
    InteractionLogger,
    rescue_stale_transcripts,
)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_one_interaction_becomes_one_finalized_file(tmp_path: Path):
    bus = Bus()
    seeded = [PersonIdentified(person_id="usman", name="Usman", is_known=True)]
    log = InteractionLogger(bus, tmp_path, people_now=lambda: seeded)
    task = asyncio.create_task(log.run())
    await asyncio.sleep(0.01)

    await bus.publish(UserSaid(text="stray line between sessions", final=True))  # no session: ignored
    await bus.publish(InteractionStarted(reason="wave", context="(someone waved)"))
    # The OpenAI shape observed live: word deltas, then an EMPTY final marker.
    await bus.publish(UserSaid(text="sala", final=False))
    await bus.publish(UserSaid(text="m", final=False))
    await bus.publish(UserSaid(text="", final=True))
    await bus.publish(ToolRequested(id="t1", name="search_knowledge", arguments={"query": "halls"}))
    # The whole-turn-on-the-final shape (Gemini's finished marker carries text).
    await bus.publish(AgentSaid(text="walaikum salam", final=True))
    await bus.publish(PersonIdentified(person_id="unknown", is_known=False))
    await bus.publish(InteractionEnded(reason="left frame"))
    await asyncio.sleep(0.02)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    files = list(tmp_path.glob(f"*{PENDING_SUFFIX}"))
    assert len(files) == 1 and not list(tmp_path.glob(f"*{ACTIVE_SUFFIX}"))
    records = _records(files[0])
    assert [r["kind"] for r in records] == [
        "start", "person", "turn", "tool", "turn", "person", "end",
    ]
    assert records[1]["person_id"] == "usman"  # seeded at open, not via the bus
    assert [r["text"] for r in records if r["kind"] == "turn"] == ["salam", "walaikum salam"]
    assert records[0]["reason"] == "wave" and records[-1]["reason"] == "left frame"


async def test_shutdown_mid_conversation_still_finalizes(tmp_path: Path):
    bus = Bus()
    log = InteractionLogger(bus, tmp_path)
    task = asyncio.create_task(log.run())
    await asyncio.sleep(0.01)
    await bus.publish(InteractionStarted(reason="button"))
    await bus.publish(UserSaid(text="hel", final=False))
    await bus.publish(UserSaid(text="lo", final=False))  # no final ever comes
    await asyncio.sleep(0.02)
    task.cancel()  # Ctrl+C mid-conversation — no InteractionEnded either
    with contextlib.suppress(asyncio.CancelledError):
        await task

    files = list(tmp_path.glob(f"*{PENDING_SUFFIX}"))
    assert len(files) == 1
    # The in-flight turn was flushed; no end record — exactly what
    # parse.digest flags as "did not end cleanly".
    records = _records(files[0])
    assert [r["kind"] for r in records] == ["start", "turn"]
    assert records[1]["text"] == "hello"


def test_rescue_promotes_a_crashed_part_file(tmp_path: Path):
    stale = tmp_path / f"interaction-x{ACTIVE_SUFFIX}"
    stale.write_text('{"kind": "start"}\n', encoding="utf-8")
    rescued = rescue_stale_transcripts(tmp_path)
    assert rescued == [tmp_path / f"interaction-x{PENDING_SUFFIX}"]
    assert rescued[0].exists() and not stale.exists()
