"""The pre-computed wake briefing (harp/memory/context). The user-set
contract: computed on sighting (ready BEFORE the wake), regenerated when who-
is-in-frame changes and on TTL lapse while someone lingers, never refreshed
during a live session (quota), cleared when the frame empties, and a failed
helper call serves '' so app.py falls back to the static identity line."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from harp.core.bus import Bus
from harp.core.events import PersonIdentified, PresenceChanged, StateChanged
from harp.memory.context import ContextWriter
from harp.memory.store import MemoryStore


class FakeAgent:
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.prompts: list[str] = []
        self.images: list[bytes | None] = []

    async def generate(self, prompt: str, *, image_jpeg=None, **kwargs) -> str | None:
        self.calls += 1
        self.prompts.append(prompt)
        self.images.append(image_jpeg)
        return None if self.fail else f"briefing {self.calls}"


def _setup(tmp_path: Path, ttl: float = 10.0):
    bus = Bus()
    agent = FakeAgent()
    store = MemoryStore(tmp_path / "people")
    people: list[PersonIdentified] = []
    writer = ContextWriter(
        bus,
        agent,
        store,
        people_now=lambda: list(people),
        frame_jpeg=lambda: b"fake-jpeg",
        ttl_seconds=ttl,
    )
    return bus, agent, store, people, writer


async def _run(writer: ContextWriter):
    task = asyncio.create_task(writer.run())
    await asyncio.sleep(0.01)  # let the subscription register
    return task


async def _stop(task: asyncio.Task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_briefing_is_ready_before_the_wake(tmp_path: Path):
    bus, agent, store, people, writer = _setup(tmp_path)
    store.upsert_person({"person_id": "usman", "name": "Usman", "role": "organizer"})
    store.add_summary("usman", "Asked about hall B.", follow_up="did they find hall B")
    task = await _run(writer)
    assert writer.context() == ""  # nobody sighted yet

    usman = PersonIdentified(person_id="usman", name="Usman", is_known=True)
    people.append(usman)
    await bus.publish(PresenceChanged(present=True, count=1))
    await bus.publish(usman)
    await asyncio.sleep(0.05)

    ctx = writer.context()
    assert "briefing 1" in ctx and ctx.startswith("(")
    # The store's memory actually reached the model, frame included.
    assert "Usman" in agent.prompts[0] and "organizer" in agent.prompts[0]
    assert "hall B" in agent.prompts[0] and "open follow-up" in agent.prompts[0]
    assert agent.images[0] == b"fake-jpeg"
    await _stop(task)


async def test_people_change_regenerates_and_stale_key_serves_nothing(tmp_path: Path):
    bus, agent, store, people, writer = _setup(tmp_path)
    usman = PersonIdentified(person_id="usman", name="Usman", is_known=True)
    people.append(usman)
    task = await _run(writer)
    await bus.publish(PresenceChanged(present=True, count=1))
    await bus.publish(usman)
    await asyncio.sleep(0.05)
    assert "briefing 1" in writer.context()

    # Someone joins: before the regenerate lands, the old briefing no longer
    # describes who's there — it must NOT be served.
    stranger = PersonIdentified(person_id="unknown", is_known=False)
    people.append(stranger)
    assert writer.context() == ""

    await bus.publish(PresenceChanged(present=True, count=2))
    await bus.publish(stranger)
    await asyncio.sleep(0.05)
    assert agent.calls == 2
    assert "briefing 2" in writer.context()
    await _stop(task)


async def test_ttl_lapse_refreshes_while_they_linger(tmp_path: Path):
    bus, agent, store, people, writer = _setup(tmp_path, ttl=0.1)
    usman = PersonIdentified(person_id="usman", name="Usman", is_known=True)
    people.append(usman)
    task = await _run(writer)
    await bus.publish(PresenceChanged(present=True, count=1))
    await bus.publish(usman)
    await asyncio.sleep(0.35)
    # No new events at all — the loop alone kept the briefing warm.
    assert agent.calls >= 2
    assert writer.context() != ""
    await _stop(task)


async def test_no_refresh_during_a_live_session_and_absence_clears(tmp_path: Path):
    bus, agent, store, people, writer = _setup(tmp_path, ttl=0.1)
    usman = PersonIdentified(person_id="usman", name="Usman", is_known=True)
    people.append(usman)
    task = await _run(writer)
    await bus.publish(PresenceChanged(present=True, count=1))
    await bus.publish(usman)
    await asyncio.sleep(0.05)
    assert agent.calls == 1

    # Session opens: sightings + TTL lapses must not burn quota mid-session.
    await bus.publish(StateChanged(old="standby", new="active"))
    await asyncio.sleep(0.02)
    await bus.publish(usman)
    await asyncio.sleep(0.25)  # > 2 TTLs
    assert agent.calls == 1

    # They walk off during the session: the stale briefing dies with them.
    people.clear()
    await bus.publish(PresenceChanged(present=False, count=0))
    await asyncio.sleep(0.02)
    await bus.publish(StateChanged(old="active", new="standby"))
    await asyncio.sleep(0.05)
    assert writer.context() == ""
    assert agent.calls == 1  # empty frame: nothing to brief about either
    await _stop(task)


async def test_failed_generate_serves_nothing_so_the_static_line_wins(tmp_path: Path):
    bus, agent, store, people, writer = _setup(tmp_path)
    agent.fail = True  # rate window spent, or the API is down
    usman = PersonIdentified(person_id="usman", name="Usman", is_known=True)
    people.append(usman)
    task = await _run(writer)
    await bus.publish(PresenceChanged(present=True, count=1))
    await bus.publish(usman)
    await asyncio.sleep(0.05)
    assert writer.context() == ""
    await _stop(task)
