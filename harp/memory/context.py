"""Pre-computed wake briefing: sighting → (frame + memories) → context.

The problem this solves is latency. The live session gets its context at
open, but a helper-model call takes ~1 s — blocking the wake on it would
delay HARP's first word. So the briefing is computed BEFORE the wake:
face-ID's slow loop announces people the moment they appear in frame
(usually seconds before they wave or speak), and this writer reacts by
asking the parallel Flash Lite agent to fuse the current camera frame with
what the store remembers about the recognized faces into a short "who you
are about to talk to" paragraph, cached and ready. At session open,
`context()` hands it over instantly; app.py falls back to the static
face-ID identity line when there's nothing fresh, so behavior never
degrades below what existed before this module.

Freshness rules (the user-set contract): regenerate when WHO is in frame
changes (identities or head-count) and every `memory.context_ttl_seconds`
(default 120 s) while someone stays in frame. The refresh loop only runs
while HARP is on standby with someone present; an empty frame clears the
cache. Calls go through the shared rate limiter non-blocking — when the
budget is spent, the briefing just stays a little staler (served up to
2×TTL) or falls back to the static line.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Callable, Iterable

from ..config import (
    FALLBACK_CONTEXT_WRITER_PROMPT,
    format_prompt,
    load_context_writer_prompt,
)
from ..core.bus import Bus
from ..core.events import ContextPrepared, PersonIdentified, PresenceChanged, StateChanged
from .agent import GeminiAgent
from .store import MemoryStore

logger = logging.getLogger(__name__)

# How many of a person's stored summaries the briefing prompt sees (newest
# last, matching store order) — enough history to be personal, small enough
# to keep the prompt cheap.
_SUMMARIES_PER_PERSON = 5

# A failed generate (rate window full, model error) retries this often while
# the person is still standing there.
_RETRY_SECONDS = 10.0

# Who-is-here fingerprint: the identity bucket set and the head-count (the
# count catches a second unknown face joining, which doesn't change the set).
_Key = tuple[frozenset[str], int]


class ContextWriter:
    def __init__(
        self,
        bus: Bus,
        agent: GeminiAgent,
        store: MemoryStore,
        people_now: Callable[[], Iterable[PersonIdentified]],
        frame_jpeg: Callable[[], bytes | None] | None = None,
        ttl_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus = bus
        self._agent = agent
        self._store = store
        self._people_now = people_now
        self._frame_jpeg = frame_jpeg
        self._ttl = ttl_seconds
        self._clock = clock

        self._standby = True  # until the first StateChanged says otherwise
        self._count = 0  # last announced head-count (PresenceChanged)
        self._cache: tuple[_Key, str, float] | None = None  # (key, text, at)
        self._task: asyncio.Task | None = None
        self._wakeup = asyncio.Event()

    def context(self) -> str:
        """The briefing for the live session at open: the cached text if it
        still describes who is in frame and isn't too stale, else '' (caller
        falls back to the static identity line)."""
        if self._cache is None:
            return ""
        key, text, at = self._cache
        if key != self._current_key() or self._clock() - at > self._ttl * 2:
            return ""
        return f"(Briefing from your vision and memory helper: {text})"

    async def run(self) -> None:
        """Track standby/presence/identity on the bus; keep the cache warm."""
        stream = self._bus.subscribe(PersonIdentified, PresenceChanged, StateChanged)
        try:
            async for event in stream:
                if isinstance(event, StateChanged):
                    self._standby = event.new == "standby"
                    if self._standby:
                        self._poke()  # someone may still be in frame
                elif isinstance(event, PresenceChanged):
                    if not event.present:
                        self._cache = None  # an empty frame invalidates it
                        self._count = 0
                    else:
                        self._count = event.count
                        self._poke()
                elif isinstance(event, PersonIdentified):
                    self._poke()
        finally:
            if self._task is not None:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

    def _current_key(self) -> _Key | None:
        people = list(self._people_now())
        if not people:
            return None
        return (frozenset(p.person_id for p in people), self._count)

    def _poke(self) -> None:
        """Something changed — make sure the refresh loop is running and
        looking at the world again."""
        if not self._standby:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._refresh_loop())
        else:
            self._wakeup.set()

    async def _refresh_loop(self) -> None:
        """While standby with someone in frame: (re)generate whenever the
        who-is-here key changed or the TTL lapsed, then sleep until the next
        deadline or the next poke. Exits when the frame empties or a session
        opens; the next poke starts a fresh loop."""
        while self._standby:
            # Clear BEFORE looking at the world: a poke that lands while we're
            # generating below re-wakes the wait immediately instead of being
            # lost until the TTL lapses.
            self._wakeup.clear()
            key = self._current_key()
            if key is None:
                return
            if (
                self._cache is None
                or self._cache[0] != key
                or self._clock() - self._cache[2] >= self._ttl
            ):
                await self._generate(key)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=self._next_deadline())

    def _next_deadline(self) -> float:
        if self._cache is None:
            return _RETRY_SECONDS  # last generate failed; try again soon
        # Floor only against busy-spin — it must stay well under any sane TTL.
        return max(0.05, self._ttl - (self._clock() - self._cache[2]))

    async def _generate(self, key: _Key) -> None:
        people = list(self._people_now())
        prompt = format_prompt(
            load_context_writer_prompt(),
            FALLBACK_CONTEXT_WRITER_PROMPT,
            people=self._people_block(people),
        )
        frame = self._frame_jpeg() if self._frame_jpeg is not None else None
        # Non-blocking on the shared budget: a briefing is a nice-to-have,
        # the summarizer's memory call is not — never starve it.
        text = await self._agent.generate(prompt, image_jpeg=frame, wait=False, timeout=20.0)
        if text is None:
            return
        self._cache = (key, text, self._clock())
        await self._bus.publish(ContextPrepared(people=sorted(key[0]), text=text))
        logger.info("memory: wake briefing ready for %s", ", ".join(sorted(key[0])))

    def _people_block(self, people: Iterable[PersonIdentified]) -> str:
        """What the store knows about everyone in frame, rendered for the
        prompt's {people} slot."""
        lines: list[str] = []
        for person in people:
            if not person.is_known:
                lines.append(
                    "- One or more visitors face recognition does not recognize "
                    "(no stored history)."
                )
                continue
            try:
                record = self._store.get(person.person_id)
            except KeyError:
                lines.append(f"- {person.name or person.person_id} (no stored record).")
                continue
            name = record.name or person.person_id
            headline = f"- {name}" + (f" ({record.role})" if record.role else "")
            if record.notes:
                headline += f". Notes: {record.notes}"
            lines.append(headline)
            for entry in record.summaries[-_SUMMARIES_PER_PERSON:]:
                line = f"    [{entry.get('ts', '?')}] {entry.get('text', '')}"
                if entry.get("follow_up"):
                    line += f" (open follow-up: {entry['follow_up']})"
                lines.append(line)
        return "\n".join(lines) if lines else "- Nobody is currently recognized in frame."
