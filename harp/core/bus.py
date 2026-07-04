"""Async publish/subscribe event bus — the decoupling spine of HARP.

Every subsystem depends ONLY on this bus and on `events.py`, never on another
subsystem. Presence publishes `PresenceChanged`; the orchestrator reacts. RAG
publishes `ToolCompleted`; the voice layer reacts. Because nobody holds a direct
reference to anybody else, each subsystem can be built and tested in isolation:
push fake events in, assert the events it publishes out.

To build:
  - `publish(event)`  fan the event out to every current subscriber; must not
    block on a slow subscriber (one stuck consumer shouldn't stall the world).
  - `subscribe(*types)`  hand back an async stream of events, optionally filtered
    to the given event types.
  - Decide the back-pressure policy for a full subscriber queue (drop-oldest is
    usually right for a real-time agent — a late presence frame is worthless).
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from .events import Event

# Cap on each subscriber's backlog. Once full, publish() drops the oldest
# queued event rather than blocking — see the module docstring.
_QUEUE_MAXSIZE = 64


class Bus:
    """In-process async pub/sub. One instance is shared by all subsystems."""

    def __init__(self) -> None:
        # Queue -> the event types it wants, or None for "everything".
        self._subscribers: dict[asyncio.Queue[Event], tuple[type[Event], ...] | None] = {}

    async def publish(self, event: Event) -> None:
        """Deliver `event` to every subscriber. Never block on a slow one."""
        for queue, types in list(self._subscribers.items()):
            if types is not None and not isinstance(event, types):
                continue
            if queue.full():
                queue.get_nowait()  # drop the oldest; a stale event is worthless
            queue.put_nowait(event)

    def subscribe(self, *types: type[Event]) -> AsyncGenerator[Event, None]:
        """Async-iterate events; if `types` is given, yield only those types."""
        # Registered here, eagerly, so events published before the caller's
        # first `await` on the stream are still queued rather than missed.
        # Cleanup lives in _stream's `finally`, which only runs once the
        # stream has actually started — read at least one event before
        # closing a subscription early, or it won't be cleaned up until GC.
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers[queue] = types or None
        return self._stream(queue)

    async def _stream(self, queue: asyncio.Queue[Event]) -> AsyncGenerator[Event, None]:
        try:
            while True:
                yield await queue.get()
        finally:
            del self._subscribers[queue]
