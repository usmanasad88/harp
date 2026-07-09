"""Append-only log of an interaction, turn by turn — the raw record memory
is built from.

One JSONL file per live conversation in `.harp/memory/interactions/`
(record vocabulary owned by memory/parse): a start record (wake reason +
context), a person record for everyone face-ID sees (seeded at open from
`people_now` — the bus won't replay a sighting that happened before the
wake — then updated live), each user/agent turn, each tool request, and an
end record. Turns arrive as STREAMED pieces: `final=False` deltas carrying
the words, then a `final=True` marker that closes the turn — carrying empty
text on the OpenAI path (verified live 2026-07-09), possibly the last piece
on Gemini's. So the logger accumulates deltas per speaker and writes one
turn record at each final; a turn still open at shutdown is flushed, not lost.

Not to be confused with the per-run developer log (core/session_log.py),
which records the whole bus verbatim: THIS is the per-conversation record the
memory summarizer consumes, and it lives as long as the memories do.

Crash-tolerant like the session log: every line is flushed as written, and
the file keeps a `.part` suffix while the conversation is live — a clean end
renames it to `.jsonl` (pending: the summarizer's cue), while a crash leaves
`.part` behind for `rescue_stale_transcripts()` to promote at next boot, so
even a conversation the app died in gets remembered. A write error drops that
line, never the agent (app.py treats a finished task as a crash).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, TextIO

from ..core.bus import Bus
from ..core.events import (
    AgentSaid,
    InteractionEnded,
    InteractionStarted,
    PersonIdentified,
    ToolRequested,
    UserSaid,
)
from . import parse

logger = logging.getLogger(__name__)

# Lifecycle by suffix: being written → awaiting the summarizer → summarized /
# judged not worth a memory. The summarizer owns the last transition.
ACTIVE_SUFFIX = ".part"
PENDING_SUFFIX = ".jsonl"
DONE_SUFFIX = ".jsonl.done"
SKIPPED_SUFFIX = ".jsonl.skipped"


def rescue_stale_transcripts(dir: Path) -> list[Path]:
    """Promote any `.part` file a crashed run left behind to a pending
    transcript so the summarizer picks it up. Call at boot, BEFORE the logger
    starts — a live run's own `.part` must never be promoted mid-write."""
    rescued: list[Path] = []
    for stale in sorted(Path(dir).glob(f"*{ACTIVE_SUFFIX}")):
        target = stale.with_name(stale.name[: -len(ACTIVE_SUFFIX)] + PENDING_SUFFIX)
        try:
            stale.rename(target)
            rescued.append(target)
        except OSError:
            logger.warning("interaction log: could not rescue %s", stale, exc_info=True)
    return rescued


class InteractionLogger:
    """Records one live conversation per file (see module docstring).

    `people_now` (face-ID's `people_now` in the real app) seeds the
    participant list at session open; PersonIdentified events add anyone who
    joins later. Publishes nothing — the summarizer watches InteractionEnded
    on the bus itself."""

    def __init__(
        self,
        bus: Bus,
        dir: Path,
        people_now: Callable[[], Iterable[PersonIdentified]] | None = None,
    ) -> None:
        self._bus = bus
        self._dir = Path(dir)
        self._people_now = people_now
        self._file: TextIO | None = None
        self._path: Path | None = None
        # In-flight turn text per speaker (streamed deltas, see docstring).
        self._partial: dict[str, list[str]] = {"user": [], "agent": []}

    async def run(self) -> None:
        """Subscribe to conversation events and persist them per interaction."""
        self._dir.mkdir(parents=True, exist_ok=True)
        stream = self._bus.subscribe(
            InteractionStarted,
            InteractionEnded,
            UserSaid,
            AgentSaid,
            ToolRequested,
            PersonIdentified,
        )
        try:
            async for event in stream:
                self._handle(event)
        finally:
            # App shutdown mid-conversation: finalize what we have — the
            # digest treats a missing end record as "did not end cleanly".
            self._close()

    def _handle(self, event) -> None:
        try:
            if isinstance(event, InteractionStarted):
                self._open(event)
            elif self._file is None:
                return  # idle chatter (e.g. face-ID sightings between sessions)
            elif isinstance(event, InteractionEnded):
                self._flush_turns()  # a turn cut off by the end still counts
                self._write({"kind": parse.KIND_END, "reason": event.reason})
                self._close()
            elif isinstance(event, (UserSaid, AgentSaid)):
                who = "user" if isinstance(event, UserSaid) else "agent"
                pieces = self._partial[who]
                if event.text:
                    pieces.append(event.text)
                if event.final:
                    text = "".join(pieces).strip()
                    pieces.clear()
                    if text:
                        self._write({"kind": parse.KIND_TURN, "who": who, "text": text})
            elif isinstance(event, ToolRequested):
                self._write(
                    {"kind": parse.KIND_TOOL, "name": event.name, "arguments": event.arguments}
                )
            elif isinstance(event, PersonIdentified):
                self._write_person(event)
        except Exception:
            logger.warning("interaction log: dropped a record", exc_info=True)

    def _open(self, event: InteractionStarted) -> None:
        if self._file is not None:
            # A start while one is open shouldn't happen; keep the old record.
            self._close()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self._dir / f"interaction-{stamp}{ACTIVE_SUFFIX}"
        n = 1
        while path.exists():  # two interactions within the same second
            n += 1
            path = self._dir / f"interaction-{stamp}-{n}{ACTIVE_SUFFIX}"
        self._file = path.open("w", encoding="utf-8")
        self._path = path
        self._partial = {"user": [], "agent": []}  # no stale pre-wake pieces
        self._write({"kind": parse.KIND_START, "reason": event.reason, "context": event.context})
        if self._people_now is not None:
            for person in self._people_now():
                self._write_person(person)

    def _write_person(self, person: PersonIdentified) -> None:
        self._write(
            {
                "kind": parse.KIND_PERSON,
                "person_id": person.person_id,
                "name": person.name,
                "is_known": person.is_known,
            }
        )

    def _write(self, record: dict) -> None:
        if self._file is None:
            return
        now = time.time()
        line = {
            "t": round(now, 3),
            "ts": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            **record,
        }
        try:
            json.dump(line, self._file, ensure_ascii=False, default=repr)
            self._file.write("\n")
            self._file.flush()
        except Exception:
            logger.warning("interaction log: write failed", exc_info=True)

    def _flush_turns(self) -> None:
        """Write any in-flight (final-less) turn text as turn records — a
        conversation cut off mid-sentence (shutdown, crash-side teardown)
        keeps what was said."""
        for who, pieces in self._partial.items():
            text = "".join(pieces).strip()
            pieces.clear()
            if text:
                self._write({"kind": parse.KIND_TURN, "who": who, "text": text})

    def _close(self) -> None:
        if self._file is None:
            return
        self._flush_turns()
        file, path = self._file, self._path
        self._file = self._path = None
        try:
            file.close()
        except Exception:
            logger.warning("interaction log: close failed", exc_info=True)
        assert path is not None
        target = path.with_name(path.name[: -len(ACTIVE_SUFFIX)] + PENDING_SUFFIX)
        try:
            path.rename(target)
        except OSError:
            logger.warning("interaction log: could not finalize %s", path, exc_info=True)
