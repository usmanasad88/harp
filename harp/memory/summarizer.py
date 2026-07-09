"""End-of-interaction → long-term memory, via the parallel Flash Lite agent.

Watches InteractionEnded; for each pending transcript the logger finalized it
runs the pipeline: mechanical digest (memory/parse — everything extractable
without a model), then ONE rate-limited model call (memory/agent) that returns
{summary, follow_up, person_facts} as JSON, then storage:

  - every ENROLLED participant gets the summary on their record
    (memory/store — this is what the next meeting's briefing reads);
  - an interaction with only unrecognized visitors goes to the guestbook
    (a JSONL file, one entry per interaction) — no face is ever stored for
    them (the 2026-07-02 decision), but the conversation is remembered and
    carries its transcript filename so a later enrollment can re-attach it.

Quota- and crash-proof by construction: a transcript is only marked done
(renamed `.jsonl.done`) after a successful model call — a failed/skipped call
leaves it pending and the sweep retries at the next interaction end or next
boot. Wakes where nobody actually spoke are marked `.jsonl.skipped` without
spending a call. The boot sweep also promotes `.part` files a crashed run
left, so even a conversation the app died in gets remembered.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from ..config import (
    FALLBACK_MEMORY_SUMMARIZER_PROMPT,
    format_prompt,
    load_memory_summarizer_prompt,
)
from ..core.bus import Bus
from ..core.events import InteractionEnded, MemoryWritten
from . import parse
from .agent import GeminiAgent, parse_json
from .logger import DONE_SUFFIX, PENDING_SUFFIX, SKIPPED_SUFFIX, rescue_stale_transcripts
from .store import MemoryStore

logger = logging.getLogger(__name__)

# Generous per-call budget: with the shared window full, a slot is guaranteed
# to free within its 60 s period, so waiting this long can't starve.
_CALL_TIMEOUT_S = 90.0


class MemorySummarizer:
    def __init__(
        self,
        bus: Bus,
        store: MemoryStore,
        agent: GeminiAgent,
        dir: Path,
        guestbook: Path,
        settle_seconds: float = 0.5,
    ) -> None:
        self._bus = bus
        self._store = store
        self._agent = agent
        self._dir = Path(dir)
        self._guestbook = Path(guestbook)
        # The logger finalizes the transcript on the same InteractionEnded we
        # subscribe to, and inter-subscriber order isn't guaranteed — give it
        # a beat before sweeping (0 in tests).
        self._settle_seconds = settle_seconds

    async def run(self) -> None:
        """Boot recovery (crashed `.part` files + anything still pending),
        then summarize after every interaction end."""
        self._dir.mkdir(parents=True, exist_ok=True)
        stream = self._bus.subscribe(InteractionEnded)  # registered eagerly
        rescued = rescue_stale_transcripts(self._dir)
        if rescued:
            logger.info("memory: rescued %d transcript(s) from a crashed run", len(rescued))
        await self.sweep()
        async for _ in stream:
            await asyncio.sleep(self._settle_seconds)
            await self.sweep()

    async def sweep(self) -> None:
        """Summarize every pending transcript, oldest first. Stops early when
        the model is unavailable (quota/network) — the rest stay pending for
        the next sweep rather than hammering a dead quota."""
        for path in sorted(self._dir.glob(f"*{PENDING_SUFFIX}")):
            try:
                if not await self._summarize_file(path):
                    break
            except Exception:
                # A corrupt file / disk hiccup must not kill the subsystem;
                # the file stays pending and the next sweep retries it.
                logger.warning("memory: failed to process %s", path.name, exc_info=True)
                break

    async def _summarize_file(self, path: Path) -> bool:
        """One transcript → one memory. False = the model was unavailable
        (leave pending, stop sweeping); True = handled (done or skipped)."""
        d = parse.digest(_read_records(path))
        if d["user_turns"] == 0:
            # A wake where nobody actually said anything — not worth a
            # rate-limited call, and a memory of it would be noise.
            _finish(path, SKIPPED_SUFFIX)
            return True

        prompt = format_prompt(
            load_memory_summarizer_prompt(),
            FALLBACK_MEMORY_SUMMARIZER_PROMPT,
            facts=parse.render_facts(d),
            transcript=parse.render_transcript(d),
        )
        reply = await self._agent.generate(
            prompt, json_response=True, wait=True, timeout=_CALL_TIMEOUT_S
        )
        if reply is None:
            return False

        # A malformed reply degrades to "the whole text is the summary" —
        # a rough memory beats a lost one.
        data = parse_json(reply) or {"summary": reply}
        summary = str(data.get("summary") or "").strip() or reply
        follow_up = str(data.get("follow_up") or "").strip()
        person_facts = str(data.get("person_facts") or "").strip()

        known = [p for p in d["participants"] if p["is_known"]]
        attached: list[str] = []
        for p in known:
            try:
                self._store.add_summary(
                    p["person_id"], summary, follow_up=follow_up, person_facts=person_facts
                )
                attached.append(p["person_id"])
            except KeyError:
                # Enrolled when sighted, deleted from the store since.
                logger.warning("memory: %s no longer in the store", p["person_id"])
        if not attached:
            self._guestbook_append(d, path, summary, follow_up, person_facts)

        await self._bus.publish(
            MemoryWritten(person_ids=attached, summary=summary, follow_up=follow_up)
        )
        _finish(path, DONE_SUFFIX)
        logger.info(
            "memory: summarized %s -> %s",
            path.name, ", ".join(attached) if attached else "guestbook",
        )
        return True

    def _guestbook_append(
        self, d: dict, path: Path, summary: str, follow_up: str, person_facts: str
    ) -> None:
        """One guestbook entry per unknown-visitor interaction. Carries the
        transcript filename (as it will be after the .done rename) so a later
        enrollment can find and re-attach the full conversation."""
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "started": d["started"],
            "summary": summary,
            "follow_up": follow_up,
            "person_facts": person_facts,
            "participants": d["participants"],
            "transcript": path.name[: -len(PENDING_SUFFIX)] + DONE_SUFFIX,
        }
        self._guestbook.parent.mkdir(parents=True, exist_ok=True)
        with self._guestbook.open("a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def _read_records(path: Path) -> list[dict]:
    """Every parseable line; a truncated last line (crash mid-write) is skipped."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def _finish(path: Path, suffix: str) -> None:
    path.rename(path.with_name(path.name[: -len(PENDING_SUFFIX)] + suffix))
