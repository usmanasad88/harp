"""Per-run developer log: one JSONL timeline a later debugging session can replay.

The consumer is whoever has to work out, after the fact, what a run of the
full agent actually did — a person, or an agent session reading it cold. Three
sources merge into ONE time-ordered file per run (harp.yaml `session_log:`,
default `.harp/logs/session-<timestamp>.jsonl`):

  - a `session_start` header — the effective settings, provider/model,
    platform — so the file states what this run's knobs ACTUALLY were, not
    what harp.yaml happens to say today;
  - every bus event, verbatim (a no-filter subscription — event types added
    later are captured with no change here, the same trick the dashboard's
    raw view uses);
  - every Python `logging` record the modules already emit (`handler()`,
    attached to the root logger by app.py) — the camera warnings, mic
    retries, and "filter heard (asr)" lines that carry most of the actual
    debugging story, currently lost when the terminal closes.

One JSON object per line, flushed as written, so a crash preserves the record
right up to the moment it happened — which is exactly when it's needed.

Fail-safe by design (the status_voice philosophy): a value json can't encode
falls back to `repr`, a write error drops that line, records after `close()`
are ignored — logging can never take the agent down (app.py treats any task
exiting as a crash). The bus's drop-oldest policy applies here like any
subscriber: if the logger ever fell behind, the oldest events would be lost
rather than the agent stalled.

This is the DEVELOPER log. The per-person interaction transcript that the
memory summarizer will read (memory/logger.py) is a separate, still-unbuilt
thing with a different consumer and lifetime.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import IO

from .bus import Bus
from .events import Event


class SessionLog:
    """One append-only JSONL file for this run. `open()` first (prunes old
    runs, writes the header), then `run()` persists bus events and `handler()`
    bridges stdlib logging records into the same timeline. Thread-safe: bus
    events arrive on the event loop, log records from any thread (camera,
    pynput, audio callbacks)."""

    def __init__(self, bus: Bus, directory: Path, keep_runs: int = 30) -> None:
        self._bus = bus
        self._dir = directory
        self._keep_runs = keep_runs
        self._file: IO[str] | None = None
        self._path: Path | None = None
        self._lock = threading.Lock()

    @property
    def path(self) -> Path | None:
        """This run's file (None before `open()`)."""
        return self._path

    def open(self, header: dict | None = None) -> Path:
        """Prune old runs, create this run's file, write the `session_start`
        header. Call before attaching `handler()` — records before open are
        dropped, not queued."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prune()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self._dir / f"session-{stamp}.jsonl"
        n = 1
        while path.exists():  # a second run within the same second
            n += 1
            path = self._dir / f"session-{stamp}-{n}.jsonl"
        self._path = path
        self._file = path.open("a", encoding="utf-8")
        self._write({"kind": "session_start", **(header or {})})
        return path

    async def run(self) -> None:
        """Persist every bus event, whatever its type, until cancelled."""
        async for event in self._bus.subscribe():
            self.log_event(event)

    def log_event(self, event: Event) -> None:
        """Write one bus event; an unserializable payload degrades to repr."""
        try:
            data = dataclasses.asdict(event)
        except (TypeError, ValueError):  # asdict deep-copies; exotic fields fail
            data = {"repr": repr(event)}
        self._write({"kind": "event", "type": type(event).__name__, "data": data})

    def handler(self) -> logging.Handler:
        """A stdlib handler that writes records into this timeline — attach to
        the root logger so every module's existing log lines land here."""
        return _RecordHandler(self)

    def close(self, reason: str = "shutdown") -> None:
        """Write the final record and stop accepting lines. Anything logged
        after this (late threads during teardown) is silently dropped."""
        self._write({"kind": "session_end", "reason": reason})
        with self._lock:
            file, self._file = self._file, None
        if file is not None:
            try:
                file.close()
            except OSError:
                pass

    def _write(self, record: dict) -> None:
        entry = {
            "t": round(time.time(), 3),
            "iso": datetime.now().isoformat(timespec="milliseconds"),
        }
        entry.update(record)
        try:
            line = json.dumps(entry, ensure_ascii=False, default=repr)
        except (TypeError, ValueError):  # circular reference / hostile repr
            return
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(line + "\n")
                self._file.flush()  # per line: a crash keeps everything before it
            except (OSError, ValueError):
                return

    def _prune(self) -> None:
        """Delete the oldest runs so this run makes at most `keep_runs` files.
        Timestamp-named files sort lexicographically = chronologically."""
        try:
            runs = sorted(self._dir.glob("session-*.jsonl"))
        except OSError:
            return
        for old in runs[: max(0, len(runs) - (self._keep_runs - 1))]:
            try:
                old.unlink()
            except OSError:
                pass


class _RecordHandler(logging.Handler):
    """Bridges stdlib logging into the timeline. `_write` itself never logs,
    so a record can't recurse back into this handler."""

    def __init__(self, log: SessionLog) -> None:
        super().__init__()
        self._log = log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "kind": "log",
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info and record.exc_info[1] is not None:
                entry["exc"] = "".join(
                    traceback.format_exception(*record.exc_info)
                ).rstrip()
            self._log._write(entry)
        except Exception:
            self.handleError(record)
