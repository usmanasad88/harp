"""Per-run session log (harp/core/session_log): the JSONL timeline must
survive the things that would silently gut it — unserializable event payloads,
a dying disk, records arriving after shutdown, log lines from other threads
racing bus events — and its retention prune must delete only the oldest runs.
The logger can NEVER raise into the agent: app.py treats a finished task as a
crash."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from pathlib import Path

from harp.core.bus import Bus
from harp.core.events import ErrorRaised, ToolCompleted, UserSaid
from harp.core.session_log import SessionLog


def _records(path: Path) -> list[dict]:
    """Every line must parse on its own — that's the crash-robustness bar."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_timeline_captures_header_events_and_log_records(tmp_path: Path):
    bus = Bus()
    log = SessionLog(bus, tmp_path)
    path = log.open({"provider": "openai", "settings": {"wake_level": 0.15}})
    task = asyncio.create_task(log.run())
    await asyncio.sleep(0.01)  # let run() register its bus subscription

    await bus.publish(UserSaid(text="salam", final=True))
    await bus.publish(ErrorRaised(where="voice.session", message="boom"))
    await asyncio.sleep(0.02)  # let the events drain to the file

    handler = log.handler()
    lg = logging.getLogger("harp.test_session_log")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    try:
        lg.info("filter heard (asr): %s", "hello")
        try:
            raise ValueError("kaboom")
        except ValueError:
            lg.exception("tool failed")
    finally:
        lg.removeHandler(handler)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    log.close()

    records = _records(path)
    assert records[0]["kind"] == "session_start"
    assert records[0]["provider"] == "openai"
    assert records[0]["settings"] == {"wake_level": 0.15}
    assert records[-1]["kind"] == "session_end"
    assert all("t" in r and "iso" in r for r in records)

    events = {r["type"]: r["data"] for r in records if r["kind"] == "event"}
    assert events["UserSaid"] == {"text": "salam", "final": True}
    assert events["ErrorRaised"]["where"] == "voice.session"

    logs = [r for r in records if r["kind"] == "log"]
    assert any(r["msg"] == "filter heard (asr): hello" for r in logs)
    # a logger.exception line keeps its traceback — that's the debugging story
    exc = next(r for r in logs if r["msg"] == "tool failed")
    assert "ValueError: kaboom" in exc["exc"]


async def test_unserializable_event_payload_degrades_to_repr(tmp_path: Path):
    """A tool result holding something json/deepcopy chokes on (a lock, a
    socket, ...) must still land as a parseable line — not kill the logger."""
    log = SessionLog(Bus(), tmp_path)
    path = log.open()
    log.log_event(ToolCompleted(id="t1", output=threading.Lock()))
    log.close()

    event = next(r for r in _records(path) if r["kind"] == "event")
    assert event["type"] == "ToolCompleted"
    assert "lock" in json.dumps(event["data"]).lower()  # survived as repr


def test_prune_keeps_only_the_newest_runs(tmp_path: Path):
    for day in range(1, 6):
        (tmp_path / f"session-2026010{day}-000000.jsonl").write_text("{}\n")
    log = SessionLog(Bus(), tmp_path, keep_runs=3)
    path = log.open()
    log.close()

    runs = sorted(p.name for p in tmp_path.glob("session-*.jsonl"))
    assert len(runs) == 3
    assert path.name in runs
    # the survivors are the newest old runs, not arbitrary ones
    assert "session-20260104-000000.jsonl" in runs
    assert "session-20260105-000000.jsonl" in runs


def test_write_failures_and_late_records_never_raise(tmp_path: Path):
    log = SessionLog(Bus(), tmp_path)
    log.open()

    class DyingDisk:
        def write(self, s):
            raise OSError("disk gone")

        def flush(self):
            raise OSError("disk gone")

        def close(self):
            pass

    log._file = DyingDisk()  # the disk dies mid-run
    log.log_event(UserSaid(text="x"))  # must not raise
    log.close()  # must not raise
    log.log_event(UserSaid(text="late"))  # after close: dropped, no raise

    record = logging.LogRecord("l", logging.INFO, __file__, 1, "late log", None, None)
    log.handler().emit(record)  # a thread logging during teardown: no raise


async def test_threads_logging_while_events_flow_yields_only_whole_lines(tmp_path: Path):
    """The mixed real load: bus events on the event loop while camera/pynput-
    style threads log concurrently. Torn or interleaved lines would corrupt
    the timeline silently — every line must parse and none may be lost."""
    bus = Bus()
    log = SessionLog(bus, tmp_path)
    path = log.open()
    task = asyncio.create_task(log.run())
    await asyncio.sleep(0.01)

    handler = log.handler()
    lg = logging.getLogger("harp.test_session_log_burst")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    lg.propagate = False

    def burst(i: int) -> None:
        for n in range(50):
            lg.info("thread %d line %d", i, n)

    threads = [threading.Thread(target=burst, args=(i,)) for i in range(4)]
    try:
        for t in threads:
            t.start()
        for _ in range(50):
            await bus.publish(UserSaid(text="turn", final=False))
        for t in threads:
            t.join()
        await asyncio.sleep(0.05)  # drain the bus queue
    finally:
        lg.removeHandler(handler)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        log.close()

    records = _records(path)  # every line parses whole
    assert sum(1 for r in records if r["kind"] == "log") == 200
    assert sum(1 for r in records if r["kind"] == "event") == 50
