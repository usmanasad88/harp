"""Watchdog: a separate, dumb process that keeps the agent alive.

The main agent touches a liveness file (`.harp/heartbeat`) every beat (see
`Orchestrator._heartbeat`). The watchdog spawns the agent as a child, watches
that file's mtime, and restarts the agent if it either exits or goes silent —
the file stops getting touched — for too long (a hang the process-exit check
alone would miss).

Intentionally tiny and **stdlib-only** so it cannot die of the same bug — or
the same broken dependency / broken config — as the agent it guards. It does a
minimal hand-parse of `harp.yaml`'s `heartbeat:` section (never importing the
app's config code) so the staleness threshold tracks the agent's real beat, and
falls back to safe defaults if that file is missing or unreadable.

Run standalone, supervising the real agent:

    python -m harp.orchestrator.watchdog                    # runs `uv run python -m harp`
    python -m harp.orchestrator.watchdog -- python -m harp  # explicit command
    python -m harp.orchestrator.watchdog --grace 30 -- ...  # tune startup grace

Crash-loop guard: if the agent keeps dying quickly, the watchdog backs off
between restarts and gives up after too many rapid failures, rather than
spinning forever.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# The agent is launched from the repo root (harp.yaml + .harp/ live here).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Defaults, used when harp.yaml can't be read. `file`/`interval_seconds` mirror
# HeartbeatSettings in config.py; keep them in sync if those defaults change.
DEFAULT_HEARTBEAT_FILE = ".harp/heartbeat"
DEFAULT_HEARTBEAT_INTERVAL = 5.0

# The agent is "hung" once its heartbeat is this many intervals stale — enough
# slack that a slow beat or a GC pause never trips a false restart.
STALE_INTERVALS = 4

# Startup grace: how long to wait after (re)launch for the first heartbeat
# before the hang check can fire. The heavy vision stack (cv2 / insightface /
# mediapipe) can take a while to import on a cold start.
DEFAULT_GRACE_SECONDS = 30.0

# Crash-loop guard. A restart within RAPID_WINDOW of the launch counts as a
# rapid failure; MAX_RAPID_FAILURES in a row and we give up. Backoff grows
# 1, 2, 4, 8 ... capped at MAX_BACKOFF.
RAPID_WINDOW = 60.0
MAX_RAPID_FAILURES = 5
BASE_BACKOFF = 1.0
MAX_BACKOFF = 30.0

# The default command to supervise, per the plan: `uv run python -m harp`.
DEFAULT_COMMAND = ["uv", "run", "python", "-m", "harp"]

# How often the supervise loop wakes to poll child + heartbeat mtime.
POLL_SECONDS = 1.0


def _read_heartbeat_config(repo_root: Path) -> tuple[Path, float]:
    """Best-effort read of harp.yaml's `heartbeat:` section, stdlib only.

    Returns (heartbeat_file_path, interval_seconds). Any problem (no file, no
    PyYAML, malformed section) falls back to the defaults — the watchdog must
    keep working even when the config that would feed the agent is broken.
    """
    file = DEFAULT_HEARTBEAT_FILE
    interval = DEFAULT_HEARTBEAT_INTERVAL
    yaml_path = repo_root / "harp.yaml"
    try:
        in_section = False
        for raw in yaml_path.read_text().splitlines():
            # Strip trailing comments, but not the value itself.
            line = raw.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line[0].isspace():
                # A new top-level key: we're in `heartbeat:` only if it's this one.
                in_section = line.strip().rstrip(":") == "heartbeat"
                continue
            if not in_section:
                continue
            key, _, value = line.strip().partition(":")
            value = value.strip().strip('"').strip("'")
            if key == "file" and value:
                file = value
            elif key == "interval_seconds" and value:
                try:
                    interval = float(value)
                except ValueError:
                    pass
    except OSError:
        pass  # no harp.yaml — defaults are fine

    hb_path = repo_root / file
    if not hb_path.is_absolute():
        hb_path = repo_root / file
    return hb_path, max(0.5, interval)


def _heartbeat_age(hb_path: Path) -> float | None:
    """Seconds since the heartbeat file was last touched, or None if it doesn't
    exist yet (the agent hasn't written its first beat)."""
    try:
        return time.time() - hb_path.stat().st_mtime
    except OSError:
        return None


def _spawn(command: list[str], repo_root: Path) -> subprocess.Popen:
    """Launch the agent from the repo root, in its own process group so we can
    signal the whole tree (uv → python → threads) on shutdown/restart."""
    kwargs: dict = {"cwd": str(repo_root)}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # own process group; see _terminate
    return subprocess.Popen(command, **kwargs)


def _terminate(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Stop the child (and its group) gracefully, escalating to kill if it
    won't exit. Idempotent — safe to call on an already-dead process."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, OSError):
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def _log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} watchdog: {msg}", flush=True)


def supervise(
    command: list[str],
    repo_root: Path = REPO_ROOT,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
) -> int:
    """Run `command`, restarting it whenever it exits or hangs, until we're
    told to stop (Ctrl+C) or the crash-loop guard trips. Returns an exit code."""
    hb_path, interval = _read_heartbeat_config(repo_root)
    stale_after = interval * STALE_INTERVALS
    _log(
        f"supervising: {' '.join(command)}  (cwd={repo_root}, "
        f"heartbeat={hb_path}, hang if silent > {stale_after:.0f}s)"
    )

    rapid_failures = 0
    while True:
        # A fresh launch: don't judge the previous run's heartbeat file.
        try:
            hb_path.unlink()
        except OSError:
            pass

        launched_at = time.monotonic()
        proc = _spawn(command, repo_root)
        _log(f"agent started (pid {proc.pid})")

        reason = _watch(proc, hb_path, stale_after, grace_seconds)

        alive_for = time.monotonic() - launched_at
        if reason == "interrupted":
            _log("shutting down — stopping agent")
            _terminate(proc)
            return 0

        _terminate(proc)  # no-op if it already exited; kills the tree on a hang
        _log(f"agent down after {alive_for:.0f}s ({reason}) — will restart")

        # Crash-loop guard: only quick deaths count toward the give-up limit; a
        # run that stayed up past the window resets the streak.
        if alive_for < RAPID_WINDOW:
            rapid_failures += 1
        else:
            rapid_failures = 0
        if rapid_failures >= MAX_RAPID_FAILURES:
            _log(
                f"agent failed {rapid_failures} times in quick succession — "
                "giving up (fix the crash, then restart the watchdog)"
            )
            return 1

        backoff = min(MAX_BACKOFF, BASE_BACKOFF * 2 ** max(0, rapid_failures - 1))
        _log(f"restarting in {backoff:.0f}s (failure #{rapid_failures})")
        if _sleep_interruptible(backoff):
            _log("shutting down before restart")
            return 0


def _watch(
    proc: subprocess.Popen,
    hb_path: Path,
    stale_after: float,
    grace_seconds: float,
) -> str:
    """Block until the child exits, its heartbeat goes stale, or we're
    interrupted. Returns the reason: "exited" | "hung" | "interrupted"."""
    started = time.monotonic()
    try:
        while True:
            if proc.poll() is not None:
                return "exited"
            # Hang detection only after the startup grace — the agent needs
            # time to import the heavy stack and write its first beat.
            if time.monotonic() - started >= grace_seconds:
                age = _heartbeat_age(hb_path)
                if age is not None and age > stale_after:
                    _log(f"heartbeat stale ({age:.0f}s) — agent appears hung")
                    return "hung"
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        return "interrupted"


def _sleep_interruptible(seconds: float) -> bool:
    """Sleep, returning True if interrupted by Ctrl+C (caller should shut down)."""
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
        return False
    except KeyboardInterrupt:
        return True


def main() -> None:
    """Supervise a child agent process, restarting it if it dies or hangs."""
    parser = argparse.ArgumentParser(
        prog="python -m harp.orchestrator.watchdog",
        description="Keep the HARP agent alive: restart it on crash or hang.",
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE_SECONDS,
        help="seconds to wait after (re)launch before the hang check can fire "
        f"(default: {DEFAULT_GRACE_SECONDS:.0f})",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the agent command to supervise, after '--' "
        f"(default: {' '.join(DEFAULT_COMMAND)})",
    )
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = list(DEFAULT_COMMAND)

    sys.exit(supervise(command, grace_seconds=args.grace))


if __name__ == "__main__":
    main()
