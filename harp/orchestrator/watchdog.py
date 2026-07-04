"""Watchdog: a separate, dumb process that keeps the agent alive.

The main agent emits Heartbeat events and/or touches a liveness file. The
watchdog watches for those; if they stop for too long it assumes the agent
crashed or hung and restarts it. Intentionally tiny and dependency-free so it
cannot die of the same bug as the agent it guards.

Run standalone, supervising the real agent, e.g.:
    python -m harp.orchestrator.watchdog -- python -m harp.app

To build:
  - spawn the agent as a child process,
  - monitor liveness (heartbeat-file mtime, or a pipe the agent writes to),
  - restart on death or hang, with backoff; give up after N rapid failures so a
    crash-loop doesn't spin forever.
"""

from __future__ import annotations


def main() -> None:
    """Supervise a child agent process, restarting it if it dies or hangs."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
