"""Error-narration + retry policy — pure functions, no I/O.

When a subsystem raises a non-fatal error, the orchestrator should (1) say what
went wrong in plain language (status_voice) and (2) retry the failed action with
sensible backoff, rather than failing silently or hammering the cloud. This
module holds only the *policy* — how long to wait, when to give up — so it can be
unit-tested without triggering real failures.
"""

from __future__ import annotations

# Tunables. `attempt` counts consecutive failures (1 = first retry); the
# orchestrator resets its counter once things work again.
BASE_DELAY = 1.0  # seconds before the first retry
MAX_DELAY = 30.0  # exponential growth is capped here
MAX_ATTEMPTS = 5  # give up after this many consecutive failures...
MAX_ELAPSED = 120.0  # ...or once we've been failing this long overall


def backoff_seconds(attempt: int) -> float:
    """Delay before retry number `attempt`: exponential with a ceiling
    (1s, 2s, 4s, 8s, ... capped at MAX_DELAY)."""
    return min(MAX_DELAY, BASE_DELAY * 2 ** max(0, attempt - 1))


def should_give_up(attempt: int, elapsed: float) -> bool:
    """True when we've retried enough / long enough and should stop and narrate a
    hard failure (orchestrator then heads to STOPPING or back to STANDBY)."""
    return attempt >= MAX_ATTEMPTS or elapsed >= MAX_ELAPSED
