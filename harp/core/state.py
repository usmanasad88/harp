"""The app-level state machine the orchestrator drives.

STARTING → STANDBY ⇄ ACTIVE, plus ERROR and STOPPING. Kept tiny and pure (no
I/O, no bus) so it can be unit-tested on its own: given a current state and a
target, is the move legal? The *behavior* attached to each state lives in the
orchestrator; only the shape of the machine lives here.
"""

from __future__ import annotations

from enum import Enum


class AppState(str, Enum):
    STARTING = "starting"  # booting: canned "starting up" line, dialing the cloud
    STANDBY = "standby"    # alive but idle; no cloud session; awaiting a wake condition
    ACTIVE = "active"      # a live voice session is open (someone is being helped)
    ERROR = "error"        # narrating a problem, deciding whether to retry
    STOPPING = "stopping"  # clean shutdown


# Legal transitions. The orchestrator must consult this before moving.
_ALLOWED: dict[AppState, set[AppState]] = {
    AppState.STARTING: {AppState.STANDBY, AppState.ERROR, AppState.STOPPING},
    AppState.STANDBY: {AppState.ACTIVE, AppState.ERROR, AppState.STOPPING},
    AppState.ACTIVE: {AppState.STANDBY, AppState.ERROR, AppState.STOPPING},
    AppState.ERROR: {AppState.STANDBY, AppState.STOPPING},
    AppState.STOPPING: set(),
}


def can_transition(current: AppState, target: AppState) -> bool:
    """True if moving current → target is allowed by the machine above."""
    return target in _ALLOWED.get(current, set())
