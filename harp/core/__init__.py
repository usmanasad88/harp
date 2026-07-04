"""Core plumbing shared by every subsystem — and nothing subsystem-specific.

`core` is the spine that lets the subsystems stay ignorant of each other:

  - `bus.py`    an async publish/subscribe event bus
  - `events.py` the shared vocabulary of events that travel on the bus
  - `state.py`  the app-level state machine (STARTING / STANDBY / ACTIVE / ...)

Rule of thumb: subsystems NEVER import each other. They publish events to the
bus and react to events from it. Only the orchestrator (and `app.py`, which
wires everything up) sees the whole picture. Keeping the dependency arrows all
pointing at `core` is what makes each subsystem independently buildable.
"""
