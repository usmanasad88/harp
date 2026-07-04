"""The supervisor that turns a set of independent subsystems into HARP.

The orchestrator owns the state machine (core/state.py), subscribes to the bus
(core/bus.py), and decides WHEN to do expensive things — above all, when to open
and close a cloud voice session (harp/voice). It also narrates errors out loud
and retries them. It is the only runtime component that sees the whole system;
every subsystem stays deliberately ignorant of every other.

  - orchestrator.py  the state machine + wake/close policy + provider bridging
  - watchdog.py      a separate process that restarts the agent if it dies
  - status_voice.py  canned status speech that works without a live model
  - retry.py         error-narration + retry/backoff policy (pure functions)
"""
