"""Proactive triggers: let HARP start conversations on its own — politely.

A rule-based engine that, while idle, may ask the orchestrator to open a session
unprompted. Two families of rule:
  - cue-based:    a wave (GestureDetected) → greet and start.
  - memory-based: a known person with an open follow-up reappears → re-engage
                  ("did you find hall B?").
Guard rails (cooldowns, per-person opt-out, quiet hours) live here so HARP stays
helpful rather than pushy. The engine only *requests* a wake; the orchestrator
still owns the decision to actually open a session.
"""
