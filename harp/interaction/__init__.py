"""Rules for the shape of an interaction — mainly, when it's over.

The orchestrator decides when to OPEN a session (wake policy); this package
decides when to CLOSE one. Keeping the two apart keeps each testable in
isolation.
"""
