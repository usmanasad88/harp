"""The always-on wake listener — HARP's ears while it sleeps.

In STANDBY there is no cloud session, so nothing else can hear. This subsystem
owns the microphone while HARP is idle and publishes `WakeRequested` on the bus
when someone seems to want attention. Two rules (both tuned in harp.yaml):

  - **Loud sound.** Level ≥ `wake_level` wakes HARP immediately.
  - **Wake word.** Level ≥ `transcribe_level` (slightly lower) starts capturing
    a phrase; a local Whisper model transcribes it, and if it contains one of
    the configured `wake_words` ("hey", "hello", "laila", ...), HARP wakes with
    the transcript passed along as model-facing context.

It releases the mic whenever the app leaves STANDBY (the live session needs it)
and resumes when STANDBY returns. Calibrate your levels with the live meter:

    uv run python -m harp.listener

  - detector.py     pure decision logic (levels in → decisions out), unit-tested
  - transcriber.py  lazy faster-whisper wrapper (model downloads on first use)
  - listener.py     the subsystem: mic + bus + state gating
"""

from .listener import AlwaysOnListener

__all__ = ["AlwaysOnListener"]
