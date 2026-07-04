"""OS-level microphone mute, via the system mixer (`pactl`, PulseAudio/PipeWire).

Deliberately NOT an app-level "ignore this audio" flag: it mutes the physical
input device itself, so every consumer of the mic — the wake listener today,
the live voice session once the bridge exists — goes silent with no code
changes on its side. Verified on this machine: muting the default source makes
`sounddevice` capture all-zero samples, not just a Python-side toggle.

Raises on failure (`pactl` missing, no default source, etc.) rather than
swallowing errors — callers (the dashboard's mic-mute handler) turn that into
an `ErrorRaised` bus event so the failure is visible instead of silently inert.
"""

from __future__ import annotations

import subprocess

_SOURCE = "@DEFAULT_SOURCE@"


def set_mic_muted(muted: bool) -> None:
    subprocess.run(
        ["pactl", "set-source-mute", _SOURCE, "1" if muted else "0"],
        check=True,
        capture_output=True,
        text=True,
    )


def get_mic_muted() -> bool:
    result = subprocess.run(
        ["pactl", "get-source-mute", _SOURCE],
        check=True,
        capture_output=True,
        text=True,
    )
    return "yes" in result.stdout.lower()
