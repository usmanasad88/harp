"""Pre-programmed status voice — speech that works WITHOUT a live model.

Boot lines, connectivity notices, and error phrases ("Starting up", "Connection
established", "No internet", "One moment please") play from local audio so HARP
can speak during startup, outages, and cloud failures — exactly the moments the
cloud voice is unavailable. Bilingual EN/Urdu.

Decide when building:
  - pre-recorded clips under an `assets/` folder (fastest, no deps, consistent
    robot voice), OR
  - offline TTS (piper / espeak-ng) rendered once and cached.
Callers reference a stable id, e.g. `play("starting_up")`, never a raw path, so
the backing implementation can change freely.
"""

from __future__ import annotations


def play(line_id: str, lang: str = "en") -> None:
    """Play a canned status line by id (e.g. 'starting_up', 'no_internet')."""
    raise NotImplementedError
