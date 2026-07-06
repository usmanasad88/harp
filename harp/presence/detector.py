"""Presence is provided by face-ID — this is a reserved seam, not wired.

`PresenceChanged` (is anyone here, and roughly how many) is already published by
vision/face_id.py: it watches the shared camera anyway, so it announces presence
transitions for free and the interaction end-rules consume them. A separate
detector would just open the same camera twice.

This stub stays only as the place a *non-camera* presence source would go (e.g.
a PIR/ultrasonic sensor, or presence while the camera is off) — publishing the
same `PresenceChanged` event so nothing downstream changes. Until such a source
exists, leave it unwired.
"""

from __future__ import annotations

from ..core.bus import Bus


class PresenceDetector:
    def __init__(self, bus: Bus) -> None:
        self._bus = bus

    async def run(self) -> None:
        """Watch a presence source and publish PresenceChanged on transitions."""
        raise NotImplementedError
