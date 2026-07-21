"""Tiny local HTTP flag shared between autonomous_patrol.py and the voice
orchestrator. While patrol is active, the voice agent should refuse to open
a session; the moment E-STOP (any controller button) fires, patrol PAUSES
(not stops) and voice resumes; Triangle resumes patrol and voice is
suppressed again.

GET /patrol -> {"active": true|false, "paused": true|false}

`active` covers the whole time autonomous_patrol.py is running (paused or
not) — this is what the orchestrator's WakeRequested check reads (see
harp/app.py::_patrol_active), so "paused for a visitor" still correctly
suppresses voice-only wake words while the robot is mid-interaction some
other way. `paused` is exposed for anything that wants to distinguish
"actively driving" from "stopped, waiting for Triangle" (e.g. a future
dashboard/face indicator) — nothing currently reads it besides this module's
own tests.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("harp.motion.patrol_state")

_active = False
_paused = False
_lock = threading.Lock()


def set_active(value: bool) -> None:
    global _active
    with _lock:
        _active = value
    logger.info("patrol state: %s", "ACTIVE (voice disabled)" if value else "stopped (voice resumed)")


def is_active() -> bool:
    with _lock:
        return _active


def set_paused(value: bool) -> None:
    global _paused
    with _lock:
        _paused = value
    logger.info("patrol state: %s", "PAUSED" if value else "RUNNING")


def is_paused() -> bool:
    with _lock:
        return _paused


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/patrol":
            body = json.dumps({"active": is_active(), "paused": is_paused()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def start(port: int = 8790) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="patrol-state-server")
    thread.start()
    logger.info("patrol state server: http://127.0.0.1:%d/patrol", port)
    return server