"""Tiny local HTTP flag shared between autonomous_patrol.py and the voice
orchestrator. While patrol is active, the voice agent should refuse to open
a session; the moment E-STOP fires, patrol clears the flag and voice resumes.

GET /patrol -> {"active": true|false}
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("harp.motion.patrol_state")

_active = False
_lock = threading.Lock()


def set_active(value: bool) -> None:
    global _active
    with _lock:
        _active = value
    logger.info("patrol state: %s", "ACTIVE (voice disabled)" if value else "stopped (voice resumed)")


def is_active() -> bool:
    with _lock:
        return _active


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/patrol":
            body = json.dumps({"active": is_active()}).encode()
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