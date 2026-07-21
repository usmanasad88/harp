"""Tiny local HTTP server that shares gimbal face state with the face.html page.

Runs in a background thread alongside the tracker. Two endpoints:
  GET /state     -> {"face": bool, "interaction": <phase>, "ts": <unix time>}
  GET /face.html -> serves the animated face page from the same folder

`face` is webcam face-presence (set by the head tracker). `interaction` is the
push-to-talk conversation phase the app forwards here so the face can react to
the whole exchange, not just whether it sees someone:
  idle       — a session is (or was) open, nobody talking right now
  listening  — the talk key is held (face goes green)
  thinking   — key released, waiting on the reply (eyes drift)
  talking    — the agent is speaking (mouth animates)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("harp.motion.face_server")

# interaction phases the face.html page knows how to render; "idle" is the
# resting phase (no conversation activity), the default at startup.
_state = {"face": False, "interaction": "idle", "ts": time.time()}
_lock = threading.Lock()

FACE_HTML_PATH = os.path.join(os.path.dirname(__file__), "face.html")


def set_face_present(present: bool) -> None:
    with _lock:
        _state["face"] = present
        _state["ts"] = time.time()


def set_interaction(phase: str) -> None:
    """Push-to-talk conversation phase: idle | listening | thinking | talking.
    Fed by the app from the same bus events the dashboard's kiosk page reads."""
    with _lock:
        _state["interaction"] = phase
        _state["ts"] = time.time()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet — don't spam the console per request

    def do_GET(self):
        if self.path == "/state":
            with _lock:
                body = json.dumps(_state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/face.html"):
            try:
                with open(FACE_HTML_PATH, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def start(port: int = 8788) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="face-server")
    thread.start()
    logger.info("face server: http://127.0.0.1:%d/face.html", port)
    return server