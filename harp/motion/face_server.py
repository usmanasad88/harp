"""Tiny local HTTP server that shares gimbal face state with the face.html page.

Runs in a background thread alongside the tracker. Two endpoints:
  GET /state     -> {"face": true|false, "ts": <unix time>}
  GET /face.html -> serves the animated face page from the same folder
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("harp.motion.face_server")

_state = {"face": False, "ts": time.time()}
_lock = threading.Lock()

FACE_HTML_PATH = os.path.join(os.path.dirname(__file__), "face.html")


def set_face_present(present: bool) -> None:
    with _lock:
        _state["face"] = present
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