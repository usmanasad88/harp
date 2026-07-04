"""Dashboard web server — subscribes to the bus, streams it to a browser.

A read-only window into the agent for developers. Every event published to the
bus (StateChanged, PresenceChanged, PersonIdentified, GestureDetected,
UserSaid / AgentSaid, ToolRequested / ToolCompleted, Heartbeat, ErrorRaised, ...)
is forwarded to every connected browser, tagged with its class name. The
frontend renders known types specially and unknown ones generically, so new
event types show up with no server-side change. Observing only, with one
narrow exception (the mic-mute button — see below) rather than a general
write channel.

Built on `websockets` (already a dependency) rather than a separate web
framework: `process_request` serves the static page over plain HTTP, and `/ws`
carries the event stream, both on one port.

One deliberate exception to "everything comes from the bus": camera frames are
too big to be bus events (the vocabulary's "keep them small" rule), so the
server optionally takes a `snapshot() -> jpeg bytes | None` callable — wired by
app.py, backed by the shared camera — and serves it read-only at `/camera.jpg`
for the page to poll. No callable = 404, the page shows "no camera".

A second, narrower exception: the mic-mute button. `/ws` also accepts one
incoming message type, `{"type": "SetMicMuted", "muted": bool}`; the server
calls an injected `set_mic_muted(bool)` callable (real one: audio_control.py,
an OS-level mute — see that module's docstring) and, on success, publishes
`MicMuteChanged` back onto the bus so every connected dashboard — the one that
clicked and any other open tab/phone — stays in sync. This is the ONE write
action the dashboard is allowed; anything else stays observe-only. A fresh
connection also gets the current mute state once via an injected
`get_mic_muted() -> bool` callable, sent directly (not through the bus, which
never replays history). Either callable missing = that connection just doesn't
get mic-mute support this run, mirroring the snapshot=None/404 camera case.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import time
from pathlib import Path
from typing import AsyncGenerator, Callable

from websockets.asyncio.server import Server, ServerConnection
from websockets.asyncio.server import serve as ws_serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from ..core.bus import Bus
from ..core.events import ErrorRaised, Event, MicMuteChanged

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# path -> (filename under static/, Content-Type)
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


@functools.lru_cache(maxsize=None)
def _read_static(filename: str) -> bytes:
    return (_STATIC_DIR / filename).read_bytes()


SnapshotFn = Callable[[], "bytes | None"]
SetMicMutedFn = Callable[[bool], None]
GetMicMutedFn = Callable[[], bool]


def _http_response(
    status: int, reason: str, content_type: str, body: bytes, cacheable: bool = True
) -> Response:
    headers = Headers([("Content-Type", content_type), ("Content-Length", str(len(body)))])
    if not cacheable:
        headers["Cache-Control"] = "no-store"
    return Response(status, reason, headers, body)


def _process_request(
    snapshot: SnapshotFn | None, _connection: ServerConnection, request: Request
) -> Response | None:
    """Serve the static dashboard page for plain HTTP GETs; return None (i.e.
    proceed with the WebSocket handshake) for the `/ws` event stream."""
    # The page polls /camera.jpg with a ?t=... cache-buster; drop any query.
    path = request.path.split("?", 1)[0]
    if path == "/ws":
        return None
    if path == "/camera.jpg":
        return _camera_response(snapshot)
    entry = _STATIC_FILES.get(path)
    if entry is None:
        return _http_response(404, "Not Found", "text/plain; charset=utf-8", b"not found")
    filename, content_type = entry
    return _http_response(200, "OK", content_type, _read_static(filename))


def _camera_response(snapshot: SnapshotFn | None) -> Response:
    if snapshot is None:
        return _http_response(
            404, "Not Found", "text/plain; charset=utf-8", b"no camera attached"
        )
    jpeg = snapshot()
    if jpeg is None:  # camera up but no frame captured yet
        return _http_response(
            503, "Service Unavailable", "text/plain; charset=utf-8", b"no frame yet",
            cacheable=False,
        )
    return _http_response(200, "OK", "image/jpeg", jpeg, cacheable=False)


def _encode(event: Event) -> str:
    """Every bus event, tagged with its type, verbatim. `default=str` means a
    stray non-JSON-serializable field (e.g. a tool output object) degrades to
    its repr instead of dropping the connection."""
    payload = {
        "type": type(event).__name__,
        "server_ts": time.time(),
        "fields": dataclasses.asdict(event),
    }
    return json.dumps(payload, default=str)


async def _forward_events(
    connection: ServerConnection, stream: AsyncGenerator[Event, None]
) -> None:
    async for event in stream:
        await connection.send(_encode(event))


async def _handle_command(
    raw: str | bytes, bus: Bus, set_mic_muted: SetMicMutedFn | None
) -> None:
    """Parse and act on one incoming client message. Anything that isn't
    exactly the one recognized shape is silently ignored — this is the
    dashboard's only write surface, so it stays deliberately narrow rather
    than a general command channel."""
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return
    if not isinstance(msg, dict) or msg.get("type") != "SetMicMuted":
        return
    muted = msg.get("muted")
    if not isinstance(muted, bool):
        return
    if set_mic_muted is None:
        await bus.publish(
            ErrorRaised(
                where="dashboard.mic_mute",
                message="mic mute isn't available this run",
            )
        )
        return
    try:
        await asyncio.to_thread(set_mic_muted, muted)
    except Exception as exc:  # pactl missing, no default source, etc.
        await bus.publish(ErrorRaised(where="dashboard.mic_mute", message=str(exc)))
        return
    await bus.publish(MicMuteChanged(muted=muted))


async def _receive_commands(
    connection: ServerConnection, bus: Bus, set_mic_muted: SetMicMutedFn | None
) -> None:
    async for raw in connection:
        await _handle_command(raw, bus, set_mic_muted)


async def _send_initial_mic_state(
    connection: ServerConnection, get_mic_muted: GetMicMutedFn | None
) -> None:
    """A fresh connection missed any past MicMuteChanged (the bus doesn't
    replay history), so hand it the current state directly, once, outside the
    bus subscription. Best-effort: a query failure just means this connection
    starts without a known mute state — not worth failing the handshake over."""
    if get_mic_muted is None:
        return
    try:
        muted = await asyncio.to_thread(get_mic_muted)
    except Exception:
        return
    with contextlib.suppress(ConnectionClosed):
        await connection.send(_encode(MicMuteChanged(muted=muted)))


async def _stream_events(
    connection: ServerConnection,
    bus: Bus,
    set_mic_muted: SetMicMutedFn | None,
    get_mic_muted: GetMicMutedFn | None,
) -> None:
    await _send_initial_mic_state(connection, get_mic_muted)
    stream = bus.subscribe()
    forward_task = asyncio.ensure_future(_forward_events(connection, stream))
    receive_task = asyncio.ensure_future(_receive_commands(connection, bus, set_mic_muted))
    try:
        # `forward_task` can sit forever awaiting the *next* bus event with
        # nothing to send — it has no way to notice the client disconnected
        # on its own. wait_closed() is what actually unblocks this handler,
        # for a client-initiated disconnect and for server shutdown alike.
        await connection.wait_closed()
    finally:
        # Cancel and fully await both tasks *before* touching `stream`
        # ourselves: forward_task may be mid-`__anext__()` on it (cancellation
        # only propagates into the generator's `finally` if it happens to be
        # suspended there, not if it's suspended in connection.send()), and an
        # async generator can't have two callers driving/closing it at once —
        # aclose() while forward_task still holds it raises "already running".
        forward_task.cancel()
        receive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
            await forward_task
        with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
            await receive_task
        await stream.aclose()


def _handler(
    bus: Bus, set_mic_muted: SetMicMutedFn | None, get_mic_muted: GetMicMutedFn | None
):
    async def handle(connection: ServerConnection) -> None:
        await _stream_events(connection, bus, set_mic_muted, get_mic_muted)

    return handle


def _build_server(
    bus: Bus,
    host: str,
    port: int,
    snapshot: SnapshotFn | None = None,
    set_mic_muted: SetMicMutedFn | None = None,
    get_mic_muted: GetMicMutedFn | None = None,
) -> Server:
    """The `websockets` server as an async context manager. Split out from
    `serve()` so tests can bind an ephemeral port (port=0) and inspect it
    without running the forever-loop below."""
    return ws_serve(
        _handler(bus, set_mic_muted, get_mic_muted),
        host,
        port,
        process_request=functools.partial(_process_request, snapshot),
    )


async def serve(
    bus: Bus,
    host: str = "127.0.0.1",
    port: int = 8787,
    snapshot: SnapshotFn | None = None,
    set_mic_muted: SetMicMutedFn | None = None,
    get_mic_muted: GetMicMutedFn | None = None,
) -> None:
    """Run the dashboard server, bound to the bus, until cancelled. Observes
    everything; the one thing it can act on is the mic-mute button (see the
    module docstring) via `set_mic_muted`/`get_mic_muted`."""
    async with _build_server(
        bus, host, port, snapshot, set_mic_muted, get_mic_muted
    ) as server:
        print(f"HARP dashboard: http://{host}:{port}")
        await server.serve_forever()
