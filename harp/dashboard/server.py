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

The same server also serves `/user` — the END-USER (kiosk) page: a full-screen
visitor-facing view driven by the same `/ws` stream, showing only "hold the
green button to talk" / "Listening" (talk key held) / the agent's streamed
reply. Because the bus never replays history, a fresh connection is seeded
with the current app state and talk-key hold via two injected getters
(`get_app_state`, `get_talk_key_held`) — same pattern as the mic-mute and
voice-tuning seeds below; either missing just means no seed this run.

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

A third exception, the same shape as the second: voice noise/VAD tuning
(loudness gate, VAD threshold/silence, noise reduction — see
harp/config.VoiceTuning). `/ws` accepts `{"type": "SetVoiceTuning", "field",
"value"}`; the server validates + clamps via an injected `set_voice_tuning`
callable and broadcasts `VoiceTuningChanged`. Wired by app.py in every run
(single-agent or two-agent) — whichever agent currently owns the microphone
picks it up.

A fourth exception, the same shape again: the camera-source dropdown (see
harp/config.CameraSourceState). `/ws` accepts `{"type": "SetCameraSource",
"source"}`; the server validates via an injected `set_camera_source` callable
and broadcasts `CameraSourceChanged`. Wired by app.py only when a camera is
attached this run — mirrors the mic-mute/snapshot=None pattern for "not
available this run".

A fifth exception, with one twist: the "move around" patrol button
(harp/motion/controller). `/ws` accepts `{"type": "SetMoveAround", "active"}`
and awaits an injected ASYNC `set_move_around(active)` callable. Unlike the
others, the confirmation event is NOT published here: the controller owns
MoveAroundChanged (it must also announce the bounded lap finishing on its
own, and starts arriving via the agent's move_around tool), so this handler
only relays the request and surfaces failures as ErrorRaised. Wired by app.py
only when harp.yaml `motion.enabled` — None = no button this run.

A sixth exception, the simplest: the "face → monitor N" buttons. `/ws` accepts
`{"type": "RelaunchFaceKiosk", "monitor": int}` and calls an injected
`relaunch_face_kiosk(monitor)` which re-pops the animated-face page fullscreen
on that 1-based display — for when a visitor minimized/moved the kiosk window,
or it landed on the wrong screen and the operator (who can't see the laptop)
needs to move it. Fire-and-forget — no confirmation event, since the launcher
can't observe the window after spawning it; failures surface as ErrorRaised.
The seed carries `monitor_count` so the page renders one button per display.
Wired by app.py only when harp.yaml `motion.face_kiosk` is on — None = the
buttons never appear.

A seventh pair, simpler still (no injected callable at all): the "start/end
session" button. `/ws` accepts `{"type": "StartSession"}` and `{"type":
"EndSession"}` and just re-publishes the events the rest of the system already
acts on — WakeRequested(reason="button") to open a session (the "button"
reason also clears the exclusive push-to-talk wake veto) and
EndOfInteractionDetected(cause="dashboard") to close one. The orchestrator
owns the guards (a wake is honored only while STANDBY, an end only while
ACTIVE), so no state is threaded here; the resulting StateChanged flips the
button's label the same way it does for every other start/stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import time
import webbrowser
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable

from websockets.asyncio.server import Server, ServerConnection
from websockets.asyncio.server import serve as ws_serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from ..core.bus import Bus
from ..core.events import (
    CameraSourceChanged,
    EndOfInteractionDetected,
    ErrorRaised,
    Event,
    FaceKioskAvailable,
    MicMuteChanged,
    MoveAroundChanged,
    StateChanged,
    TalkKeyChanged,
    VoiceTuningChanged,
    WakeRequested,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# path -> (filename under static/, Content-Type)
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    # The end-user (kiosk) page: a full-screen visitor-facing view — "hold the
    # green button to talk" / "Listening" / the agent's reply — nothing else.
    "/user": ("user.html", "text/html; charset=utf-8"),
    "/user.js": ("user.js", "text/javascript; charset=utf-8"),
    "/user.css": ("user.css", "text/css; charset=utf-8"),
}


@functools.lru_cache(maxsize=None)
def _read_static(filename: str) -> bytes:
    return (_STATIC_DIR / filename).read_bytes()


SnapshotFn = Callable[[], "bytes | None"]
SetMicMutedFn = Callable[[bool], None]
GetMicMutedFn = Callable[[], bool]
# Voice noise/VAD tuning: apply(field, value) -> the full new snapshot dict;
# snapshot() -> the current one. Both injected by app.py, every run.
SetVoiceTuningFn = Callable[[str, object], dict]
GetVoiceTuningFn = Callable[[], dict]
# Camera source (auto/realsense/webcam/usb_webcam): set(source) -> {"source",
# "backend"}; get() -> the same shape. Both injected by app.py only when a
# camera is attached this run (None = the dropdown isn't available).
SetCameraSourceFn = Callable[[str], dict]
GetCameraSourceFn = Callable[[], dict]
# The "move around" patrol (harp/motion/controller): set(active) is ASYNC —
# starting opens serial ports and spawns the patrol task — and returns the
# controller's result dict ({"ok"/"error", "note"}); get() -> {"active": bool}
# for seeding. Injected by app.py only when harp.yaml `motion.enabled`.
SetMoveAroundFn = Callable[[bool], Awaitable[dict]]
GetMoveAroundFn = Callable[[], dict]
# The "face → monitor N" buttons: relaunch the animated-face page fullscreen on
# a chosen 1-based display (harp/motion/head_tracker.open_face_kiosk, already
# bound to the port by app.py — this call passes the monitor). A fire-and-forget
# action — no confirmation event; failures surface as ErrorRaised. Injected by
# app.py only when harp.yaml motion.face_kiosk is on (None = no buttons).
RelaunchFaceKioskFn = Callable[[int], None]
# Seeds for the end-user page (the bus never replays history): the current app
# state ("standby"/"active"/...) and whether the talk key is held right now.
GetAppStateFn = Callable[[], str]
GetTalkKeyHeldFn = Callable[[], bool]


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
    raw: str | bytes,
    bus: Bus,
    set_mic_muted: SetMicMutedFn | None,
    set_voice_tuning: SetVoiceTuningFn | None,
    set_camera_source: SetCameraSourceFn | None,
    set_move_around: SetMoveAroundFn | None,
    relaunch_face_kiosk: RelaunchFaceKioskFn | None,
) -> None:
    """Parse and act on one incoming client message. Only the recognized
    shapes (SetMicMuted, SetVoiceTuning, SetCameraSource, SetMoveAround,
    RelaunchFaceKiosk, StartSession, EndSession) do anything; everything else
    is silently ignored — the
    dashboard's write surface stays deliberately narrow rather than a general
    command channel."""
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return
    if not isinstance(msg, dict):
        return
    if msg.get("type") == "SetMicMuted":
        await _handle_set_mic_muted(msg, bus, set_mic_muted)
    elif msg.get("type") == "SetVoiceTuning":
        await _handle_set_voice_tuning(msg, bus, set_voice_tuning)
    elif msg.get("type") == "SetCameraSource":
        await _handle_set_camera_source(msg, bus, set_camera_source)
    elif msg.get("type") == "SetMoveAround":
        await _handle_set_move_around(msg, bus, set_move_around)
    elif msg.get("type") == "RelaunchFaceKiosk":
        await _handle_relaunch_face_kiosk(msg, bus, relaunch_face_kiosk)
    elif msg.get("type") == "StartSession":
        await _handle_start_session(bus)
    elif msg.get("type") == "EndSession":
        await _handle_end_session(bus)


async def _handle_set_mic_muted(
    msg: dict, bus: Bus, set_mic_muted: SetMicMutedFn | None
) -> None:
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


async def _handle_set_voice_tuning(
    msg: dict, bus: Bus, set_voice_tuning: SetVoiceTuningFn | None
) -> None:
    """One tuning-knob change: {type, field, value}. The setter validates and
    clamps (config.VoiceTuning.apply) and returns the full new snapshot, which
    we broadcast as VoiceTuningChanged so every open tab tracks it."""
    field = msg.get("field")
    if not isinstance(field, str) or "value" not in msg:
        return
    if set_voice_tuning is None:
        await bus.publish(
            ErrorRaised(
                where="dashboard.voice_tuning",
                message="voice tuning isn't available this run",
            )
        )
        return
    try:
        snapshot = set_voice_tuning(field, msg["value"])
    except (ValueError, TypeError) as exc:  # unknown field / unusable value
        await bus.publish(ErrorRaised(where="dashboard.voice_tuning", message=str(exc)))
        return
    await bus.publish(VoiceTuningChanged(**snapshot))


async def _handle_set_camera_source(
    msg: dict, bus: Bus, set_camera_source: SetCameraSourceFn | None
) -> None:
    """One camera-source change: {type, source}. The setter (app.py) applies
    it to the shared Camera and returns {"source", "backend"}, which we
    broadcast as CameraSourceChanged so every open tab tracks the selection —
    same confirmation-based shape as mic-mute and voice tuning."""
    source = msg.get("source")
    if not isinstance(source, str):
        return
    if set_camera_source is None:
        await bus.publish(
            ErrorRaised(
                where="dashboard.camera_source",
                message="camera source isn't available this run",
            )
        )
        return
    try:
        result = set_camera_source(source)
    except (ValueError, TypeError) as exc:  # unknown source
        await bus.publish(ErrorRaised(where="dashboard.camera_source", message=str(exc)))
        return
    await bus.publish(CameraSourceChanged(**result))


async def _handle_set_move_around(
    msg: dict, bus: Bus, set_move_around: SetMoveAroundFn | None
) -> None:
    """One patrol toggle: {type, active}. The confirmation event is NOT
    published here — the MoveAroundController owns MoveAroundChanged (it also
    announces the lap finishing on its own, and starts arriving via the agent
    tool), so this handler only awaits the controller and surfaces failures."""
    active = msg.get("active")
    if not isinstance(active, bool):
        return
    if set_move_around is None:
        await bus.publish(
            ErrorRaised(
                where="dashboard.move_around",
                message="move around isn't available this run",
            )
        )
        return
    try:
        result = await set_move_around(active)
    except Exception as exc:  # a controller crash mustn't drop the connection
        await bus.publish(ErrorRaised(where="dashboard.move_around", message=str(exc)))
        return
    if isinstance(result, dict) and result.get("error"):
        await bus.publish(
            ErrorRaised(where="dashboard.move_around", message=str(result["error"]))
        )


async def _handle_relaunch_face_kiosk(
    msg: dict, bus: Bus, relaunch_face_kiosk: RelaunchFaceKioskFn | None
) -> None:
    """The "face → monitor N" buttons: {type, monitor}. Re-pops the animated-
    face page fullscreen on the requested 1-based display — for when the kiosk
    window got minimized/moved, or landed on the wrong screen. Fire-and-forget:
    no confirmation event, failures surface as ErrorRaised. The launch spawns a
    browser (subprocess.Popen), so run it off the loop."""
    monitor = msg.get("monitor", 1)
    if not isinstance(monitor, int) or isinstance(monitor, bool) or monitor < 1:
        return
    if relaunch_face_kiosk is None:
        await bus.publish(
            ErrorRaised(
                where="dashboard.face_kiosk",
                message="face kiosk isn't available this run",
            )
        )
        return
    try:
        await asyncio.to_thread(relaunch_face_kiosk, monitor)
    except Exception as exc:  # best-effort launcher, but never drop the socket
        await bus.publish(ErrorRaised(where="dashboard.face_kiosk", message=str(exc)))


async def _handle_start_session(bus: Bus) -> None:
    """The "start/end session" button, START half: {type: "StartSession"}.
    Publishes WakeRequested(reason="button") — the same event the physical
    talk button sends, so it opens a session AND passes the exclusive
    push-to-talk wake veto (which only lets reason=="button" through). No
    injected callable and no confirmation event: the orchestrator honors it
    only while STANDBY (a click during a live session is a no-op) and the
    session it opens publishes StateChanged/InteractionStarted, which already
    reach the dashboard and flip the button's label."""
    await bus.publish(WakeRequested(reason="button", context=""))


async def _handle_end_session(bus: Bus) -> None:
    """The "start/end session" button, END half: {type: "EndSession"}.
    Publishes EndOfInteractionDetected so the orchestrator closes the live
    session (ACTIVE → STANDBY). `cause="dashboard"` lets the status rule book
    narrate a manual close distinctly; the orchestrator ignores it unless a
    session is actually open, so a stray click while idle is harmless."""
    await bus.publish(
        EndOfInteractionDetected(reason="ended from dashboard", cause="dashboard")
    )


async def _receive_commands(
    connection: ServerConnection,
    bus: Bus,
    set_mic_muted: SetMicMutedFn | None,
    set_voice_tuning: SetVoiceTuningFn | None,
    set_camera_source: SetCameraSourceFn | None,
    set_move_around: SetMoveAroundFn | None,
    relaunch_face_kiosk: RelaunchFaceKioskFn | None,
) -> None:
    async for raw in connection:
        await _handle_command(
            raw, bus, set_mic_muted, set_voice_tuning, set_camera_source,
            set_move_around, relaunch_face_kiosk,
        )


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


async def _send_initial_voice_tuning(
    connection: ServerConnection, get_voice_tuning: GetVoiceTuningFn | None
) -> None:
    """Same one-shot seeding as the mic state: a fresh tab needs the current
    tuning knobs to position its sliders. None = no panel this run (shouldn't
    happen — app.py always wires this — but keeps the server standalone-safe,
    e.g. `python -m harp.dashboard` against an empty bus)."""
    if get_voice_tuning is None:
        return
    try:
        snapshot = get_voice_tuning()
    except Exception:
        return
    with contextlib.suppress(ConnectionClosed):
        await connection.send(_encode(VoiceTuningChanged(**snapshot)))


async def _send_initial_camera_source(
    connection: ServerConnection, get_camera_source: GetCameraSourceFn | None
) -> None:
    """Same one-shot seeding as mic state/voice tuning: a fresh tab needs the
    current selection to position the dropdown. None = no camera this run."""
    if get_camera_source is None:
        return
    try:
        snapshot = get_camera_source()
    except Exception:
        return
    with contextlib.suppress(ConnectionClosed):
        await connection.send(_encode(CameraSourceChanged(**snapshot)))


async def _send_initial_move_around(
    connection: ServerConnection, get_move_around: GetMoveAroundFn | None
) -> None:
    """Same one-shot seeding as mic state/voice tuning/camera source: a fresh
    tab needs to know whether a patrol is running to label its button (the
    page keeps the button hidden until this arrives). None = motion disabled
    this run — the button never appears."""
    if get_move_around is None:
        return
    try:
        snapshot = get_move_around()
    except Exception:
        return
    with contextlib.suppress(ConnectionClosed):
        await connection.send(_encode(MoveAroundChanged(**snapshot)))


async def _send_initial_face_kiosk(
    connection: ServerConnection,
    relaunch_face_kiosk: RelaunchFaceKioskFn | None,
    face_monitor_count: int,
) -> None:
    """One-shot: reveal the face-kiosk controls only when the launcher is wired
    this run (harp.yaml motion.face_kiosk brought the face server up), and tell
    the page how many displays exist so it can render one "face → monitor N"
    button each. Nothing to toggle — the buttons are plain actions — so this
    just gates visibility, mirroring the move-around seed."""
    if relaunch_face_kiosk is None:
        return
    with contextlib.suppress(ConnectionClosed):
        await connection.send(
            _encode(FaceKioskAvailable(available=True, monitor_count=face_monitor_count))
        )


async def _send_initial_snapshots(
    connection: ServerConnection,
    get_app_state: GetAppStateFn | None,
    get_talk_key_held: GetTalkKeyHeldFn | None,
) -> None:
    """Seed a fresh connection with the current app state (as a self-transition
    StateChanged) and talk-key hold. The end-user page renders exactly these; a
    kiosk page that reconnects mid-conversation must not guess. None = not
    wired this run (standalone server, or push-to-talk disabled)."""
    events: list[Event] = []
    try:
        if get_app_state is not None:
            state = get_app_state()
            events.append(StateChanged(old=state, new=state))
        if get_talk_key_held is not None:
            events.append(TalkKeyChanged(held=get_talk_key_held()))
    except Exception:  # a getter failing shouldn't kill the handshake
        return
    with contextlib.suppress(ConnectionClosed):
        for event in events:
            await connection.send(_encode(event))


async def _stream_events(
    connection: ServerConnection,
    bus: Bus,
    set_mic_muted: SetMicMutedFn | None,
    get_mic_muted: GetMicMutedFn | None,
    set_voice_tuning: SetVoiceTuningFn | None,
    get_voice_tuning: GetVoiceTuningFn | None,
    set_camera_source: SetCameraSourceFn | None,
    get_camera_source: GetCameraSourceFn | None,
    set_move_around: SetMoveAroundFn | None,
    get_move_around: GetMoveAroundFn | None,
    relaunch_face_kiosk: RelaunchFaceKioskFn | None,
    face_monitor_count: int,
    get_app_state: GetAppStateFn | None,
    get_talk_key_held: GetTalkKeyHeldFn | None,
) -> None:
    await _send_initial_mic_state(connection, get_mic_muted)
    await _send_initial_voice_tuning(connection, get_voice_tuning)
    await _send_initial_camera_source(connection, get_camera_source)
    await _send_initial_move_around(connection, get_move_around)
    await _send_initial_face_kiosk(connection, relaunch_face_kiosk, face_monitor_count)
    await _send_initial_snapshots(connection, get_app_state, get_talk_key_held)
    stream = bus.subscribe()
    forward_task = asyncio.ensure_future(_forward_events(connection, stream))
    receive_task = asyncio.ensure_future(
        _receive_commands(
            connection, bus, set_mic_muted, set_voice_tuning, set_camera_source,
            set_move_around, relaunch_face_kiosk,
        )
    )
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
    bus: Bus,
    set_mic_muted: SetMicMutedFn | None,
    get_mic_muted: GetMicMutedFn | None,
    set_voice_tuning: SetVoiceTuningFn | None,
    get_voice_tuning: GetVoiceTuningFn | None,
    set_camera_source: SetCameraSourceFn | None,
    get_camera_source: GetCameraSourceFn | None,
    set_move_around: SetMoveAroundFn | None,
    get_move_around: GetMoveAroundFn | None,
    relaunch_face_kiosk: RelaunchFaceKioskFn | None,
    face_monitor_count: int,
    get_app_state: GetAppStateFn | None,
    get_talk_key_held: GetTalkKeyHeldFn | None,
):
    async def handle(connection: ServerConnection) -> None:
        await _stream_events(
            connection, bus, set_mic_muted, get_mic_muted,
            set_voice_tuning, get_voice_tuning,
            set_camera_source, get_camera_source,
            set_move_around, get_move_around,
            relaunch_face_kiosk, face_monitor_count,
            get_app_state, get_talk_key_held,
        )

    return handle


def _build_server(
    bus: Bus,
    host: str,
    port: int,
    snapshot: SnapshotFn | None = None,
    set_mic_muted: SetMicMutedFn | None = None,
    get_mic_muted: GetMicMutedFn | None = None,
    set_voice_tuning: SetVoiceTuningFn | None = None,
    get_voice_tuning: GetVoiceTuningFn | None = None,
    set_camera_source: SetCameraSourceFn | None = None,
    get_camera_source: GetCameraSourceFn | None = None,
    set_move_around: SetMoveAroundFn | None = None,
    get_move_around: GetMoveAroundFn | None = None,
    relaunch_face_kiosk: RelaunchFaceKioskFn | None = None,
    face_monitor_count: int = 0,
    get_app_state: GetAppStateFn | None = None,
    get_talk_key_held: GetTalkKeyHeldFn | None = None,
) -> Server:
    """The `websockets` server as an async context manager. Split out from
    `serve()` so tests can bind an ephemeral port (port=0) and inspect it
    without running the forever-loop below."""
    return ws_serve(
        _handler(
            bus, set_mic_muted, get_mic_muted, set_voice_tuning, get_voice_tuning,
            set_camera_source, get_camera_source,
            set_move_around, get_move_around,
            relaunch_face_kiosk, face_monitor_count,
            get_app_state, get_talk_key_held,
        ),
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
    set_voice_tuning: SetVoiceTuningFn | None = None,
    get_voice_tuning: GetVoiceTuningFn | None = None,
    set_camera_source: SetCameraSourceFn | None = None,
    get_camera_source: GetCameraSourceFn | None = None,
    set_move_around: SetMoveAroundFn | None = None,
    get_move_around: GetMoveAroundFn | None = None,
    relaunch_face_kiosk: RelaunchFaceKioskFn | None = None,
    face_monitor_count: int = 0,
    get_app_state: GetAppStateFn | None = None,
    get_talk_key_held: GetTalkKeyHeldFn | None = None,
    open_browser: bool = False,
) -> None:
    """Run the dashboard server, bound to the bus, until cancelled. Observes
    everything; the things it can act on are the mic-mute button, the voice
    noise/VAD tuning sliders, the camera-source dropdown, the move-around
    patrol button, and the face-kiosk monitor buttons (see the module
    docstring). `open_browser` pops the dashboard open once the socket is
    actually bound — opt-in so the standalone `python -m harp.dashboard` and
    tests importing this module don't get a surprise browser tab."""
    async with _build_server(
        bus, host, port, snapshot, set_mic_muted, get_mic_muted,
        set_voice_tuning, get_voice_tuning, set_camera_source, get_camera_source,
        set_move_around, get_move_around,
        relaunch_face_kiosk, face_monitor_count,
        get_app_state, get_talk_key_held,
    ) as server:
        print(f"HARP dashboard: http://{host}:{port}")
        if open_browser:
            # host may be 0.0.0.0 (LAN-visible bind) — that's not a browser-
            # openable address, so always open the loopback URL, which works
            # regardless of bind mode since the server listens on all of them.
            webbrowser.open(f"http://127.0.0.1:{port}")
        await server.serve_forever()
