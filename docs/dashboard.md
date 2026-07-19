# Dashboard (`harp/dashboard/`)

[← Back to index](index.md)

One small web server, two pages: the **developer dashboard** at `/` (a live window into the
agent) and the **end-user kiosk page** at `/user` (the full-screen visitor-facing display). Both
are driven by the same `/ws` websocket stream of bus events.

| File | Role |
|---|---|
| `server.py` | The websocket + static-file server |
| `static/index.html`, `app.js`, `styles.css` | The developer dashboard frontend |
| `static/user.html`, `user.js`, `user.css` | The kiosk page frontend |
| `__main__.py` | `python -m harp.dashboard` — standalone against an empty bus |

Configuration: `harp.yaml` `dashboard:` — `bind: localhost` (this PC only) or `network`
(LAN-visible; `app.py` prints the LAN IP), `port` (default 8787), and `open_browser` (pop the
dashboard open at boot).

## `server.py` — design

Built directly on the `websockets` library (already a dependency) rather than a web framework:
its `process_request` hook serves the static files over plain HTTP, and `/ws` carries the event
stream — both on one port.

**Observing, with a deliberately narrow write surface.** Every bus event is forwarded to every
connected browser as `{"type": <event class name>, "server_ts", "fields": {...}}` (`_encode`,
with `default=str` so a stray non-JSON field degrades to its repr instead of dropping the
connection). The frontend renders known types specially and unknown ones generically in the raw
log — so new event types show up with **no server or frontend change**. Incoming messages are
limited to exactly five recognized shapes; everything else is silently ignored:

| Incoming message | Injected callable | Confirmation event |
|---|---|---|
| `{"type": "SetMicMuted", "muted"}` | `set_mic_muted` (→ `audio_control.py`, an OS-level mute) | server publishes `MicMuteChanged` on success |
| `{"type": "SetVoiceTuning", "field", "value"}` | `set_voice_tuning` (→ `VoiceTuning.apply`, which validates + clamps) | server publishes `VoiceTuningChanged` with the full new snapshot |
| `{"type": "SetCameraSource", "source"}` | `set_camera_source` (→ `CameraSourceState.select` + `Camera.request_switch`) | server publishes `CameraSourceChanged` |
| `{"type": "SetMoveAround", "active"}` | `set_move_around` (async → `MoveAroundController.set_active`) | **not** published here — the controller owns `MoveAroundChanged`, since it must also announce the agent tool starting a patrol and the lap finishing on its own; this handler only relays and surfaces failures as `ErrorRaised` |

Every callable is optional (None = "not available this run"): a request against a missing one
publishes an `ErrorRaised` explaining that, mirroring the camera case below. Failures inside a
callable are caught and published as `ErrorRaised`, never allowed to drop the connection.

**Camera frames are the one non-bus data path**: frames are too big to be bus events, so the
server takes an optional `snapshot() -> jpeg bytes | None` callable and serves it read-only at
`/camera.jpg` (404 = no camera this run; 503 = camera up but no frame yet; `Cache-Control:
no-store`). The page polls it ~4×/s with a cache-busting query.

**Connection seeding**: the bus never replays history, so each fresh `/ws` connection is first
sent, directly (outside the bus): the current mic-mute state, the voice-tuning snapshot, the
camera source, the move-around state, and — for the kiosk page — the current app state (as a
self-transition `StateChanged`) and talk-key hold, all via injected getters. Any getter that is
missing or fails just means no seed for that panel; the handshake never fails over it.

**Connection lifecycle** (`_stream_events`): per connection, one task forwards bus events out and
one receives commands in; the handler itself awaits `connection.wait_closed()` (the forward task,
blocked awaiting the *next* bus event, has no way to notice a disconnect on its own). Teardown
cancels and fully awaits both tasks **before** closing the bus subscription — the comments
document the async-generator hazard this ordering avoids ("already running" from two drivers of
one generator).

`serve()` runs the server forever; `_build_server()` is split out so tests can bind an ephemeral
port and inspect it. `open_browser=True` opens the loopback URL once the socket is bound
(loopback deliberately, since a `0.0.0.0` bind address isn't browser-openable).

## The developer dashboard (`static/app.js`)

Vanilla JS, no build step. Connects to `/ws` with exponential-backoff reconnect. Panels:

- **State** — the latest `StateChanged` ("standby → active").
- **Presence / person / gesture** — latest `PresenceChanged`, `PersonIdentified` (name, known?,
  confidence), `GestureDetected`.
- **Heard while idle** — every `PhraseHeard` with a wake/no-wake badge: exactly what the wake
  listener's ears picked up and why it did or didn't wake.
- **Transcript** — `UserSaid`/`AgentSaid` accumulated properly: consecutive non-final pieces from
  one speaker append to one turn; `InteractionStarted/Ended` insert dividers.
- **Tool calls** — `ToolRequested` rows (name + pretty-printed arguments, "pending" badge)
  matched by id with `ToolCompleted` ("done" + output).
- **Health** — heartbeat age with a staleness badge (>10 s = red), plus an errors list
  (fatal/non-fatal).
- **Camera** — the polled `/camera.jpg` with the gesture/face overlays baked in server-side, and
  the **camera source** dropdown (auto / RealSense / laptop webcam / USB webcam) with a badge
  showing which backend `auto` actually resolved to.
- **Voice tuning** — sliders for the loudness gate, VAD threshold, VAD silence, and a
  noise-reduction select.
- **Mic mute** and **Move around** buttons.
- **Raw log** — every event verbatim, including types with no dedicated panel.

A consistent interaction rule: all controls are **confirmation-based, not optimistic** — clicking
sends the request, but the label/slider/dropdown only changes when the confirming event comes
back over the bus. That is what keeps every open tab (and a phone on the LAN) in sync, and what
makes the move-around button track patrols started by the *agent* too. Controls are hidden or
disabled until their seed arrives ("mic — state unknown"), and known-stale state is discarded on
disconnect. Sliders send only on release (not while dragging), and a control the user is actively
touching is never yanked by an incoming update.

## The kiosk page (`static/user.js`, `/user`)

The visitor-facing screen (put it full-screen with F11). The same `/ws` stream rendered as ONE
full-screen state instead of panels, with strict priority:

1. `offline` — not connected: "Connecting…"
2. `listening` — the talk key is held (`TalkKeyChanged`): whole screen green, "Listening" (a held
   key always wins)
3. `responding` — the agent's reply, streamed in as it speaks (`AgentSaid` accumulation; any
   reply text beats the dots)
4. `thinking` — key released, reply on its way: pulsing dots
5. `idle` — "Hold the green button to talk" (EN + Urdu)

Kiosk-hardening details worth knowing: a finished reply lingers for its estimated remaining
*speaking* time (~65 ms/character from turn start, clamped 3–30 s) because the text finishes
streaming well before the audio finishes playing; "thinking" gives up after 12 s if no reply ever
starts; a reply that stalls mid-stream without a final piece is swept after 10 s (a kiosk must
never wedge); holding the key clears any old reply (barge-in: the tail is dropped); leaving the
ACTIVE state resets to the idle prompt; and on reconnect the page trusts only the server's seeds,
never pre-disconnect leftovers.

## `audio_control.py` (in `harp/`, used only by the dashboard)

The mic-mute implementation: an **OS-level** mute of the default input device via `pactl`
(PulseAudio/PipeWire) — deliberately not an app-level "ignore this audio" flag, so every consumer
of the mic (wake listener, live session) goes silent with no code changes. It raises on failure
(`pactl` missing, no default source); the dashboard handler converts that into a visible
`ErrorRaised`. Note: `pactl` is a Linux tool — on Windows the button reports an error rather than
muting.

## `__main__.py` — standalone

`python -m harp.dashboard` binds the server to a fresh, empty bus: every panel shows its honest
"nothing yet" state. This is the intended way to verify the dashboard itself before wiring
anything real to it.
