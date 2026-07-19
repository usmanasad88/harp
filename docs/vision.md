# Vision (`harp/vision/`)

[← Back to index](index.md)

The camera stack: one shared camera device, a gesture recognizer (the wave-to-wake cue), face
identification (which doubles as the presence signal), JPEG snapshots with overlays for the
dashboard, and the `describe_scene` tool.

| File | Role |
|---|---|
| `camera.py` | The single owner of the camera device (RealSense or webcam), background capture thread, live source switching |
| `frames.py` | `jpeg_snapshot()` + the `Overlay` type — the dashboard's read-only camera view |
| `gestures.py` | MediaPipe gesture recognition → `GestureDetected("wave")` |
| `face_id.py` | InsightFace recognition → `PersonIdentified` + `PresenceChanged` |
| `describe.py` | The `describe_scene` tool (delegates to the memory helper model) |

## `camera.py` — the shared camera

Presence, face-ID, gestures, the memory helper's snapshots, and the dashboard all need frames,
but opening a webcam several times fails on most hardware — so `Camera` is the single owner.
It captures continuously on a background thread (`harp-camera`) and keeps only the **newest**
frame under a lock; consumers call `latest()` for a copy whenever they want one. Nobody needs
every frame, only the freshest, and a blocking `read()` on the asyncio loop would stall voice and
bus dispatch.

Two backends, chosen by `harp.yaml` `camera.backend`:

- `_RealSenseBackend` — the Intel RealSense **color stream only** via pyrealsense2 (imported
  lazily so webcam mode never needs the SDK). Depth belongs to the standalone motion process; a
  RealSense can be owned by only ONE process, so if you run `python -m harp.motion` alongside the
  full agent, pin this camera to `webcam`. Construction raises when no device is connected —
  which is exactly the detection `auto` mode relies on. It tries the full-rate profile first,
  then a 15 fps profile that keeps frames coming over a USB 2 link (with a warning). Frames are
  copied out of librealsense's recycled buffer pool because `latest()` keeps them indefinitely.
- `_WebcamBackend` — plain `cv2.VideoCapture` at 640×480.

Behavior:

- `start()` opens the backend off-loop (`run_in_executor`) and starts the capture thread.
  `auto` prefers a plugged-in RealSense, else the webcam. In `app.py`, a camera that fails to
  start is a warning, not a crash — gestures/face-ID just stay off for the run.
- On a failed read the thread **reconnects**: release, retry-open every second. Because reopen
  runs `_open()` again, a RealSense unplugged mid-run hands `auto` over to the webcam, and back
  when it reappears.
- `request_switch(backend, device)` — the dashboard's camera dropdown lands here (from any
  thread). The switch is only *recorded*; the capture thread applies it at the top of its next
  iteration, so the device is only ever opened/closed by the thread that owns it. A switch
  requested while a previous one is still failing to open takes over immediately.
- `active_backend` — which backend is actually driving frames right now ("realsense"/"webcam"),
  shown by the dashboard so you can see what `auto` resolved to.

## `frames.py` — snapshots and overlays

The dashboard must stay decoupled from the vision layer, so it receives a plain
`snapshot() -> bytes | None` callable (wired in `app.py`) rather than a `Camera`. This module is
that callable's implementation:

- `Overlay(label, box)` — one thing a vision service saw: a text label plus a bounding box in
  **normalized [0,1] coordinates**, so providers never need to know the frame's pixel size.
- `jpeg_snapshot(camera, overlays)` — take `camera.latest()`, ask each overlay provider what it
  currently sees (each may return one `Overlay`, a list, or `None`), draw green boxes + labels on
  the *copy*, and JPEG-encode at quality 75.

`app.py` wires two snapshot variants: the dashboard's (with gesture + face-ID overlays) and a
**clean** one (no overlays) for the memory helper — the wake briefing and `describe_scene` should
see the room, not our boxes.

## `gestures.py` — wave detection

While idle, a raised open palm is enough to start a conversation. Built on MediaPipe's
**pretrained GestureRecognizer** (downloaded once to `~/.cache/mediapipe/`) rather than
hand-rolled landmark tracking: there is no literal "wave" category, but a held-up `Open_Palm` is
a natural, reliably-classified stand-in.

`GestureRecognizer(bus, camera)` polls `camera.latest()` at 10 Hz. `process_frame()`:

1. runs recognition, records the sighting (hand bounding box from the 21 landmarks + gesture
   label) for the dashboard overlay (with a 1 s TTL so a stale box goes blank);
2. debounces two ways: the same gesture must hold for **4 consecutive frames** before it counts
   (single-frame misclassifications are ignored), and once fired it will not fire again for the
   same continuous hold — only after the gesture changes away *and* a 2 s cooldown passes. One
   physical wave → one event.
3. on a confirmed cue, publishes `GestureDetected(kind="wave")`.

The [trigger engine](architecture.md#41-while-idle-standby) (`harp/triggers/engine.py`) is the
thin translation from that event to `WakeRequested(reason="wave")` — deliberately thin because
debouncing already happened upstream; richer proactive rules (e.g. re-engaging a known person
with an open follow-up) are meant to join there later.

## `face_id.py` — who is in frame

A continuous slow loop (~1 pass / 1.5 s — InsightFace's `buffalo_l` on CPU costs a few hundred
ms per pass, so this is deliberately much slower than gestures) that identifies **everyone** in
frame, not just one person.

Per pass (`process_frame`):

1. **Detect** off-thread: every face, sorted largest-first (`faces[0]` = the most prominent).
   No faces → clear all state, publish `PresenceChanged(present=False)` once.
2. **Identify** each face: delegate to `memory/matcher.match()` against the store — this module
   never decides who someone is, it only embeds and asks memory. Unknown faces are *reported*
   with the shared `person_id="unknown"` bucket but **never stored** (a deliberate decision:
   auto-remembering strangers waits until real conversation memories exist; enrollment is the
   `people/` folder + script).
3. **Publish only changes**: `PersonIdentified` is published for each person who *newly appeared*
   since the last pass; between changes it stays quiet (subscribers care about identity changes,
   not a 1.5 s drumbeat). `PresenceChanged(present, count)` similarly only on transitions of the
   (present, count) pair — starting from "absent, 0" so an empty frame at boot isn't a spurious
   transition.
4. **Record overlays**: one labelled box per face (name + confidence, normalized coordinates)
   for the dashboard camera view, with a 4 s TTL (comfortably above the poll interval so the box
   doesn't flicker between passes).

State exposed to the rest of the system (all read via callables wired in `app.py`):

- `current` — the most prominent person right now (largest face), or None. Used by the identity
  context at session open, by the end rules' presence seed, and by follow mode's target gating.
- `people_now()` — everyone in frame as of the last pass (one identity per person_id; all
  strangers share the one "unknown" bucket). Consumed by the memory subsystem: the context
  writer's cache key and the interaction logger's participant seeding.

Face-ID **is** HARP's presence detector — `harp/presence/detector.py` is an explicitly reserved
stub, kept only as the place a non-camera presence source (PIR/ultrasonic) would go, publishing
the same `PresenceChanged` so nothing downstream would change.

## `describe.py` — the `describe_scene` tool

The realtime session doesn't stream video; what it gets at open is the pre-computed briefing.
This tool covers the rest of the conversation: when a visitor asks "what can you see?" or shows
the robot something, the model calls `describe_scene(focus?)` and the parallel Flash Lite helper
(`memory/agent.py`) describes the **current** clean camera frame.

The handler grabs the clean snapshot (`{"error": "the camera has no frame right now"}` if none),
formats the vision prompt (`prompts/describe_scene.md`, `{focus}` filled with what the model
asked to look for, default "anything notable"), and calls the helper with `wait=True` and a 15 s
timeout — it *waits* for a rate-limiter slot because a model and a visitor are audibly holding
for the answer, but still degrades to an `{"error": ...}` payload (never an exception) when the
helper is unavailable. The tool description (`prompts/describe_scene_tool.md`) tells the model to
say something brief first because the call takes a moment.
