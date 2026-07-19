# The composition root — `harp/app.py`

[← Back to index](index.md)

`app.py` is the ONE place that knows all subsystems exist. It constructs the shared `Bus`,
instantiates each enabled subsystem with it, wires the cross-subsystem callables (the seams the
bus can't carry, like "what does face-ID see right now"), and runs everything concurrently.
Because all communication goes through the bus, any subsystem can be commented out here and the
rest still runs — that is the modularity guarantee, and the recommended way to bring subsystems
online one at a time.

This page walks `run_app(provider_name)` in construction order, because the order itself encodes
the dependencies.

## 1. Settings, bus, session log

`load_settings()` reads `harp.yaml`; a single `Bus()` is created. If `session_log.enabled`, the
`SessionLog` is opened **before anything else** so even the earliest warnings (a missing camera)
land in the timeline; its `logging.Handler` is attached to the root logger, and its header
records the effective settings + resolved model/voice (`_run_header` — no secrets).

## 2. Camera and the vision pair

A `Camera` is constructed per `harp.yaml` `camera:` and started; failure (missing/busy device)
is a **warning** — `camera = None` and every camera-fed subsystem below simply stays off for the
run. A `CameraSourceState` is seeded for the dashboard dropdown, with two closures:
`get_camera_source()` (adds the camera's actually-active backend to the snapshot) and
`set_camera_source(source)` (validates, then `camera.request_switch`).

With a camera: `GestureRecognizer(bus, camera)` and `FaceID(bus, camera, store)` are built once —
the same instances both run (publishing events) and contribute their overlays to the dashboard's
camera view. `store` is the `MemoryStore` over `.harp/memory/people/`. `clean_snapshot` is a
partial of `jpeg_snapshot` **without** overlays, for the memory helper — the briefing and
`describe_scene` should see the room, not our boxes.

## 3. Memory

If `memory.enabled`: the `InteractionLogger` always runs (raw transcripts need no API), seeded
with `face_id.people_now`. Only if `GEMINI_API_KEY` is set are the helper pieces built: one
`GeminiAgent` with one `RateLimiter(calls_per_minute)`, the `MemorySummarizer`, and — camera
permitting — the `ContextWriter` (fed `people_now`, the clean snapshot, and the TTL). Without a
key, a warning explains that transcripts still record and get summarized by a later keyed run.

### `identity_context()` — the "who you're talking to" seam

A closure handed to the voice bridge, resolved fresh at each session open. Preference order:

1. the context writer's pre-computed briefing, if fresh (`context()` returns "" otherwise);
2. else, if face-ID currently recognizes someone: the static identity line — the
   `identity_context_with_notes` template when the person's record has notes, else the plain one;
3. else "" (no line at all for strangers/no camera).

## 4. Status voice and motion

`StatusVoice` is built only if `status_voice.enabled` (disabled = the orchestrator runs
silently, and the boot connectivity probe below isn't wired either, since its only consumer is
the spoken online/offline line).

If `motion.enabled`: a `MoveAroundController` — with a `conflict` closure that refuses a patrol
while follow mode is active — and, if the camera and face-ID are also up (follow needs eyes; with
no face-ID nobody can be "known"), a `FollowController` wired with `camera.latest`,
`face_id.current`, a `person_in_front` closure, an `announce` closure (fires the rule book's
`follow.*` clips via fire-and-forget tasks kept in a strong-reference set), and the reverse
conflict closure. The two conflict callbacks are how patrol and follow can never hold the motors
at once.

## 5. The session config factory and the tool dispatcher

`make_session_config()` — called fresh at **every** session open — builds the provider config
with the current `voice_tuning` stamped on, then assembles the tool list from what's enabled this
run:

- always: `search_knowledge` + `web_search` (knowledge) and `end_session` (session tools);
- `memory.enabled`: `search_memory` (needs no key);
- helper + camera present: `describe_scene`;
- motion on: `move_around`; motion + camera + face-ID: `follow_person`.

`dispatch(name, arguments)` is the one router the bridge calls for every `ToolCall`:
`end_session` → session_tools (needs the bus); `search_memory` → memory tools (store +
guestbook); `describe_scene` → the vision tool (helper + clean snapshot, with an error payload if
the helper is off this run); `move_around` / `follow_person` → the controllers (error payloads
when unavailable); everything else → `knowledge_tools.dispatch`.

## 6. Push-to-talk, tuning, and the voice bridge

If `push_to_talk.enabled`, a `PushToTalk` is built; `ptt_gate` is a closure over its `mic_open`.
`exclusive_ptt` (enabled + exclusive) drives two wiring decisions later: the wake listener is not
started, and the orchestrator receives a `wake_allowed=(reason == "button")` policy.

`voice_tuning` is the runtime `VoiceTuning` seeded from `harp.yaml`.

Then the bridge — **either** shape, same interface:

- `filter_agent.enabled` → `TwoAgentBridge(bus, provider, make_config,
  make_filter_config=..., tool_dispatch, identity_context, response_tail_seconds,
  external_mic_gate=ptt_gate, near_field_level=...)`, with the filter provider defaulting to the
  responder's.
- else → `VoiceBridge(bus, provider, make_config, tool_dispatch, identity_context,
  mic_gate=ptt_gate, near_field_level=...)`.

`near_field_level` is passed as a lambda reading `voice_tuning`, so a dashboard slider move takes
effect on the next mic chunk.

## 7. The orchestrator

`Orchestrator(bus, provider, heartbeat settings, voice_bridge, status_voice,
connectivity_check=_internet_reachable if narration is on, wake_allowed=... if exclusive_ptt)`.
`_internet_reachable` is a real TCP connect to 8.8.8.8:53 (DNS + routing + completed handshake,
not "is a cable plugged in"), run off-thread so a dead network can't stall the loop.
`_lan_ip()` (a UDP "connect" that only does a routing-table lookup) supplies the address printed
for LAN dashboard access.

## 8. The runners

A dict of named coroutines, spawned as tasks:

| Runner | Condition | Notes |
|---|---|---|
| `session_log` | session_log.enabled | First in the dict so its catch-all subscription precedes the orchestrator's boot events |
| `dashboard` | always | Wired with the snapshot (overlays), mic-mute get/set, voice-tuning get/set, camera-source get/set (camera runs only), move-around get/set (motion only), and the kiosk seeds `get_app_state` / `get_talk_key_held` |
| `orchestrator` | always | |
| `end_rules` | always | `EndOfInteractionMonitor` seeded with `is_present` from face-ID |
| `silence_rules` | silence_timeout > 0 | `SilenceMonitor` |
| `interaction_log` | memory.enabled | |
| `memory_summarizer` | helper built | |
| `context_writer` | helper + face-ID | |
| `listener` | listener.enabled and NOT exclusive PTT | in exclusive mode its wakes would all be vetoed, so it isn't started at all — no CPU burned transcribing the room, no mic held open |
| `push_to_talk` | push_to_talk.enabled | plus `idle_prompt` when status voice is on, the rule book maps the moment, and `idle_prompt_seconds > 0` |
| `gestures` + `triggers` | camera present | the `TriggerEngine` turns waves into wake requests |
| `face_id` | camera present | |

## 9. Run and teardown

`asyncio.wait(tasks, return_when=FIRST_COMPLETED)`: the orchestrator reaching STOPPING ends the
app normally; **any other task finishing first means a subsystem crashed**, and `task.result()`
re-raises that crash. The `finally` block: cancel and gather all tasks; stop any patrol or
follow mid-motion (zeroing wheels and releasing serial ports — idempotent when idle); stop the
camera; and close the session log **last** so teardown warnings are still captured.

## `main()` and logging setup

Arg parsing (`--provider`, default from `HARP_PROVIDER`, note `python -m harp.app` defaults to
gemini while `python -m harp` defaults to openai), `logging.basicConfig(INFO)`, and one quirk
worth knowing: the `websockets.server` logger is raised to WARNING because every plain HTTP GET
on the dashboard port (the page, app.js, `/camera.jpg` polled ~4×/s) logs an INFO "connection
rejected (200 OK)" line — the library's phrasing for a non-upgrade request, not an actual
problem.
