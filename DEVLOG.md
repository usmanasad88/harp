# HARP — Development Log

**What this file is:** the running status of the build — what exists, what's
stubbed, and what to do next. If you are an agent or a person picking this up
cold, read this first, then [PLAN.md](PLAN.md) for the vision and locked
decisions.

**Division of docs:** [PLAN.md](PLAN.md) = the vision + locked decisions (don't
churn it). This file = what's actually been done. [README.md](README.md) = how to
run it.

**How to update:** keep "Current state" and "Suggested next steps" below accurate;
add a one-line milestone to the compressed "Log" when you finish a chunk. Keep it
short and factual. The full, detailed dated history (every chunk's design notes,
bugs found, test counts) lives in [DEVLOG.backup.md](DEVLOG.backup.md) — consult it
when you need the "why" behind a decision.

---

## Environment gotchas (read before running anything)

- `.env` must exist (copy from `.env.example`) with at least `GEMINI_API_KEY`
  set to run the voice core; `pytest` and other non-voice commands work without it.
- **`insightface`, `mediapipe`, and `opencv-python` (a transitive dep of both)
  fight over the `cv2/` namespace.** `insightface` pulls in plain
  `opencv-python`; `mediapipe` pulls in `opencv-contrib-python` (a superset,
  same version) — both physically install files into the same `site-packages/cv2/`
  directory. They coexist because both were resolved at the same version in one
  `uv add`. **Do not `uv remove` `insightface` or `mediapipe` individually** — it
  deletes files the other package also owns and leaves `cv2` broken (`import cv2`
  succeeds but `cv2.__version__` etc. fail). Fix if it recurs:
  `rm -rf .venv/lib/*/site-packages/cv2* .venv/lib/*/site-packages/opencv_*`
  then `uv sync --reinstall`.
- **This mediapipe version (0.10.35) removed the old `mp.solutions` API**
  (`mp.solutions.hands` etc. — what most MediaPipe tutorials online use). Only the
  Tasks API remains (`mediapipe.tasks.python.vision.{HandLandmarker,GestureRecognizer}`,
  model files downloaded from `storage.googleapis.com/mediapipe-models/...` on
  first use, cached in `~/.cache/mediapipe/`).
- **A cold webcam open + first `.read()` can take 1-2s** (sensor/USB warm-up,
  reproduced identically with plain `ffmpeg`, not an OpenCV or `harp` issue) —
  irrelevant once `Camera` is opened once at startup and kept open, but worth
  knowing if a one-shot script's first frame comes back `None`.
- **On this (Windows) machine `cv2.VideoCapture(0)` open can hang indefinitely**
  (OS-level camera block, sibling of a mic-access block seen earlier). It runs in
  the default executor, so boot never proceeds and the process can't exit. A
  timeout around the camera open is a good next vision-robustness step.
- **The dashboard mic-mute uses `pactl` ([audio_control.py](harp/audio_control.py))
  — Linux/PulseAudio only, so it won't work on Windows.** Both call sites catch the
  failure (→ `ErrorRaised`), so it's non-fatal; a cross-platform mute is a good next
  step.
- Note: the detailed history in [DEVLOG.backup.md](DEVLOG.backup.md) has older
  Linux-host `uv`-on-PATH notes; on this Windows host use `uv run ...` as in
  [README.md](README.md).

---

## Current state

**Working (implemented and runnable):**

- **Entry point:** `python -m harp` now runs the **full supervised agent** (see
  app.py below) — same as `python -m harp.app`. `python -m harp --voice-only`
  runs just the bare voice core described next.
- **Real-time voice core — now with data retrieval.** `python -m harp
  --voice-only [--provider gemini|openai]` — talk into the mic, HARP answers
  through the speakers, grounded in `data/` via `search_knowledge`. Bilingual
  EN/Urdu.
  - Provider abstraction: [harp/voice/provider.py](harp/voice/provider.py) — one
    interface (`SessionConfig`, normalized `VoiceEvent`s, `VoiceConnection`).
  - Backends: [harp/voice/gemini.py](harp/voice/gemini.py) and
    [harp/voice/openai.py](harp/voice/openai.py).
  - Runner + audio I/O: [harp/voice/session.py](harp/voice/session.py),
    [harp/voice/audio_io.py](harp/voice/audio_io.py). The runner takes an
    injected `tool_dispatch` and returns a tool's result to the model with
    `respond_tool` — so the bare `python -m harp` path retrieves from `data/`
    exactly like harp.app, just without the bus/dashboard. [__main__.py](harp/__main__.py)
    is the composition root that attaches `knowledge_tools.declarations(...)` to
    the config and wires `knowledge_tools.dispatch`; it prints the indexed-chunk
    count at startup. Provider/mic/speaker are injectable, so the runner is
    unit-tested with fakes ([tests/test_session.py](tests/test_session.py), 4).
  - Config/persona: [harp/config.py](harp/config.py); persona in
    [prompts/system_instructions.md](prompts/system_instructions.md).
- **Model/voice are shared with the sandbox** via `.env` (`REALTIME_MODEL` /
  `REALTIME_VOICE`), so `web-realtime/` and `python -m harp` stay in sync.
  The built-in OpenAI default is `gpt-realtime-2` (the model the sandbox
  proved out), used if `.env` sets nothing.
- **Voice bridge — the supervised agent talks end to end.**
  [harp/voice/bridge.py](harp/voice/bridge.py) (`VoiceBridge`) runs one live
  session (provider connection + mic + speaker) and translates its VoiceEvents
  onto the bus: UserSaid/AgentSaid (with final markers so the dashboard closes
  turns), ToolRequested/ToolCompleted around a tool dispatch, ProviderError →
  ErrorRaised (which triggers the orchestrator's close + backoff). At session
  open it sends the wake context plus face-ID's "you are talking to <name>
  (+notes)" line. The orchestrator runs it: `_open_session` starts
  `bridge.run(context)` as a task, `_close_session` cancels it; a bridge that
  ends on its own publishes `EndOfInteractionDetected` so the app returns to
  STANDBY; a crash becomes `ErrorRaised(voice.session)`. The bridge is injected
  (app.py composes it). Tests: [tests/test_bridge.py](tests/test_bridge.py),
  + bridge-driving tests in [tests/test_orchestrator.py](tests/test_orchestrator.py).
- **Knowledge: search_knowledge is live (ported from the sandbox).**
  [harp/knowledge/retriever.py](harp/knowledge/retriever.py) is a Python port
  of web-realtime/knowledge.js — BM25 keyword search over `data/*.md` chunked at
  headings (no embeddings; right for this corpus size; indexer.py remains the
  reserved seam for a future vector store).
  [harp/knowledge/tools.py](harp/knowledge/tools.py) provides the declaration
  in both providers' formats (OpenAI GA session format with `tool_choice: auto`;
  Gemini `function_declarations` dicts) + `dispatch()` which returns results or
  `{"note": "no matches found"}` / `{"error": ...}` instead of raising.
  Tests: [tests/test_knowledge.py](tests/test_knowledge.py) (8).
- **Always-on wake listener.** [harp/listener/](harp/listener/) — owns the mic
  while HARP is idle (releases it whenever the app leaves STANDBY). Two wake
  rules, both tuned in `harp.yaml`: loudness ≥ `wake_level` wakes immediately;
  sound ≥ `transcribe_level` captures a phrase, transcribes it locally
  (faster-whisper, lazy-loaded; **offline-first** — once cached it loads with
  `local_files_only=True` so a flaky network can't stall the load), and wakes if
  a configured wake word is in it. Publishes `WakeRequested(reason, context)`
  where `context` is model-facing text the orchestrator passes to the live
  session at open. Calibrate levels with the live meter:
  `uv run python -m harp.listener`. Detector logic is pure and unit-tested
  ([tests/test_listener.py](tests/test_listener.py)). **Mic-open failures retry
  instead of crashing the agent**; native-rate-and-resample in audio_io is the
  next audio-robustness step (mics here reject a 16 kHz open).
- **User-tweakable settings file.** [harp.yaml](harp.yaml) (thresholds, wake
  words, whisper model, heartbeat) loaded by `load_settings()` in
  [harp/config.py](harp/config.py); missing file/keys fall back to defaults,
  typo'd keys warn instead of crash ([tests/test_config.py](tests/test_config.py)).
  Secrets stay in `.env`.
- **Status voice — canned lines play at boot / errors / standby / shutdown.**
  [orchestrator/status_voice.py](harp/orchestrator/status_voice.py)
  (`StatusVoice`) resolves a stable id → clip via
  `assets/status_voice/manifest.json` and plays it (serialized; every failure —
  missing clip, broken manifest, no audio device — is a logged no-op, never a
  crash). The orchestrator narrates at each transition (injected, so tests run
  silent): boot → `starting_up`, then a real connectivity probe
  (`_internet_reachable` in app.py, TCP to 8.8.8.8:53 off-thread) →
  `connection_established` / `no_internet`; normal session end → `going_standby`;
  error → `mic_problem`/`connection_lost`/`error_recoverable` by `where`, fatal →
  `error_fatal`; shutdown → `shutting_down`. Toggle in harp.yaml
  (`status_voice.enabled`/`lang`). Tests:
  [tests/test_status_voice.py](tests/test_status_voice.py) (7) + orchestrator
  narration cases.
  - **⚠ CLIPS ARE NOT IN THE REPO YET.** `.gitignore`'s blanket `*.wav` had
    silently excluded them, so only `manifest.json` was committed — the
    Kokoro-rendered WAVs lived only on the old Linux machine. `.gitignore` now
    has an exception (`!assets/status_voice/**/*.wav`). Until the
    `assets/status_voice/en/*.wav` files (matching the manifest ids) are dropped
    in, status voice stays SILENT (the no-op path); everything else runs
    normally. Regenerate with
    [scripts/generate_status_voice.py](scripts/generate_status_voice.py) (needs
    a Kokoro venv) or copy the clips over, then `git add` them.
- **Orchestrator.** [harp/orchestrator/orchestrator.py](harp/orchestrator/orchestrator.py)
  + [retry.py](harp/orchestrator/retry.py). The supervisor state machine boots
  STARTING→STANDBY, honors `WakeRequested` (only while STANDBY), closes on
  `EndOfInteractionDetected`, handles errors (non-fatal → ERROR + capped
  exponential backoff + back to STANDBY; fatal or budget exhausted → STOPPING),
  graceful shutdown, periodic Heartbeat — published on the bus AND written as a
  liveness file (`heartbeat.file` in harp.yaml, default `.harp/heartbeat`) for
  the future cross-process watchdog. Runs the real voice bridge (see above).
- **Vision: camera + face-ID detection + gesture cue.** [harp/vision/](harp/vision/):
  - `camera.py` — single shared `cv2.VideoCapture`, capture on a background
    thread, `latest()`/`stop()`, reconnects on device drop-out. Verified against
    a real Logitech C310.
  - `face_id.py` — InsightFace (`buffalo_l`, CPU) detects + embeds **every** face
    in frame, matching each against memory/store. A **continuous slow loop**
    (`run()`, ~1 pass / 1.5 s, off-thread) publishes a `PersonIdentified` per
    person as who-is-in-frame *changes*. **Unknown faces are report-only**
    (`person_id="unknown"`, never stored). Also **doubles as presence**:
    publishes `PresenceChanged(present, count)` on change (the end-rules consume
    this). Exposes `current` (most prominent face) for the identity line and
    `overlays()` for the dashboard camera view. Wired into app.py.
  - `gestures.py` — a raised **Open_Palm**, held then released, is the proactive
    greeting cue (`GestureDetected(kind="wave")`), via MediaPipe's pretrained
    `GestureRecognizer` with a hold-frames + release-then-cooldown debounce so one
    raised palm fires exactly one event (verified live). Also feeds the dashboard
    camera overlay (top gesture + hand bbox), exposed via `overlay()`.
  - Tests: [tests/test_camera.py](tests/test_camera.py) (5),
    [tests/test_face_id.py](tests/test_face_id.py), [tests/test_gestures.py](tests/test_gestures.py)
    — none need real hardware or models. Manual verification scripts (real webcam
    + models): `scripts/preview_camera.py`, `scripts/preview_face_id.py`,
    `scripts/preview_gestures.py`.
  - Deps added: `insightface`, `onnxruntime`, `mediapipe` (see cv2 gotcha above).
- **Dashboard.** [harp/dashboard/](harp/dashboard/) — a read-only web view of the
  bus: `uv run python -m harp.dashboard`, open http://127.0.0.1:8787. Renders
  StateChanged, Presence/PersonIdentified/GestureDetected, the transcript, tool
  calls, Heartbeat/ErrorRaised, plus a raw log of every event verbatim (so event
  types added later need no dashboard change). Also: a "Heard while idle" panel
  (`PhraseHeard`, wake/no-wake badge) and a live camera view (polls `/camera.jpg`,
  served read-only from an optional snapshot callable; 404 = no camera) with the
  gesture overlay drawn over the frame. Built on `websockets` (`process_request`
  serves the static page; `/ws` streams events). **LAN access:** bind address via
  `dashboard.bind` in [harp.yaml](harp.yaml) (`localhost` / `network` = `0.0.0.0`).
  **Mic-mute button:** the one write action the dashboard allows — mutes the
  OS-level mic ([harp/audio_control.py](harp/audio_control.py), `pactl`,
  **Linux-only, see gotcha**). `/ws` accepts `{"type": "SetMicMuted", "muted":
  bool}`; the server publishes `MicMuteChanged` back so every tab stays in sync.
  **End-user (kiosk) page:** the same server serves `/user` — a full-screen
  visitor-facing view (idle prompt "Hold the green button to talk" EN+Urdu →
  green "Listening" while the talk key is held → thinking dots → the agent's
  reply streamed in, then back to the prompt). Driven by the same `/ws` stream:
  `StateChanged`, `AgentSaid`, and the new `TalkKeyChanged` (push-to-talk
  mirrors the debounce-bridged hold onto the bus, so the arcade button's tap
  train doesn't flicker the screen). Fresh connections are seeded with the
  current app state + hold via `get_app_state`/`get_talk_key_held` getters
  (the bus never replays). Auto-reconnects if HARP restarts.
  Tested: [tests/test_dashboard.py](tests/test_dashboard.py).
- **Push-to-talk (on-demand, per-session).** [interaction/push_to_talk.py](harp/interaction/push_to_talk.py)
  — with `push_to_talk.enabled: true` in harp.yaml, a talk key (default space) is
  armed *alongside* the hands-free listener/wave. Pressing it **while idle** opens
  a session whose mic is gated (`mic_open` — real audio only while held); the gate
  clears when the session ends. Gating lives in the voice bridge (`mic_gate`), so
  it's cross-platform. Keyboard via pynput (injected, tested without a display).
  Tests: [tests/test_push_to_talk.py](tests/test_push_to_talk.py).
- **Two-agent noise/intent filter (experimental, opt-in — off by default).**
  With `filter_agent.enabled: true` in harp.yaml, a first realtime agent
  ([voice/filter_agent.py](harp/voice/filter_agent.py)) hears the room and relays
  only the intended message (as text) to the normal responder
  ([bridge.py](harp/voice/bridge.py) in a mic-less text-driven mode), wired by
  [voice/two_agent.py](harp/voice/two_agent.py) which exposes the same
  `run(context)` interface so the orchestrator is unchanged. Text relay,
  half-duplex (no barge-in). Implemented + unit-tested, **not yet verified live.**
- **Voice/noise tuning — one dashboard panel, both modes.** [harp/config.py](harp/config.py)'s
  `VoiceTuning`/`VoiceTuningSettings` (harp.yaml `voice_tuning:`) and the shared
  [voice/loudness_gate.py](harp/voice/loudness_gate.py) `LoudnessGate` apply to
  whichever agent owns the mic — the plain single-agent `VoiceBridge` (default)
  or the two-agent filter. The dashboard's "Voice tuning" panel (loudness gate,
  VAD threshold/silence, noise reduction) is wired in every run.
- **Agent can end its own session.** [interaction/session_tools.py](harp/interaction/session_tools.py)
  declares an `end_session` tool; the model calls it when the visitor says
  goodbye, closing the session via `EndOfInteractionDetected`. Tests:
  [tests/test_session_tools.py](tests/test_session_tools.py).
- **Wave → wake + auto-end.** A wave opens a session
  ([triggers/engine.py](harp/triggers/engine.py): `GestureDetected(kind="wave")`
  → `WakeRequested`), and a session ends on its own when the person walks off
  ([interaction/end_rules.py](harp/interaction/end_rules.py): no face for
  `interaction.absence_timeout_seconds`, default 10, → `EndOfInteractionDetected`).
  The end-rule seeds presence at each open from `face_id.current` (the bus won't
  replay the last `PresenceChanged`). Tested:
  [tests/test_triggers.py](tests/test_triggers.py), [tests/test_end_rules.py](tests/test_end_rules.py).
- **Per-run session log — the developer log.** [core/session_log.py](harp/core/session_log.py)
  — every run of the full agent writes one JSONL timeline to
  `.harp/logs/session-<timestamp>.jsonl`: a header with the settings/model the
  run ACTUALLY used, every bus event, and every internal log line, flushed per
  line so a crash keeps everything up to the moment it happened (no `session_end`
  record = the run died there). Old runs pruned at startup; configure/disable via
  harp.yaml `session_log:`. Tests: [tests/test_session_log.py](tests/test_session_log.py) (5).
- **Long-term memory — HARP remembers the people it talks to.**
  [harp/memory/](harp/memory/) is fully implemented around one **parallel Gemini
  Flash Lite helper agent** ([memory/agent.py](harp/memory/agent.py), harp.yaml
  `memory:`, needs `GEMINI_API_KEY`) behind a shared sliding-window rate cap (14
  calls/min — the free-tier budget). Three flows: **(1)** every conversation is
  recorded turn-by-turn ([memory/logger.py](harp/memory/logger.py),
  `.harp/memory/interactions/`, crash-safe `.part`→`.jsonl`), digested
  ([memory/parse.py](harp/memory/parse.py)), then summarized
  ([memory/summarizer.py](harp/memory/summarizer.py)) into `{summary, follow_up,
  person_facts}` attached to EVERY enrolled participant; unknown-visitor
  conversations land in `.harp/memory/guestbook.jsonl` (no face stored). A failed
  model call leaves the transcript pending and a boot sweep retries. **(2)** A
  **pre-computed wake briefing** ([memory/context.py](harp/memory/context.py)):
  when face-ID sees someone while idle, the helper fuses the frame + stored
  memories into a "who you're about to talk to" paragraph, cached/refreshed on
  people-change or every `context_ttl_seconds` (120 s) — delivered through
  app.py's `identity_context` seam at session open (zero wake latency; static
  face-ID line stays as fallback). **(3)** Mid-session tools: `search_memory`
  ([memory/tools.py](harp/memory/tools.py)) and `describe_scene`
  ([vision/describe.py](harp/vision/describe.py)). New events
  `MemoryWritten`/`ContextPrepared` appear on the dashboard/session log.
  **Verified live 2026-07-09** (briefing ready 11 s before the wave; model id
  confirmed; the turn-recording bug the first live run exposed — empty-text
  finals — is fixed, see the Log). Still to see live: `search_memory` /
  `describe_scene` calls, and a stranger landing in the guestbook.
- **app.py — the full supervised agent.** `uv run python -m harp.app
  [--provider gemini|openai]` wires ONE shared `Bus` into everything — orchestrator
  + VoiceBridge (search_knowledge + face-ID identity line composed here), wake
  listener (if `listener.enabled`), camera + gesture recognizer + face-ID slow
  loop, wave→wake trigger engine, interaction end-rules, dashboard — and runs them
  concurrently. A missing/busy webcam logs a warning and disables the camera-fed
  subsystems for that run instead of blocking the voice side. First task to exit
  wins: orchestrator reaching STOPPING is a clean shutdown; any other task
  finishing means a subsystem crashed and it's re-raised. **A wake opens a real
  conversation and the person leaving ends it** — end to end.
- **Robot body, phase 1 — standalone face tracking + PS5 teleop (no ROS).**
  [harp/motion/](harp/motion/) ports the `harpcontrol` repo's hardware stack to
  plain Python per PLAN.md "Motion / robot body": [gimbal.py](harp/motion/gimbal.py)
  (ESP32 serial head — same PID gains/limits/rest/look-around as facetracking.py,
  incl. the +15° pitch mount bias), [base_motors.py](harp/motion/base_motors.py)
  (RMD-X8 speed frames byte-identical to rmd2.py + the **mandatory deadman stop**
  the original lacked: a 20 Hz writer thread zeroes both wheels if teleop stops
  refreshing for 0.25 s, at shutdown, and as its own last act on crash),
  [teleop_ps5.py](harp/motion/teleop_ps5.py) (pygame.joystick replaces Linux-only
  evdev; **hold-to-drive** D-pad — a deliberate change from the original's
  latching presses, since holding is what feeds the deadman — Cross=stop,
  Square/Circle=speed ±100), [face_tracker.py](harp/motion/face_tracker.py)
  (YOLOv8n-face ONNX port of depth_detect.py; RealSense color+aligned depth →
  *nearest* face, plain-webcam fallback → *largest* face). Entry point:
  `uv run python -m harp.motion` (`--list-ports` shows VID:PID/serial to tell the
  ESP32 from the two motor adapters since COM letters aren't replug-stable;
  `--test-controller` verifies DualSense button indices; `--preview` shows the
  detection window). Every piece of hardware is optional — a missing
  port/camera/controller disables that piece with a warning, rest runs. Deps
  added: `pyserial`, `pygame`, `pyrealsense2` (2.58.2 wheels fine on cp312/
  Windows); model at [assets/models/yolov8n-face-lindevs.onnx](assets/models/).
  This is the STANDALONE runner; the in-app wiring followed (move_around +
  follow tools, and — 2026-07-21 — head tracking on the shared camera, see the
  Log). Tests: [tests/test_motion.py](tests/test_motion.py) (6 —
  golden RMD frame bytes, deadman timing, hardware-proven teleop signs, ESP32
  wire format). **Partially verified on the real hardware (2026-07-09, via a
  USB hub):** RealSense D435i opened live (USB 2 link — 30 fps profile still
  resolved; a 15 fps USB2 fallback profile is in as insurance) and the full
  frame→detect→depth→nearest-face path returned a real face at 4.78 m. The
  controller is a **DualShock 4** exposed by SDL's HIDAPI driver (16 buttons,
  **no hat** — D-pad is buttons 11-14, Cross=0/Circle=1/Square=2), which the
  hat-polling first cut silently missed; teleop now auto-picks
  HIDAPI vs classic DirectInput layout per controller (tested). Serial
  verified same day (after installing the missing FTDI VCP driver): **COM3**
  (CH9102 `1A86:55D4`, sn 5698005628) = the ESP32 gimbal — acked `Y55P55`
  with `New Target -> Y:55 P:55`; **COM4** (FTDI sn `A50285BIA`) = a live
  RMD-X8 — replied to a zero-speed frame with a valid status frame (temp
  33 °C, speed 0); **COM5** (FTDI sn `6`) enumerated but stayed silent to
  the same probe — NOT necessarily broken: rmd2.py never read replies on the
  old robot, so that adapter's RX line may simply not be wired and the motor
  may still accept drive commands. Left-vs-right mapping and actual motion
  are the remaining live checks (wheels off the ground first).

**Scaffolded only (skeletons — import cleanly, but raise `NotImplementedError`):**

- **core/** — `bus.py` (async pub/sub, ✅ **implemented + tested**),
  `events.py` (shared event vocabulary), `state.py` (app state machine),
  `session_log.py` (per-run dev log, ✅ implemented + tested).
- **orchestrator/** — `orchestrator.py` + `retry.py` + `status_voice.py` are ✅
  implemented (see above); only `watchdog` is still a stub.
- **presence/** — `detector.py` stays an unused stub: face-ID publishes
  `PresenceChanged`, so a separate webcam presence detector isn't needed. Kept as
  the seam for a future non-camera presence source.
- **vision/** — `camera.py`, `face_id.py` (multi-face + presence), and
  `gestures.py` are all ✅ implemented and wired.
- **knowledge/** — `retriever` + `tools` are ✅ implemented (BM25 port); `indexer`
  (future vector store) and `web_search` (internet fallback) are still stubs.
- **memory/** — ✅ ALL implemented + tested: `store.py` + `matcher.py` (JSON per
  person in `.harp/memory/people/`, cosine matcher, enrollment via
  [scripts/enroll_people.py](scripts/enroll_people.py)), `agent.py` (shared Flash
  Lite helper + rate limiter), `logger.py`, `parse.py`, `summarizer.py`,
  `context.py`, `tools.py`.
- **interaction/** — `end_rules` (face-absence auto-close) and `push_to_talk` are
  both ✅ implemented.
- **triggers/** — `engine` is ✅ implemented for the wave→wake rule; richer rules
  (known-person-with-open-follow-up → re-engage) can join later.

**Reference only (not the product, don't build on directly):**

- [spike_gemini_voice.py](spike_gemini_voice.py) — the original throwaway spike.
- [web-realtime/](web-realtime/) — the user's sandbox for tuning OpenAI features
  and RAG/tool-calling before they land in `harp/`.

## Suggested next steps

1. **Live verification (user):** run `uv run python -m harp --provider openai`
   and check: a wake word / loud sound / **wave** each open a conversation; the
   transcript + tool calls appear on the dashboard; an expo question triggers
   `search_knowledge`; an enrolled person is named on the camera overlay and
   greeted at open; **two+ people are each boxed**; and **walking out of frame
   closes the session after ~10 s**. Also a pass with `--provider gemini`.
   **Push-to-talk (on by default):** *hold space while idle* to start a gated
   session — confirm the model hears you only while the key is down and background
   noise no longer wakes/interrupts *that session*. **And the end_session tool:**
   tell the agent to close / say goodbye and confirm it hangs up (ACTIVE →
   STANDBY). Two known refinements if live testing shows them: a manual turn-commit
   on key-up (if the model is slow to reply after release — it relies on provider
   VAD for now), and a graceful drain before `end_session` tears down (if the
   spoken goodbye clips).
2. **Drop the status-voice clips in.** Playback is wired but the
   `assets/status_voice/en/*.wav` files aren't in this checkout — put them there
   (matching the manifest ids) and `git add` them (`.gitignore` now allows it) so
   HARP actually speaks.
3. **Finish live verification of long-term memory.** The 2026-07-09 live test
   (see Log) already proved: transcript → `.jsonl.done` → summary on
   `.harp/memory/people/usman-asad.json`, the pre-computed briefing
   (`ContextPrepared` 11 s before the wave), and the
   `gemini-3.1-flash-lite` model id. Still to exercise live: (a) HARP now has
   2 stored memories — wake it and check it greets with your history and an
   open follow-up; (b) ask "what can you see?" (describe_scene) and "do you
   remember me?" (search_memory); (c) a stranger-only conversation landing in
   `.harp/memory/guestbook.jsonl`; (d) turns now recording — the transcript
   should contain `"kind": "turn"` lines (the first run's bug, fixed).
4. **Hardware verification of the robot body (user, needs the robot):** plug in
   the ESP32 head, both motor adapters, the RealSense, and the PS5 controller;
   `uv run python -m harp.motion --list-ports` to map the three serial devices,
   then run with `--gimbal-port/--left-port/--right-port --preview`. Check:
   servo directions track your face (not mirror it), D-pad **held** drives the
   correct way and **release stops within the deadman window**, Cross stops,
   Square/Circle change speed, and `--test-controller` confirms the DualSense
   button indices (constants at the top of
   [teleop_ps5.py](harp/motion/teleop_ps5.py) if they differ). Then phase 2:
   wire `harp/motion/` onto the bus (harp.yaml config, voice teleop tools,
   dashboard hardware presence) per the PLAN.md sketch.
5. Then PLAN.md's remaining order: a dashboard "delete logs" button, the
   **watchdog** (last piece of phase 2), web-search fallback, **richer proactive
   triggers** (stored memories carry `follow_up` intents the trigger engine can
   match against a recognized returning face), and a silence-based end rule. Also
   still open from the gotchas: a **timeout around the camera open** (Windows hang)
   and a **cross-platform mic-mute** / native-rate audio resample. Build each
   subsystem so it runs and is testable **on its own** before wiring it into
   `app.py`.

---

## Log (compressed — full dated detail in [DEVLOG.backup.md](DEVLOG.backup.md))

One line per milestone, newest first. Test counts are the full-suite total at the
time. For the design rationale, bugs found, and per-chunk notes behind any of
these, read the matching dated entry in the backup.

- **2026-07-21** — Gimbal head tracking merged INTO `python -m harp` (was a
  separate `python -m harp.motion --gimbal-port` process that opened its own
  RealSense and starved the app's shared camera). New `HeadTracker`
  ([harp/motion/head_tracker.py](harp/motion/head_tracker.py)) reads the SAME
  shared camera as gestures/face-ID/follow (color-only → largest-face pick, like
  follow) and drives the ESP32 gimbal + face server on a worker thread, wired as
  an app runner. COM port + on/off now in `harp.yaml` (`motion.gimbal_enabled` /
  `gimbal_port` / `face_server_port`). The app also opens the fullscreen face
  page itself (`motion.face_kiosk`, Edge kiosk with a default-browser fallback),
  so `start_harp.bat` no longer needs the extra Gimbal process or the msedge
  line. Suite 268. **Serial/servo path unchanged from phase 1; not re-verified
  on the physical head.**
- **2026-07-17** — End-user (kiosk) page at `/user` on the dashboard server:
  full-screen prompt / green "Listening" (talk key held) / thinking dots /
  streamed reply, EN+Urdu. New `TalkKeyChanged` bus event (debounce-bridged,
  no tap-train flicker) + connection seeding of current state/hold. Verified
  in headless Edge against a scripted bus (all modes, barge-in, Urdu RTL,
  server-death reconnect). Suite 230.
- **2026-07-09** — Memory **verified live** (OpenAI provider, enrolled face):
  briefing pre-computed 11 s before the wave, `gemini-3.1-flash-lite` id
  confirmed, agent-driven end_session recorded. One bug found and fixed: turn
  finals on the bus carry EMPTY text (words are in the `final=False` deltas),
  so the first live run's transcripts had zero turns and got `.skipped` — the
  logger now accumulates deltas per speaker and flushes on the final (+ flush
  of in-flight turns at shutdown). The two lost conversations were rebuilt
  from the session log and summarized (2 summaries now on `usman-asad.json`).
  Suite 219.
- **2026-07-09** — Robot body phase 1 (PLAN phase 9, first chunk): `harp/motion/`
  standalone runner (`python -m harp.motion`) — gimbal PID head tracking
  (RealSense nearest-face / webcam largest-face) + PS5 hold-to-drive teleop of
  the RMD-X8 base with a new mandatory deadman stop; no ROS, all hardware
  optional. Deps: pyserial, pygame, pyrealsense2. Suite 219. **Not verified on
  the physical robot.**
- **2026-07-09** — Long-term memory (PLAN phase 6): parallel Flash Lite helper +
  shared rate cap; transcript logging → digest → summaries on all enrolled
  participants (guestbook for unknowns); pre-computed wake briefing; `search_memory`
  + `describe_scene` tools. Suite 213. **Not verified live.**
- **2026-07-08** — Voice/noise tuning generalized to single-agent mode (one
  dashboard panel for both); per-run developer session log (`.harp/logs/*.jsonl`,
  suite 188); all remaining agent prompts + wake/identity context strings moved
  out of code into `prompts/` (indexed in [prompts/README.md](prompts/README.md)).
- **2026-07-06** — Two-agent noise/intent filter (experimental, opt-in);
  filter/voice tuning knobs (loudness gate + VAD) + live dashboard sliders + filter
  debug logging; status voice wired to transitions (⚠ clips still not in repo).
  Listener now retries on mic-open failure instead of crashing. Suite ~178.
- **2026-07-05** — `python -m harp` unified to the full agent (bare core behind
  `--voice-only`); data retrieval added to the bare runner; multi-face + face-ID
  presence + auto-end rule + wave→wake; push-to-talk + agent-driven `end_session`;
  Urdu transcribed in Latin script (+ transcription-prompt-echo fix); whisper
  offline-first load. Suite 122–139.
- **2026-07-02** — Vision set (camera / face-ID detection / gesture cue) built +
  verified on real hardware; memory store + face matcher + folder-convention
  enrollment; dashboard built end-to-end (camera view `/camera.jpg`, LAN bind,
  mic-mute button, gesture overlay, PhraseHeard panel); status-voice clips
  generated (Kokoro); app.py wired (full agent + dashboard runnable). Suite 33–67.
- **2026-07-01** — Architecture skeleton scaffolded from PLAN.md (subsystems talk
  only via the core bus); `core/bus.py` implemented + tested; orchestrator
  skeleton + retry policy; voice core + OpenAI/Gemini providers; Gemini voice
  spike verified.
