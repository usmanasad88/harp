# HARP — Development Log

**What this file is:** the running status of the build — what exists, what's
stubbed, and what to do next. If you are an agent or a person picking this up
cold, read this first, then [PLAN.md](PLAN.md) for the vision and locked
decisions.

**Division of docs:** [PLAN.md](PLAN.md) = the vision + locked decisions (don't
churn it). This file = what's actually been done. [README.md](README.md) = how to
run it.

**How to update:** add a dated entry under "Log" (newest first) when you finish a
chunk, and keep "Current state" below accurate. Keep it short and factual.

---

## Environment gotchas (read before running anything)

- **`uv` is not on `PATH` in Claude Code's Bash tool.** Bare `uv ...` fails with
  `command not found` — that shell doesn't source `~/.bashrc` (interactive-only
  guard at the top). Use the full path
  **`/home/mani/.local/bin/uv`** for every `uv` command in this repo (`uv run`,
  `uv add`, `uv sync`, ...).
- **There are two `uv` binaries on this machine, different versions:**
  `/home/mani/.local/bin/uv` (0.5.24) and `/home/mani/miniconda3/bin/uv`
  (0.6.14, added by conda init). The existing `.venv` was created by 0.5.24
  (see `.venv/pyvenv.cfg`) — **always use `/home/mani/.local/bin/uv`**, not the
  conda one, to avoid a version mismatch messing with the lockfile/venv. In a
  normal interactive terminal `~/.bashrc` puts `.local/bin` first on `PATH` so
  plain `uv` already resolves correctly there — this only matters for
  non-interactive tool shells (Claude Code, CI, etc).
- `.env` must exist (copy from `.env.example`) with at least `GEMINI_API_KEY`
  set to run the voice core; `pytest` and other non-voice commands work without it.
- **`insightface`, `mediapipe`, and `opencv-python` (a transitive dep of both)
  fight over the `cv2/` namespace.** `insightface` pulls in plain
  `opencv-python`; `mediapipe` pulls in `opencv-contrib-python` (a superset,
  same version) — both physically install files into the same `site-packages/cv2/`
  directory. They currently coexist fine because both were resolved at the
  same version in one `uv add`. **Do not `uv remove` `insightface` or
  `mediapipe` individually** — it deletes files the other package also owns
  and leaves `cv2` broken (`import cv2` succeeds but `cv2.__version__` etc.
  fail). Hit this once already; fix if it recurs:
  `rm -rf .venv/lib/*/site-packages/cv2* .venv/lib/*/site-packages/opencv_*`
  then `uv sync --reinstall`.
- **This mediapipe version (0.10.35) removed the old `mp.solutions` API**
  (`mp.solutions.hands` etc. — what aura's gesture_monitor.py and most
  MediaPipe tutorials online use). Only the Tasks API remains
  (`mediapipe.tasks.python.vision.{HandLandmarker,GestureRecognizer}`,
  model files downloaded from `storage.googleapis.com/mediapipe-models/...`
  on first use, cached in `~/.cache/mediapipe/`).
- **A cold webcam open + first `.read()` can take 1-2s** (sensor/USB
  warm-up, reproduced identically with plain `ffmpeg`, not an OpenCV or
  `harp` issue) — irrelevant once `Camera` is opened once at startup and
  kept open, but worth knowing if a one-shot script's first frame comes back
  `None`.

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
    [harp/voice/audio_io.py](harp/voice/audio_io.py). The runner now takes an
    injected `tool_dispatch` and returns a tool's result to the model with
    `respond_tool` (before, tool calls were only printed) — so the bare
    `python -m harp` path retrieves from `data/` exactly like harp.app, just
    without the bus/dashboard. [__main__.py](harp/__main__.py) is the
    composition root that attaches `knowledge_tools.declarations(...)` to the
    config and wires `knowledge_tools.dispatch`; it prints the indexed-chunk
    count at startup. Provider/mic/speaker are injectable (same shape as the
    bridge), so the runner is unit-tested with fakes
    ([tests/test_session.py](tests/test_session.py), 4).
  - Config/persona: [harp/config.py](harp/config.py); persona in
    [prompts/system_instructions.md](prompts/system_instructions.md).
- **Model/voice are shared with the sandbox** via `.env` (`REALTIME_MODEL` /
  `REALTIME_VOICE`), so `web-realtime/` and `python -m harp` stay in sync.
  The built-in OpenAI default is now `gpt-realtime-2` (the model the sandbox
  proved out), used if `.env` sets nothing.
- **Voice bridge — the supervised agent talks end to end.**
  [harp/voice/bridge.py](harp/voice/bridge.py) (`VoiceBridge`) runs one live
  session (provider connection + mic + speaker, same runner shape as
  session.py) and translates its VoiceEvents onto the bus: UserSaid/AgentSaid
  (with final markers so the dashboard closes turns), ToolRequested/
  ToolCompleted around a tool dispatch, ProviderError → ErrorRaised (which
  triggers the orchestrator's close + backoff). At session open it sends the
  wake context plus face-ID's "you are talking to <name> (+notes)" line.
  The orchestrator now runs it: `_open_session` starts `bridge.run(context)`
  as a task, `_close_session` cancels it; a bridge that ends on its own
  (provider closed the stream) publishes `EndOfInteractionDetected` so the
  app returns to STANDBY instead of sticking in ACTIVE; a crash becomes
  `ErrorRaised(voice.session)`. The bridge is injected (app.py composes it) —
  an orchestrator built without one behaves exactly as before, which the
  older bus-driven tests still cover. Tests:
  [tests/test_bridge.py](tests/test_bridge.py) (7, fake provider/mic/speaker),
  +4 bridge-driving tests in [tests/test_orchestrator.py](tests/test_orchestrator.py).
- **Knowledge: search_knowledge is live (ported from the sandbox).**
  [harp/knowledge/retriever.py](harp/knowledge/retriever.py) is a Python port
  of web-realtime/knowledge.js — the proven BM25 keyword search over
  `data/*.md` chunked at headings (no embeddings; right for this corpus size;
  indexer.py remains the reserved seam for a future vector store).
  [harp/knowledge/tools.py](harp/knowledge/tools.py) provides the declaration
  in both providers' formats (OpenAI GA session format with
  `tool_choice: auto`; Gemini `function_declarations` dicts, validated
  against the installed google-genai) + `dispatch()` which returns results or
  `{"note": "no matches found"}` / `{"error": ...}` instead of raising. The
  tool description keeps the sandbox's proven levers (English keywords, call
  BEFORE answering, admit uncertainty) but is corpus-agnostic per PLAN.md.
  Tests: [tests/test_knowledge.py](tests/test_knowledge.py) (8).
- **Always-on wake listener.** [harp/listener/](harp/listener/) — owns the mic
  while HARP is idle (releases it whenever the app leaves STANDBY). Two wake
  rules, both tuned in `harp.yaml`: loudness ≥ `wake_level` wakes immediately;
  sound ≥ `transcribe_level` captures a phrase, transcribes it locally
  (faster-whisper, lazy-loaded; **offline-first** — once cached it loads with
  `local_files_only=True` so a flaky network can't stall the load, downloading
  only on a genuine cache miss), and wakes if a configured wake word is in it.
  Publishes `WakeRequested(reason, context)`
  where `context` is model-facing text (e.g. the transcript) the orchestrator
  passes to the live session at open. Calibrate levels with the live meter:
  `uv run python -m harp.listener`. Detector logic is pure and unit-tested
  ([tests/test_listener.py](tests/test_listener.py)).
- **User-tweakable settings file.** [harp.yaml](harp.yaml) (thresholds, wake
  words, whisper model, heartbeat) loaded by `load_settings()` in
  [harp/config.py](harp/config.py); missing file/keys fall back to defaults,
  typo'd keys warn instead of crash ([tests/test_config.py](tests/test_config.py)).
  Secrets stay in `.env`.
- **Status voice — canned lines play at boot / errors / standby / shutdown.**
  [orchestrator/status_voice.py](harp/orchestrator/status_voice.py)
  (`StatusVoice`) resolves a stable id → clip via
  `assets/status_voice/manifest.json` and plays it (serialized, one clip at a
  time; every failure — missing clip, broken manifest, no audio device — is a
  logged no-op, never a crash). The orchestrator narrates at each transition
  (injected, so tests run silent): boot → `starting_up`, then a real
  connectivity probe (`_internet_reachable` in app.py, TCP to 8.8.8.8:53
  off-thread) → `connection_established` / `no_internet`; normal session end →
  `going_standby`; error → `mic_problem`/`connection_lost`/`error_recoverable`
  by `where`, fatal → `error_fatal`; shutdown → `shutting_down`. Toggle in
  harp.yaml (`status_voice.enabled`/`lang`). Clips are the same 14 Kokoro
  (`af_heart`, 24 kHz mono PCM) lines the manifest describes. Tests:
  [tests/test_status_voice.py](tests/test_status_voice.py) (7, fake sink) +6
  orchestrator narration cases.
  - **⚠ CLIPS ARE NOT IN THE REPO YET.** `.gitignore`'s blanket `*.wav` had
    silently excluded them, so only `manifest.json` was ever committed — the
    Kokoro-rendered WAVs lived only on the old Linux machine and are absent from
    this checkout. `.gitignore` now has an exception
    (`!assets/status_voice/**/*.wav`) so clips get tracked once present. Until
    the `assets/status_voice/en/*.wav` files (matching the manifest ids) are
    dropped in, status voice stays SILENT (the no-op path) — everything else
    runs normally. Regenerate with
    [scripts/generate_status_voice.py](scripts/generate_status_voice.py) (needs
    a Kokoro venv) or copy the clips over, then `git add` them.
- **Orchestrator skeleton.** [harp/orchestrator/orchestrator.py](harp/orchestrator/orchestrator.py)
  + [retry.py](harp/orchestrator/retry.py), tested
  ([tests/test_orchestrator.py](tests/test_orchestrator.py),
  [tests/test_retry.py](tests/test_retry.py)). The supervisor state machine is
  real: boots STARTING→STANDBY, honors `WakeRequested` (only while STANDBY),
  closes on `EndOfInteractionDetected`, handles errors (non-fatal → ERROR +
  capped exponential backoff + back to STANDBY; fatal or budget exhausted →
  STOPPING), graceful shutdown, periodic Heartbeat — published on the bus AND
  written as a liveness file (`heartbeat.file` in harp.yaml, default
  `.harp/heartbeat`; mtime = last beat) for the future cross-process watchdog.
  Forwards `WakeRequested.context` into `InteractionStarted` and keeps it for
  the voice bridge. **Caveat:** `_open_session`/`_close_session` drive state +
  publish the right events but don't start the real voice session yet — that
  bridge is the next chunk.
- **Vision: camera + face-ID detection + gesture cue.** [harp/vision/](harp/vision/):
  - `camera.py` — single shared `cv2.VideoCapture`, capture on a background
    thread (`.read()` blocks; keeps the event loop free), `latest()`/`stop()`,
    reconnects on device drop-out. Verified against a real Logitech C310.
  - `face_id.py` — InsightFace (`buffalo_l`, CPU) detects + embeds **every**
    face in frame, matching each against memory/store via memory/matcher. A
    **continuous slow loop** (`run()`, ~1 pass / 1.5 s, detection off-thread)
    that publishes a `PersonIdentified` per person as who-is-in-frame *changes*
    (a newcomer appears / comes back after absence) — quiet for people already
    there. **Unknown faces are report-only** (`person_id="unknown"`, never
    stored — decision from the 2026-07-02 memory log entry). Also **doubles as
    presence**: publishes `PresenceChanged(present, count)` on change, which the
    end-rules consume. Exposes `current` (the most prominent face, None when
    nobody's there) for the voice bridge's identity line + presence seeding, and
    `overlays()` (a name + similarity + box per face) for the dashboard camera
    view. Wired into app.py.
  - `gestures.py` — a raised **Open_Palm**, held then released, is the
    proactive greeting cue (`GestureDetected(kind="wave")`) — this is the
    `GestureRecognizer(bus, camera).run()` app.py already wires in. Went
    through two designs: first attempt tracked raw MediaPipe HandLandmarker
    points and classified "wave" as x-position oscillation (a hand-rolled
    open-hand heuristic + zig-zag reversal counter) — passed unit tests but
    the open-hand heuristic (distance-from-wrist) gave real false negatives
    live (perspective distortion during hand motion, and requiring all 4
    fingers unanimous). Corrected to MediaPipe's own pretrained
    GestureRecognizer (canned categories: Closed_Fist, Open_Palm, Pointing_Up,
    Thumb_Down, Thumb_Up, Victory, ILoveYou — same model aura's
    gesture_monitor.py uses), with a hold-frames + release-then-cooldown
    debounce so one raised palm fires exactly one event — verified live
    (3 clean fires, no re-fire across a 12s continuous hold, correct reset on
    hand leaving frame). No literal "wave" category exists in MediaPipe, but
    this matches PLAN.md's own "palm-gesture recognition" wording; the
    multi-frame hold/cooldown logic is ours, MediaPipe only classifies each
    frame. Also feeds the dashboard camera overlay: each pass records what it
    saw (top gesture + hand bbox from the landmarks, normalized coords),
    exposed via `overlay()` — None when no hand or the sighting is stale.
  - Tests: [tests/test_camera.py](tests/test_camera.py) (5, fakes
    `cv2.VideoCapture`), [tests/test_face_id.py](tests/test_face_id.py) (5,
    fakes `FaceAnalysis` + memory), [tests/test_gestures.py](tests/test_gestures.py)
    (6, fakes `GestureRecognizer`) — none need real hardware or models.
    Manual verification scripts (real webcam + real models, not part of the
    suite): `scripts/preview_camera.py`, `scripts/preview_face_id.py`,
    `scripts/preview_gestures.py`.
  - Deps added: `insightface`, `onnxruntime`, `mediapipe`. Gotcha below if
    you touch these.
- **Dashboard.** [harp/dashboard/](harp/dashboard/) — a read-only web view of
  the bus: `uv run python -m harp.dashboard`, open http://127.0.0.1:8787.
  Renders StateChanged, Presence/PersonIdentified/GestureDetected, the
  transcript (UserSaid/AgentSaid, streamed deltas grouped into turns), tool
  calls, Heartbeat/ErrorRaised, plus a raw log of every event verbatim — so
  event types added later need no dashboard change to show up. Also: a
  "Heard while idle" panel (`PhraseHeard` from the listener, wake/no-wake
  badge) and a live camera view — the page polls `/camera.jpg`, served
  read-only from an optional snapshot callable app.py wires in (404 = no
  camera this run), with what the gesture recognizer currently sees (label +
  hand box) drawn over the frame via the overlay seam in
  [harp/vision/frames.py](harp/vision/frames.py). Built on
  `websockets` (`process_request` serves the static page; `/ws` streams
  events) rather than a new web-framework dependency. Bound to whatever `Bus`
  it's given — `python -m harp.dashboard` uses an empty one, so it honestly
  shows "nothing yet" until it's given the same bus a real orchestrator uses
  (that wiring is app.py, below). Verified end-to-end against a
  real `Orchestrator` sharing one bus, not just synthetic test events.
  Tested: [tests/test_dashboard.py](tests/test_dashboard.py). **LAN access:**
  bind address is configurable via `dashboard.bind` in [harp.yaml](harp.yaml)
  (`localhost` = this machine only, the old hard-coded default; `network` =
  `0.0.0.0`, reachable from a phone/other PC on the same Wi-Fi). `app.js`
  already builds its WebSocket URL from `location.host`, so no frontend
  change was needed — only the server bind address and app.py's startup
  print (now shows the detected LAN IP when bound to the network) changed.
  **Mic-mute button:** the one write action the dashboard is allowed (see its
  own docstring) — a big button that mutes the OS-level mic input
  ([harp/audio_control.py](harp/audio_control.py), `pactl set-source-mute`,
  works regardless of which subsystem is reading the mic). `/ws` accepts one
  incoming message shape, `{"type": "SetMicMuted", "muted": bool}`; the
  server calls an injected `set_mic_muted` callable and publishes
  `MicMuteChanged` back onto the bus so every open tab/phone stays in sync;
  a fresh connection gets the current state once via an injected
  `get_mic_muted` (the bus itself never replays history). Not unit-tested
  this round (explicit call at the time: verify live instead).
- **Push-to-talk (on-demand, per-session).** [interaction/push_to_talk.py](harp/interaction/push_to_talk.py)
  — with `push_to_talk.enabled: true` in harp.yaml, a talk key (default space) is
  armed *alongside* the hands-free listener/wave. Pressing it **while idle** opens
  a session whose mic is gated (`mic_open` — real audio only while held, silence
  otherwise), so a loud room can't interfere; the gate clears when the session
  ends and HARP returns to hands-free. Gating lives in the voice bridge
  (`mic_gate`), so it's cross-platform. Keyboard via pynput (injected, tested
  without a display). Tests: [tests/test_push_to_talk.py](tests/test_push_to_talk.py).
- **Two-agent noise/intent filter (experimental, opt-in — off by default).**
  With `filter_agent.enabled: true` in harp.yaml, a first realtime agent
  ([voice/filter_agent.py](harp/voice/filter_agent.py)) hears the room and relays
  only the intended message (as text) to the normal responder (today's
  [bridge.py](harp/voice/bridge.py) in a new mic-less text-driven mode), wired by
  [voice/two_agent.py](harp/voice/two_agent.py) which exposes the same
  `run(context)` interface so the orchestrator is unchanged. Text relay,
  half-duplex (no barge-in). Implemented + unit-tested, **not yet verified live**
  (needs a mic + API keys). See the 2026-07-06 log entry.
- **Agent can end its own session.** [interaction/session_tools.py](harp/interaction/session_tools.py)
  declares an `end_session` tool; the model calls it when the visitor says
  goodbye, and it closes the session via `EndOfInteractionDetected` (the same
  path as the face-absence rule). Tests:
  [tests/test_session_tools.py](tests/test_session_tools.py).
- **Wave → wake + auto-end.** A wave opens a session
  ([triggers/engine.py](harp/triggers/engine.py): `GestureDetected(kind="wave")`
  → `WakeRequested`), and a session ends on its own when the person walks off
  ([interaction/end_rules.py](harp/interaction/end_rules.py): no face for
  `interaction.absence_timeout_seconds`, default 10, → `EndOfInteractionDetected`).
  The end-rule seeds presence at each open from `face_id.current` (the bus
  won't replay the last `PresenceChanged`), so a session woken with nobody in
  frame still closes. Tested: [tests/test_triggers.py](tests/test_triggers.py),
  [tests/test_end_rules.py](tests/test_end_rules.py).
- **app.py — the full supervised agent.** `uv run python -m harp.app
  [--provider gemini|openai]` wires ONE shared `Bus` into everything built so
  far — orchestrator + VoiceBridge (with the search_knowledge tool and the
  face-ID identity line composed here, in the composition root), wake
  listener (if `listener.enabled` in harp.yaml), camera + gesture recognizer
  + face-ID slow loop, the wave→wake trigger engine, the interaction end-rules,
  dashboard at http://127.0.0.1:8787 — and runs them concurrently. A
  missing/busy webcam logs a warning and disables the camera-fed subsystems for
  that run instead of blocking the voice side. First task to exit wins:
  orchestrator reaching STOPPING is a clean shutdown; any other task finishing
  means a subsystem crashed and the crash is re-raised. **A wake opens a real
  conversation and the person leaving ends it** — end to end.

**Scaffolded only (skeletons — import cleanly, but raise `NotImplementedError`):**

The full architecture from PLAN.md exists as stubs so subsystems can be built
independently. Each file has a docstring + a "To build" list. Nothing below is
functional yet, except where noted.

- **core/** — the spine everything else depends on: `bus.py` (async pub/sub,
  ✅ **implemented + unit-tested**, see [tests/test_bus.py](tests/test_bus.py)),
  `events.py` (the shared event vocabulary), `state.py` (the app state machine).
- **orchestrator/** — `orchestrator.py` + `retry.py` + `status_voice.py` are
  ✅ implemented (see above), including running the voice bridge and canned
  status narration; only `watchdog` is still a stub.
- **presence/** — `detector.py` stays an unused stub: face-ID (`vision/face_id.py`)
  now publishes `PresenceChanged`, so a separate webcam presence detector isn't
  needed. Kept only as the seam for a future non-camera presence source.
- **vision/** — `camera.py`, `face_id.py` (multi-face + presence), and
  `gestures.py` are all ✅ implemented and wired (see above).
- **knowledge/** — `retriever` + `tools` are ✅ implemented (BM25 port of the
  sandbox, see above); `indexer` (future vector store) and `web_search`
  (internet fallback) are still stubs.
- **memory/** — `store.py` + `matcher.py` are ✅ implemented + tested (JSON
  per person in `.harp/memory/people/`, cosine matcher, enrollment via
  [scripts/enroll_people.py](scripts/enroll_people.py) over `people/` — see
  the log entry); `logger` and `summarizer` are still stubs (the voice bridge
  now exists, so these are unblocked).
- **interaction/** — `end_rules` (face-absence auto-close) and `push_to_talk`
  (hold-key-to-talk mode for noisy rooms) are both ✅ implemented (see above).
- **triggers/** — `engine` is ✅ implemented for the wave→wake rule; richer
  rules (known-person-with-open-follow-up → re-engage) can join later.

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
   closes the session after ~10 s** (harp.yaml `interaction.absence_timeout_seconds`).
   Also worth a pass with `--provider gemini`. **Push-to-talk (now on by default
   in harp.yaml):** *hold space while idle* to start a gated session — confirm
   the model hears you only while the key is down, background noise no longer
   wakes/interrupts *that session*, and when it ends (walk away, or say goodbye)
   HARP goes back to hands-free. **And the end_session tool:** tell the agent to
   close / say goodbye and confirm it hangs up (ACTIVE → STANDBY). Two known
   refinements if live testing shows them: a manual turn-commit on key-up (if the
   model is slow to reply after release — it relies on provider VAD for now), and
   a graceful drain before `end_session` tears down (if the spoken goodbye clips).
2. **Drop the status-voice clips in.** `status_voice` playback is now wired
   (see current state), but the `assets/status_voice/en/*.wav` files aren't in
   this checkout — put them there (matching the manifest ids) and `git add` them
   (`.gitignore` now allows it) so HARP actually speaks. `watchdog` is the last
   piece of PLAN phase 2.
3. Then PLAN.md's remaining order — in progress this cycle: memory's remaining
   half (`logger` + `summarizer`; the latter Gemini-based with a free-tier quota
   fallback), plus a dev event logger + dashboard "delete logs" button. After
   that: web-search fallback, richer proactive triggers (known-person
   follow-ups), and a silence-based end rule (a natural companion to the
   face-absence rule now in place).
   Build each subsystem so it runs and is testable **on its own** before wiring
   it into `app.py`.

---

## Log

### 2026-07-06 — Filter tuning knobs (loudness gate + VAD) + live dashboard sliders + debug

Follow-up to the two-agent filter after first live testing: with `filter_agent`
on, the filter relayed hallucinated greetings ("Hi, iPhone also.", "Peace be.")
on silence. Root cause: the filter is a native-audio realtime model, and those
invent plausible speech from ambient/near-silence when their VAD commits a turn —
the two-agent split inherits the single-agent noise problem; it only helps if the
filter reliably ignores noise. Addressed on three fronts.

- **Debuggability + two clear bugs** ([filter_agent.py](harp/voice/filter_agent.py),
  [two_agent.py](harp/voice/two_agent.py), [prompts/filter_instructions.md](prompts/filter_instructions.md)):
  the filter now logs, at INFO, both `filter heard (asr): …` (input transcription)
  and its raw output (relay vs. `[[ignore]]`), so a bogus turn is traceable to the
  audio it committed vs. the model inventing one. Stopped priming the filter with
  the responder's wake context ("someone said hello — greet them"), which was
  teaching it to emit greetings on silence — the filter now gets NO wake context.
  Hardened the persona with an explicit anti-hallucination directive.
- **(1) Loudness/proximity gate** (`LoudnessGate` in filter_agent.py): only mic
  audio at/above a live RMS threshold reaches the filter; quieter room noise
  becomes digital silence, so it never commits a turn. Pre-roll (no clipped
  onsets) + hangover (no chopped words). Same RMS formula as
  `listener.detector.rms_level`, so it calibrates against `python -m harp.listener`.
  Provider-agnostic; read per chunk so dashboard changes are instant. 0 = off.
- **(2) Filter-session VAD/noise** threaded through the provider abstraction:
  `SessionConfig` gained `vad_threshold` / `vad_silence_ms` / `noise_reduction`;
  [openai.py](harp/voice/openai.py) maps them onto server-VAD + input
  noise-reduction; [gemini.py](harp/voice/gemini.py) maps threshold→start/end
  sensitivity + silence (guarded/best-effort — google-genai types vary; noise
  reduction is OpenAI-only). Applied at the next session open.
- **Config + live tuning** ([config.py](harp/config.py)): `FilterAgentSettings`
  gained the four knobs (defaults in [harp.yaml](harp.yaml)); new mutable
  `FilterTuning` (validates + clamps every value in `apply`) is shared by the
  bridge and dashboard. `build_filter_config(provider, tuning)` stamps the VAD
  knobs onto each new filter session.
- **Dashboard knobs** ([dashboard/server.py](harp/dashboard/server.py) + static):
  a "Filter tuning" panel with sliders (loudness gate, VAD threshold, VAD silence)
  + a noise-reduction select. Confirmation-based like the mute button: a change
  sends `SetFilterTuning`, the server validates via `FilterTuning.apply` and
  broadcasts `FilterTuningChanged` (new event) so every tab syncs; a fresh tab is
  seeded once. Panel is hidden unless two-agent mode is on (server only wires the
  callables then). The loudness gate is instant; VAD/noise apply next conversation.
- **Also this session:** the listener now **retries on mic-open failure instead of
  crashing the whole agent** ([listener.py](harp/listener/listener.py)) — a Windows
  PortAudioError at startup (Intel Smart Sound mic array + blocked OS mic access)
  had taken the whole app down. Diagnosed as an OS-level mic-access block (every
  device failed to open on every host API, even at native rate); the mics here also
  reject a 16 kHz open (native 44.1/48 kHz) — a native-rate-and-resample fix in
  audio_io is the next audio robustness step, still to do.
- Tests: loudness gate (5), gate wired into the mic pump, `build_filter_config`
  → OpenAI VAD mapping (2), `FilterTuning` validate/clamp (2), `FilterAgentSettings`
  config, dashboard `SetFilterTuning` apply+broadcast + bad-input→error (2),
  listener mic-retry (1). **Full suite: 178 pass.** NOT yet verified live — the
  knobs need calibration against the real mic/room (that's the tuning loop the
  dashboard exists for).

### 2026-07-06 — Status voice wired (boot / connectivity / standby / error / shutdown)

Completed the remaining half of PLAN phase 2's orchestrator: HARP now speaks
canned status lines at every transition, so a live run is legible by ear (you
HEAR boot, connectivity, session-end, errors) instead of being silent about its
own state. This was step 1 of the user's ordered plan (status voice → interaction
logger → Gemini memory).

- **`StatusVoice` player** ([orchestrator/status_voice.py](harp/orchestrator/status_voice.py)):
  loads `assets/status_voice/manifest.json` once, resolves a stable id (+lang,
  falling back to `en`) → clip path, and plays it. Playback is **serialized** (an
  asyncio lock — two quick transitions never talk over each other) and
  **fail-safe by design**: an unknown id, a missing file, a broken manifest, or a
  machine with no audio device is always a logged no-op, never an exception that
  could take the supervisor down. The audio backend is injected (`sink`), so it's
  unit-tested without a sound card; the default sink lazily imports
  sounddevice/numpy and plays via `sd.play`/`sd.wait`.
- **Orchestrator narration** ([orchestrator.py](harp/orchestrator/orchestrator.py)):
  `StatusVoice` and a connectivity probe are **injected** (both optional → the
  orchestrator behaves exactly as before when absent, which every existing
  bus-driven test relies on; no new bus events). Mapping: boot → `starting_up`,
  then (probe) `connection_established`/`no_internet`; normal end → `going_standby`;
  non-fatal error → `_error_line(where)` (`mic_problem` / `connection_lost` /
  `error_recoverable`); fatal → `error_fatal`; shutdown → `shutting_down`. The
  error/shutdown paths close the session with `narrate=False` so they don't
  double-announce a standby cue over their own line.
- **Real connectivity probe** ([app.py](harp/app.py) `_internet_reachable`): a TCP
  connect to `8.8.8.8:53` (real reachability — DNS+routing+handshake, not "is a
  cable in"), run off-thread by the orchestrator so a dead network never stalls
  the loop. Only wired when narration is on (its sole consumer is that boot line).
- **Config/wiring:** new `StatusVoiceSettings(enabled, lang)` in
  [config.py](harp/config.py) + a `status_voice:` block in
  [harp.yaml](harp.yaml); app.py builds the player (or None) and injects it.
- Tests: [tests/test_status_voice.py](tests/test_status_voice.py) (7: id→clip
  resolution, unknown-id/missing-file/missing-manifest/dead-sink no-ops, lang
  fallback, non-overlap) + 6 orchestrator narration cases in
  [tests/test_orchestrator.py](tests/test_orchestrator.py). **Full suite: 165 pass.**
- **⚠ The clips themselves aren't in the repo yet — status voice is SILENT until
  they're added.** `.gitignore` had a blanket `*.wav` that silently excluded the
  Kokoro-rendered clips; only `manifest.json` was ever committed, and the WAVs
  lived only on the old Linux machine. Added a `.gitignore` exception
  (`!assets/status_voice/**/*.wav`) and verified a dropped-in clip now shows as
  tracked (`??`), while stray recordings under `audio/` stay ignored. **Action:
  drop the 14 `assets/status_voice/en/*.wav` files in (user is providing them) and
  `git add` — then it speaks.** Everything else runs normally meanwhile (the
  no-op path). Not verified live here (no clips + non-interactive shell); the
  wiring/resolution is verified against the real manifest and the full suite.

### 2026-07-06 — Two-agent noise/intent filter (experimental, opt-in)

Tackling the user's biggest real-world problem — noise and misread intent in a
loud hall — with a **two realtime-agent** pipeline in front of the responder.
Design decisions taken with the user: **text relay** (not audio) and
**half-duplex** (no barge-in). All of it is **opt-in behind a flag**
(`filter_agent.enabled` in harp.yaml, default **false**); with the flag off the
single-agent `VoiceBridge` runs exactly as before, which is why the whole
existing suite is untouched.

- **Shape.** `mic ─audio─▶ FilterAgent ─clean text─▶ VoiceBridge(responder) ─▶
  speaker`. Only the filter holds a microphone; the responder is **mic-less** and
  gets "silence, then a clean message", so it can never hear the room or itself.
- **Agent 1 — the filter** ([harp/voice/filter_agent.py](harp/voice/filter_agent.py)).
  Native audio in; it relays ONLY what the visitor means to say to HARP and drops
  background chatter / crowd noise / the assistant's own voice. Persona in
  [prompts/filter_instructions.md](prompts/filter_instructions.md): output the
  intended message, or the sentinel `[[ignore]]` for anything else. We read its
  **transcript** (`AgentTranscript`) as the relayed text — so **no provider
  change was needed**; the model's thrown-away audio-out is the prototype's cost
  (a text-output modality is the obvious later latency/cost optimization).
  `clean_relay` strips the sentinel and any punctuation-only remainder so
  `[[ignore]].` can't leak a fake turn. Feedback comes in as `CONTEXT:` notes the
  persona absorbs, so short follow-ups ("yes", "how much") stay interpretable.
  The half-duplex mic gate reuses the bridge's silence-substitution trick.
- **Agent 2 — the responder** = today's [bridge.py](harp/voice/bridge.py),
  gaining an optional **text-driven, mic-less mode** (`text_inbox`): when a queue
  is given it opens no mic and forwards each relayed message with `send_text`.
  Tools, RAG, `end_session`, transcripts, speaker, barge-in-clear are all
  unchanged. `run()` was refactored onto an `AsyncExitStack` so the mic is only
  opened in mic mode (existing mic-mode tests still pass verbatim).
- **Coordinator** ([harp/voice/two_agent.py](harp/voice/two_agent.py),
  `TwoAgentBridge`) exposes the **same `run(context)` interface** as VoiceBridge,
  so the orchestrator and app.py drive it interchangeably — app.py just builds
  this instead of a plain bridge when the flag is on. Two wires: a relay becomes
  a bus `UserSaid` (so the dashboard shows the filtered user turn) **and** goes to
  the responder's inbox; the responder's finished reply is fed back to the filter
  as context. Half-duplex: the filter's mic is muted while the responder is
  speaking (tracked from `AgentSaid` deltas) plus a `response_tail_seconds` tail,
  with a 12 s safety cap so a dropped reply can't mute the mic forever. Any
  external push-to-talk gate is AND-ed in.
- **Config** ([config.py](harp/config.py)): `FilterAgentSettings`
  (`enabled`/`provider`/`response_tail_seconds`), `build_filter_config` (responder
  defaults + filter persona + no tools), `load_filter_persona`, `IGNORE_SENTINEL`.
  New `filter_agent:` block in [harp.yaml](harp.yaml).
- Tests: [tests/test_filter_agent.py](tests/test_filter_agent.py) (8),
  [tests/test_two_agent.py](tests/test_two_agent.py) (4), +1 text-driven-mode case
  in [tests/test_bridge.py](tests/test_bridge.py). **Full suite: 151 pass.**
  **NOT verified live** — needs API keys + a real mic (ideally a loud room). Known
  costs to confirm live: a second live session, ~1–2 s extra latency, and no
  barge-in. Turn it on with `filter_agent.enabled: true` and watch the dashboard:
  the transcript should show only *intended* user turns, background chatter
  dropped. Push-to-talk stays the guaranteed fallback.

### 2026-07-05 — Urdu transcribed in Latin script + the transcription-prompt leak

Two linked fixes to how spoken Urdu is turned into text.

- **The transcription prompt was leaking into the transcript.** During live
  testing the dashboard showed the *user* saying the transcriber's own priming
  sentence ("The speaker uses only English or Urdu. Transcribe Urdu in the …
  script …"). Cause: the OpenAI Realtime input transcriber
  (`gpt-4o-mini-transcribe`) is primed with a `prompt` to steer script, and
  Whisper-family models **regurgitate that prompt verbatim** when the server VAD
  commits a turn of silence / a breath / crowd noise. That echo flowed out as
  `UserTranscript` → `UserSaid` → the dashboard. [openai.py](harp/voice/openai.py)
  now withholds a user transcript **only while it's still a prefix of the prompt**
  and streams it the instant it diverges (genuine speech diverges within a word
  or two, so there's no added lag); a turn that never diverges is dropped. Guard
  is pure/normalized so a delta splitting a word mid-token still reads as a
  prefix. Tests: [tests/test_transcript_echo.py](tests/test_transcript_echo.py) (4).
- **Urdu is now transcribed in Latin (Roman) script, not Perso-Arabic.** The
  wake words in [harp.yaml](harp.yaml) are already romanized (`salam`, `assalam`,
  `laila`), so Perso-Arabic output could never match them — Latin is what the
  matcher expects, and it's easier to read on the dashboard. Flipped both
  transcribers: the OpenAI prompt now says "romanize spoken Urdu into Latin
  letters" (overridable via `OPENAI_TRANSCRIBE_PROMPT`), and the local wake-word
  Whisper ([transcriber.py](harp/listener/transcriber.py)) gained a romanized
  `initial_prompt` bias (overridable via `HARP_WHISPER_PROMPT`, holds no wake
  word so an echo can't false-wake). Raw Whisper romanization via `initial_prompt`
  is only a nudge — **needs a live check** that spoken Urdu actually comes out in
  Latin.
- Full suite: 139 pass. The OpenAI echo path is unit-tested; the romanized-Urdu
  output (both providers) is the user's live verification.

### 2026-07-05 — Push-to-talk (per-session, on-demand) + agent-driven end_session

Addressed the biggest real-world blocker for the expo deployment: an always-on
realtime model in a loud hall false-wakes on crowd noise and has its turn-taking
wrecked by background speech. Added an **on-demand hold-to-talk session** and
gave the model a way to hang up on itself.

- **Push-to-talk is a per-session mode, not a global switch.**
  [harp/interaction/push_to_talk.py](harp/interaction/push_to_talk.py)
  (`PushToTalk`) runs **alongside** the always-on listener — HARP still wakes
  hands-free. Pressing the talk key **while idle (STANDBY)** publishes
  `WakeRequested(reason="button")` AND marks that session push-to-talk; for its
  duration the mic is gated. A session woken hands-free (wave / wake word) is
  never gated. It tracks `StateChanged` to know when it's idle and to clear the
  gate on return to STANDBY, so **when the session ends (face-absence, the
  agent's own end_session, or an error) push-to-talk goes inactive and the
  listener/wave resume automatically.** Exposes `mic_open` = `(not this-is-a-PTT-
  session) or key-held`, the gate the bridge consults. Keyboard backend is
  **pynput** (global listener on its own thread), injected via `listener_factory`
  so the class is pynput-agnostic and unit-tested by driving `press()`/`release()`
  (no display, no keypresses); pynput is imported lazily so importing the module
  never needs it; a listener that can't start logs + disables PTT instead of
  crashing.
- **Mic gating lives in the voice bridge, not the OS** (cross-platform, unlike
  the Linux-only `pactl` mic-mute). [bridge.py](harp/voice/bridge.py) gained an
  injected `mic_gate: () -> bool`; `_pump_mic` sends `_mic_payload(pcm)` — real
  mic audio when the gate is open (or when ungated, the default), else **same-
  length digital silence**. Silence rather than nothing keeps the stream
  continuous so the provider's own VAD still sees the trailing quiet and ends
  the turn — with no room noise ever reaching it.
- **The model can end its own session.** New
  [harp/interaction/session_tools.py](harp/interaction/session_tools.py)
  declares an `end_session` tool (per-provider, same shape as
  knowledge/tools.py); calling it publishes `EndOfInteractionDetected` — the
  exact event the face-absence end rule uses — so the orchestrator closes the
  session (ACTIVE → STANDBY, InteractionEnded) through the existing path. The
  model calls it when the visitor says goodbye / asks to stop. app.py composes
  the tool declarations (`knowledge + session`) and a `dispatch` that routes
  `end_session` to the bus and everything else to knowledge retrieval.
- **Wiring** ([app.py](harp/app.py)): the listener and PTT now both run (no
  longer mutually exclusive); `mic_gate=lambda: ptt.mic_open`. New
  `PushToTalkSettings(enabled=False, key="space")` in [config.py](harp/config.py);
  the repo [harp.yaml](harp.yaml) sets `enabled: true` (arms the key — HARP still
  boots hands-free). PLAN.md's vision list gained the mode.
- Tests: [tests/test_push_to_talk.py](tests/test_push_to_talk.py) (7: press-while-
  idle wakes+gates, session-end returns to hands-free, hands-free sessions
  ungated, press-before-standby doesn't wake, auto-repeat wakes once, cancel
  stops the listener, failing listener disables PTT),
  [tests/test_session_tools.py](tests/test_session_tools.py) (3), +2 mic-gate
  cases in [tests/test_bridge.py](tests/test_bridge.py), +1 config case. **Full
  suite: 135 pass.** NOT yet run to a live conversation here (needs a mic + API
  key + real keypresses / a spoken "goodbye") — that's the user's check:
  `uv run python -m harp`, hold space + talk, and try telling it to close.
- Dep added: `pynput`. Known refinement (as with the mic-gate's reliance on
  provider VAD): `end_session` tears down immediately, which can clip a trailing
  spoken "goodbye" — a graceful drain (finish the current turn first) is a later
  polish; the tool description tells the model to say goodbye *before* calling it.

### 2026-07-05 — Multi-face, face-ID presence, auto-end, wave→wake, whisper offline-load

Closed the "a wake opens a conversation but nothing ends one" gap and made the
vision + wake paths actually work end to end.

- **Face-ID now sees everyone, and doubles as presence.**
  [face_id.py](harp/vision/face_id.py) identifies *every* face per pass (not
  just the largest): publishes a `PersonIdentified` per newcomer (quiet for
  people already in frame), keeps `current` = the most prominent face for the
  voice bridge's "who am I talking to" line, and draws one labelled box per
  face. It also publishes `PresenceChanged(present, count)` on change, so
  face-ID *is* the presence signal — no separate detector needed. The overlay
  seam ([frames.py](harp/vision/frames.py)) now accepts a list of boxes per
  provider (a single `Overlay`/`None` still works).
- **Auto-end rule implemented.** [interaction/end_rules.py](harp/interaction/end_rules.py):
  no face for `absence_timeout` seconds (harp.yaml `interaction.absence_timeout_seconds`,
  default 10) during an ACTIVE session → `EndOfInteractionDetected` →
  orchestrator closes ACTIVE→STANDBY. A returning face resets the countdown, so
  brief detection dropouts don't cut people off.
  - **Bug found + fixed (the session that wouldn't close):** the bus doesn't
    replay, and face-ID only publishes presence on *change*, so a session woken
    with nobody in frame (voice/loud-sound wake, or a poor camera angle) never
    got an "absent" event — the monitor kept its optimistic "present" default
    and never armed. It now seeds presence at `InteractionStarted` from an
    injected `is_present` getter (wired to `face_id.current`), the same way the
    dashboard seeds mic-mute state on a fresh connection.
- **Wave → wake wired.** [triggers/engine.py](harp/triggers/engine.py) turns a
  wave (`GestureDetected(kind="wave")`) into a `WakeRequested`, so a wave opens
  a session (the startup banner already promised it). The orchestrator still
  only honors wakes while STANDBY, and the gesture recognizer already debounces,
  so the engine is a thin translation.
- **Whisper "stuck on transcribing…" debugged — not a download.** The model was
  fully cached; faster-whisper/huggingface_hub re-validates the cached files
  with a network HEAD request on *every* load, which stalled on a flaky
  connection. [transcriber.py](harp/listener/transcriber.py) now loads
  offline-first (`local_files_only=True`), downloading only on a genuine cache
  miss — first-phrase load dropped to <1 s. (The earlier "nothing on the wake
  transcriber" was simply the one-time model download not finishing before
  Ctrl+C.)
- **People:** registered [people/usman-asad/](people/usman-asad/) (developer;
  4 photos enrolled) and fixed a Windows cp1252 `UnicodeEncodeError` in
  [scripts/enroll_people.py](scripts/enroll_people.py) (a `→` in the summary
  print crashed *after* saving).
- Wiring: `end_rules` + `triggers` added to app.py runners; new `interaction:`
  section in [harp.yaml](harp.yaml). Tests added (`test_end_rules`,
  `test_triggers`, `test_transcriber`, multi-face + presence cases in
  `test_face_id`, interaction config). **Full suite: 122 pass.**

### 2026-07-05 — `python -m harp` is now the full agent (unified entry point)

- **`uv run python -m harp` launches the full supervised agent by default** —
  orchestrator + real voice session (VoiceBridge + search_knowledge) + always-on
  wake listener (threshold-based session start from `harp.yaml`) + camera /
  gestures / face-ID + dashboard, all on one bus. Previously this entry point
  ran only the *bare* voice core and the full stack lived behind
  `python -m harp.app`; the user wanted their known entry point to "incorporate
  it all," so [__main__.py](harp/__main__.py) now delegates to
  `harp.app.run_app` (the composition root, unchanged). `python -m harp.app`
  still works and runs the same thing.
- **The bare voice core is preserved behind `--voice-only`**
  (`python -m harp --voice-only [--provider ...]`) — mic + speaker +
  search_knowledge, no bus/orchestrator/dashboard/vision — for a fast check of
  the provider/audio path. `run_app` is imported *inside* that branch's negative
  path (not at module top) so `--voice-only` stays a lightweight fallback that
  doesn't import or depend on the heavy vision stack (cv2/insightface/mediapipe).
- Verified on this (Windows) machine: 105 tests still pass; `python -m harp`
  boots through the full stack (config → camera → face-ID → gesture model
  download → dashboard/orchestrator). NOT yet run to a live conversation here
  (needs a wake + API key round-trip) — that's the user's check.
- **Windows caveat surfaced:** [audio_control.py](harp/audio_control.py) is
  `pactl`-based (Linux/PulseAudio), so the dashboard's **mic-mute button won't
  work on Windows** — but both call sites catch the failure (→ `ErrorRaised`),
  so it's non-fatal; the rest of the agent runs. A cross-platform mute is a
  good next step (see below).

### 2026-07-05 — Data retrieval in the bare `python -m harp` runner

- The bare voice core (`python -m harp` → [harp/voice/session.py](harp/voice/session.py))
  advertised no tools and only *printed* ToolCalls to stderr — so the model
  could never actually look anything up. It now retrieves from `data/` like
  the web-realtime sandbox and harp.app do: the runner gained an injected
  `tool_dispatch`, runs the requested tool on a ToolCall, and returns the
  result via `respond_tool` (a missing handler or a failing tool degrades into
  an `{"error": ...}` the model apologizes for, never a dead session — same
  contract as the bridge). [harp/__main__.py](harp/__main__.py) is the
  composition root: it sets `cfg.tools = knowledge_tools.declarations(provider)`
  and passes `knowledge_tools.dispatch`, keeping session.py provider- and
  corpus-agnostic. New `knowledge_tools.index_size()` warms the index and feeds
  a startup line (`43 chunks indexed from data/`).
- Restructured `run()` off `asyncio.TaskGroup`: the mic now runs as a side task
  that's cancelled when the provider closes its event stream, so `run()` returns
  on stream-end instead of both tasks blocking forever. Also made the
  provider/mic/speaker injectable (same shape as the bridge) so the runner is
  testable without hardware.
- Tests: [tests/test_session.py](tests/test_session.py) (4 — tool round-trip,
  tool failure → error payload, no-dispatcher still closes the call, run()
  returns on stream-end). 105 pass repo-wide. Verified the wiring live-ish
  without a mic/key: 43 chunks index from the real `data/`, and
  `dispatch("search_knowledge", {"query": "venue location"})` returns the
  Venue chunks. NOT yet verified with a real mic + API key end to end — that's
  the user's check (`uv run python -m harp`, ask an expo question, watch for the
  `[tool search_knowledge(...) -> N result(s)]` line on stderr).
- Note: `python -m harp` stays the *bare* voice core (no orchestrator, bus, or
  dashboard) by design — this only closes the retrieval gap; harp.app remains
  the full supervised agent.

### 2026-07-05 — Face-ID wired + voice bridge: the supervised agent talks

Two chunks, both user-directed (enrollment had been verified on real photos
via preview_face_id.py the session before).

- **Face-ID wiring** ([harp/vision/face_id.py](harp/vision/face_id.py)
  rewritten): unknown-face upsert flipped to **report-only**
  (`person_id="unknown"`, nothing persisted — the decision from 2026-07-02);
  `identify_current()` replaced by a **continuous slow loop** (`run()`,
  1 pass / 1.5 s, InsightFace call moved off-thread) that publishes
  `PersonIdentified` only when who-is-in-frame changes and clears on an empty
  frame so a returning person re-publishes; added `current` (for the voice
  bridge's identity line) and `overlay()` — "name similarity" + face box on
  the dashboard camera view, same seam gestures use. app.py builds
  `MemoryStore(PEOPLE_STORE)` + `FaceID` and runs the loop alongside gestures.
  Tests rewritten: [tests/test_face_id.py](tests/test_face_id.py) (9).
- **Voice bridge** ([harp/voice/bridge.py](harp/voice/bridge.py), new):
  `VoiceBridge.run(context)` opens provider + mic + speaker, sends the wake
  context and the composed identity line at open, then pumps: mic →
  `send_audio`; VoiceEvents → bus (`UserSaid`/`AgentSaid` with final markers,
  `ToolRequested`/`ToolCompleted` around the injected dispatcher with the
  result returned via `respond_tool`, `ProviderError` → `ErrorRaised`),
  `AudioOut` → speaker, `Interrupted` → `speaker.clear()`. The orchestrator's
  `_open_session`/`_close_session` seams are now real: session task started
  on open, cancelled on close; a bridge that returns by itself (provider
  closed the stream) publishes `EndOfInteractionDetected` → STANDBY, so
  ACTIVE is no longer a dead end; a crash → `ErrorRaised(voice.session)` →
  the existing backoff path. Bridge is optional/injected, so the orchestrator
  without one behaves exactly as before (old tests unchanged and still pass).
- **web-realtime consolidation** (the sandbox-proven OpenAI settings, per
  PLAN's "Open" note): default OpenAI model is now **gpt-realtime-2** in
  config.py (matching `.env`'s `REALTIME_MODEL`); openai.py emits final
  transcript markers from the `...transcription.completed` /
  `...transcript.done` events (the sandbox handled these; the dashboard uses
  them to close turns) and sets `tool_choice: "auto"` when tools are present;
  and the **search_knowledge** tool went from sandbox to product:
  [harp/knowledge/retriever.py](harp/knowledge/retriever.py) is a faithful
  BM25 port of knowledge.js (heading-chunked data/*.md, English + Urdu-script
  tokenization, same result shape), [harp/knowledge/tools.py](harp/knowledge/tools.py)
  declares it per provider (OpenAI flat function format; Gemini
  `function_declarations` dicts — validated to coerce into `types.Tool` with
  the installed google-genai) and dispatches with error-payloads-not-raises.
  Tool description kept the proven levers but was reworded corpus-agnostic
  (PLAN.md locks "nothing hardcoded to a specific corpus").
- Verified without hardware: 101 tests pass (+24 new across
  [tests/test_bridge.py](tests/test_bridge.py),
  [tests/test_knowledge.py](tests/test_knowledge.py), orchestrator-with-
  bridge cases, rewritten face-ID suite); retriever probed against the real
  data/ corpus (43 chunks; "venue location" → about.md/Venue etc.). NOT yet
  verified live end-to-end — that's the user's next step (see Suggested next
  steps). Known gaps: no automatic end-of-interaction (sessions end only via
  provider close / error / Ctrl+C), and any provider `error` event currently
  tears the session down via the orchestrator's error path — if live testing
  shows benign errors (e.g. cancellation races on barge-in) killing sessions,
  soften that in the bridge.

### 2026-07-02 — Memory store + face matcher + enrollment (face-ID unblocked)

- Design decided with the user (options discussed, all recommendations
  accepted): people are fed in via a **folder convention + one-shot script**
  (not live-webcam or dashboard-UI enrollment); **unknown faces are reported
  but never stored** — auto-remembering strangers waits until real
  conversations exist, so face_id.py's current upsert-on-unknown is to be
  flipped to report-only during wiring; face-ID will run as a **continuous
  slow loop** (~1 check / 1–2 s; InsightFace CPU cost is a few hundred ms)
  rather than only at interaction start, so the dashboard shows who's in
  frame and later proactive triggers get their "known person reappeared"
  signal.
- [harp/memory/store.py](harp/memory/store.py): one human-editable JSON per
  person under `.harp/memory/people/` (gitignored) — id, name, role,
  model-facing notes, embeddings, summaries. Two lifecycles in one record:
  enrollment fields are replaced wholesale on re-enrollment; summaries
  (interaction history) accumulate and survive it. Atomic writes
  (tmp + rename). Slug ids with collision suffixes; `people()` for the
  matcher to scan.
- [harp/memory/matcher.py](harp/memory/matcher.py): brute-force cosine over
  every stored embedding (normed ArcFace vectors → dot product), best match
  wins above `DEFAULT_THRESHOLD = 0.4`; returns the similarity as confidence
  either way. A vector index would be overhead at dozens-of-people scale.
- [scripts/enroll_people.py](scripts/enroll_people.py): scans
  `people/<person-id>/` (photos + `info.yaml`; convention documented in
  [people/README.md](people/README.md)), embeds each photo with buffalo_l,
  skips unreadable/no-face/multi-face photos with per-photo messages,
  upserts into the store. `--only <id>` for one person. `people/*` is
  gitignored (README excepted) — real people's photos/fingerprints stay out
  of git. New config constants `PEOPLE_DIR` / `PEOPLE_STORE` in
  [harp/config.py](harp/config.py).
- [scripts/preview_face_id.py](scripts/preview_face_id.py) upgraded from
  detection-only: it now matches each detected face against the store and
  labels the saved frame with name + similarity (or "unknown") — this is the
  user's verification loop for enrollment before anything is wired into the
  live app.
- Tests: [tests/test_store.py](tests/test_store.py) (6),
  [tests/test_matcher.py](tests/test_matcher.py) (5) — pure tmp_path, no
  models. Interaction-history *content* (logger/summarizer) still needs the
  voice bridge; the store's summaries field and `add_summary` are ready for it.
- NOT done yet (next chunk, after the user verifies enrollment on real
  photos): face_id.py report-only change + continuous loop + app.py wiring +
  name overlay on the dashboard camera view.

### 2026-07-02 — Gesture overlay on the dashboard camera view

- The camera view now shows what the gesture recognizer sees: a box around
  the detected hand plus the classified gesture name, drawn live over
  `/camera.jpg`. Fills the overlay seam frames.py's docstring reserved when
  the camera view was built.
- Mechanism: [harp/vision/frames.py](harp/vision/frames.py) gained an
  `Overlay` dataclass (label + box in **normalized** [0,1] coords, so
  providers never need the snapshot frame's pixel size) and `jpeg_snapshot`
  takes optional overlay callables it composites before encoding — drawing
  happens on the copy `Camera.latest()` returns, never the capture buffer.
  [gestures.py](harp/vision/gestures.py) records each pass's sighting (top
  gesture + hand bbox from MediaPipe's 21 landmarks) and exposes it as
  `overlay()` — None when no hand, cleared when the hand leaves frame, and
  stale after 1s (`_OVERLAY_TTL_S`) so a stopped recognizer doesn't leave a
  frozen box on screen. app.py now builds ONE `GestureRecognizer` and gives
  the same instance to both the runner and the snapshot partial. No frontend
  change — the overlay is baked into the JPEG the page already polls.
- Face-ID deliberately NOT tied in: its identity half still needs
  memory/matcher (PLAN phase 6), and running InsightFace continuously just to
  draw a face box duplicates what presence (phase 3) will do properly.
- Tests: +3 in [tests/test_gestures.py](tests/test_gestures.py) (overlay
  reflects last sighting, goes stale, clears when hand leaves; fake now
  carries `hand_landmarks`), +2 in [tests/test_frames.py](tests/test_frames.py)
  (overlay changes the JPEG, None-provider draws nothing). 67 pass repo-wide.

### 2026-07-02 — Mic-mute button on the dashboard

- User wants "literally mute the default mic input on the PC" — not an
  app-level flag, so it's `pactl set-source-mute @DEFAULT_SOURCE@`
  ([harp/audio_control.py](harp/audio_control.py)), which mutes the physical
  input at the OS mixer. Verified empirically this actually zeroes what
  `sounddevice` captures on this machine (max abs sample: 0 while muted, 433
  unmuted) — the ALSA "default" device routes through PipeWire, so a
  PulseAudio/PipeWire-level mute is enough, no per-library plumbing needed.
  This also sidesteps the "which subsystem" question from the design
  discussion (listener vs. future voice-bridge mic vs. speaker): OS-level
  mute affects every consumer of the mic uniformly, including subsystems
  that don't exist yet.
- Gave the dashboard a second deliberate exception to "observe-only" (the
  first was `/camera.jpg`, see above): `/ws` now also accepts one incoming
  message shape from the browser, `{"type": "SetMicMuted", "muted": bool}`.
  [harp/dashboard/server.py](harp/dashboard/server.py) validates it strictly
  (unrecognized shapes are silently dropped — this is not a general command
  channel), calls an injected `set_mic_muted` callable off-thread
  (`asyncio.to_thread`, since `pactl` is a blocking subprocess call), and
  publishes the new `MicMuteChanged` bus event
  ([core/events.py](harp/core/events.py)) on success so every connected
  tab/phone — not just the one that clicked — stays in sync. A fresh
  connection gets the current state once via an injected `get_mic_muted`,
  sent directly to that connection (the bus doesn't replay history, same
  reasoning as the existing forward/receive task split). Both callables are
  optional (default None), following the same injectable-callable pattern as
  `snapshot` for the camera, so `dashboard/` still doesn't hard-depend on
  `audio_control` and stays testable with fakes.
  [harp/app.py](harp/app.py) wires the real `audio_control.set_mic_muted`/
  `get_mic_muted` in.
  A `pactl` failure (missing binary, no default source) is caught and turned
  into `ErrorRaised(where="dashboard.mic_mute", ...)` rather than crashing
  the connection — consistent with PLAN.md's "narrates its own problems"
  philosophy.
- Frontend: a large button above the event panels
  ([static/index.html](harp/dashboard/static/index.html)); confirmation-based,
  not optimistic — it only flips label/color once the server echoes back
  `MicMuteChanged`, so it stays correct even if muted from a different
  tab/phone. `ws` was hoisted out of `connect()` to module scope so the click
  handler can send on it.
- PLAN.md's locked "dashboard is not part of the end-user flow" decision now
  has one documented exception (mic-mute) — noted inline rather than rewritten.
- Explicit user call: no automated tests for this chunk (verify live instead
  of writing `tests/test_audio_control.py` / dashboard write-path tests) —
  noting this is a deliberate exception to this repo's normal "test
  everything" convention, not a new default.

### 2026-07-02 — Dashboard reachable over the LAN (phone/other PC)

- The dashboard's `websockets` server was hard-bound to `host="127.0.0.1"` in
  both its default (`server.py`) and its one caller (`app.py`) — unreachable
  from any other device by construction, before firewall/network even come
  into it. Frontend was already fine: `app.js` builds the WebSocket URL from
  `location.host`, not a hard-coded address.
- New `dashboard:` section in [harp.yaml](harp.yaml) (`bind: localhost|network`,
  `port`), loaded via a new `DashboardSettings` in
  [harp/config.py](harp/config.py). `config.dashboard_bind_host()` maps
  `localhost` → `127.0.0.1` (old behavior, still the dataclass default) and
  `network` → `0.0.0.0`; an unrecognized value warns and falls back to
  localhost-only rather than accidentally exposing the dashboard.
  [harp/app.py](harp/app.py) now passes this through to `serve_dashboard(...)`
  instead of relying on its defaults.
- When bound to the network, app.py prints the actual LAN IP to browse to
  (`_lan_ip()`: a UDP "connect" to a public IP, which only triggers a routing
  lookup and sends no packet, so it works offline too — picked over parsing
  `ip addr` because it naturally returns the real Wi-Fi/Ethernet address
  instead of a Docker/VirtualBox bridge IP when several interfaces exist).
- Checked this machine's firewall (`ufw`) — inactive, so no rule changes were
  needed here; noted as something to check if the dashboard is unreachable on
  a different machine.
- Tests: +4 in [tests/test_config.py](tests/test_config.py) (defaults,
  override, and both branches of `dashboard_bind_host`). 62 pass repo-wide.

### 2026-07-02 — Camera view on the dashboard (/camera.jpg)

- The dashboard now shows what HARP sees. Mechanism decided with the user
  after frames-as-bus-events was rejected (events.py's "keep them small"
  rule): the dashboard server takes an optional `snapshot() -> jpeg bytes |
  None` callable and serves it **read-only** at `/camera.jpg`
  (`Cache-Control: no-store`; 404 when no camera is attached, 503 before the
  first frame). The page polls it ~4×/s with a cache-buster and falls back to
  a "no camera feed" note. This is the one deliberate exception to
  "dashboard = pure bus observer" — documented in server.py's docstring.
- The callable lives in [harp/vision/frames.py](harp/vision/frames.py)
  (`jpeg_snapshot(camera)`), and app.py — the composition root — is what
  hands it to the dashboard, so dashboard/ still imports nothing from
  vision/. Overlay intent (user's direction): services that process frames
  will contribute overlays drawn over this view; none produce overlay data
  yet, so the seam waits in frames.py.
- Tests: +2 in [tests/test_dashboard.py](tests/test_dashboard.py) (route
  serves the snapshot + ignores the `?t=` cache-buster; 404 without a
  camera), +2 in [tests/test_frames.py](tests/test_frames.py) (JPEG magic,
  None before first frame). 59 pass. Live-verified: `/camera.jpg` returned a
  real ~15 KB frame from the running app.
- Agreed next chunk: the **voice bridge** (orchestrator `_open_session` →
  real harp/voice session → UserSaid/AgentSaid on the bus).

### 2026-07-02 — Vision: camera, face-ID detection, gesture cue implemented + tested

- Built the three `harp/vision/` files as a set, verifying each against real
  hardware before moving on (real webcam, real InsightFace, real MediaPipe —
  not just mocked tests).
- `camera.py`: single shared `cv2.VideoCapture`, background-thread capture
  (`.read()` blocks — must not stall the shared asyncio loop), `latest()`
  returns a copy, reconnect-with-backoff on read failure. Verified against a
  real Logitech C310 (`scripts/preview_camera.py`); diagnosed a misleading
  first run where no frame appeared within 1s — confirmed via `ffmpeg`
  grabbing a frame in the same ~2s that it's real USB/sensor warm-up on a
  cold open, not a bug (see gotcha above), and irrelevant once the camera is
  opened once at startup.
- `face_id.py`: InsightFace `buffalo_l` (CPU) detects + embeds faces, picks
  the most prominent by bbox area, delegates identity resolution to
  memory/matcher + memory/store (both still stubs, so this raises
  `NotImplementedError` at that call for now — expected, matches the
  existing scaffold convention). Verified real detection against the webcam
  (`scripts/preview_face_id.py`): correct bbox, 512-d embedding.
- `gestures.py`: went through a real design correction. First pass used
  MediaPipe's raw `HandLandmarker` points with a hand-rolled "is the hand
  open" heuristic (distance of each fingertip from the wrist) plus a
  zig-zag x-position reversal counter to detect a literal waving motion —
  passed unit tests, but live testing showed real false-negative "closed"
  reads on an actually-open hand (the heuristic doesn't hold up well under
  the hand rotation/perspective change that happens mid-swing, and it
  required all 4 fingers to agree unanimously). Corrected, per user
  direction, to MediaPipe's own pretrained `GestureRecognizer` (canned
  categories, same model aura's `gesture_monitor.py` uses) instead of
  trying to out-heuristic a real classifier — a raised **Open_Palm** is the
  greeting cue, debounced with hold-frames + release-then-cooldown so one
  raised palm fires exactly one `GestureDetected(kind="wave")`. Verified
  live: 3 clean fires, correctly did not re-fire across a 12s continuous
  hold, correct reset when the hand left frame. This is the exact
  `GestureRecognizer(bus, camera).run()` surface app.py already wires in —
  confirmed no API change was needed on that side.
- Tests: [tests/test_camera.py](tests/test_camera.py) (5),
  [tests/test_face_id.py](tests/test_face_id.py) (5),
  [tests/test_gestures.py](tests/test_gestures.py) (6) — all fake the heavy
  dependency (`cv2.VideoCapture` / `FaceAnalysis` / `GestureRecognizer`) so
  the suite needs no camera, GPU, or model download. 55 pass total.
- Deps added: `insightface`, `onnxruntime`, `mediapipe` (see the `cv2`
  namespace-collision gotcha above — hit it once, documented the fix).
- This work overlapped in time with the orchestrator/config/dashboard/app.py
  session below; no file conflicts (disjoint files), `pyproject.toml`/
  `uv.lock` merged cleanly across both sessions' `uv add` calls.

### 2026-07-02 — Idle-time hearing visible on the dashboard (PhraseHeard)

- Live testing app.py showed the listener transcribing phrases that never
  reached the dashboard: it only published on a wake-word MATCH, so misses
  ("Hallo!" ≠ "hello") were invisible outside the terminal log.
- New bus event `PhraseHeard(text, wake_word)` (core/events.py): the listener
  now publishes every non-empty transcript, matched or not (`wake_word` = the
  matched word or None). Dashboard gained a "Heard while idle" panel showing
  time + wake/no-wake badge + text. Test added (bus-level, transcriber
  injected — still no mic/Whisper in tests): 49 pass.
- Camera feed on the dashboard was requested but NOT built this chunk: frames
  as bus events were considered and rejected (events.py's contract is "keep
  them small"; base64 JPEG blobs aren't). Mechanism TBD with the user —
  likely the dashboard serving snapshots/MJPEG from the shared Camera
  directly.

### 2026-07-02 — app.py wired: full agent + dashboard runnable

- Implemented [harp/app.py](harp/app.py) (`run_app` + a `python -m harp.app`
  entry point): one shared `Bus` feeding the orchestrator, the wake listener,
  camera + gestures, and the dashboard server, run concurrently. The dashboard
  finally shows real traffic instead of the empty-bus "nothing yet" state.
- Wiring choices: listener honors `listener.enabled` from harp.yaml;
  orchestrator gets its heartbeat interval/file from harp.yaml too
  (REPO_ROOT-anchored). A webcam that won't open logs a warning and skips the
  camera-fed subsystems instead of killing the run. Shutdown: first task to
  exit wins — orchestrator STOPPING ends the app cleanly, any other task
  finishing first is treated as a crash and re-raised. Face-ID left unwired on
  purpose: `identify_current()` calls the still-stubbed memory/matcher.
- Known dead end until the voice bridge lands: a wake (loud sound / wake word)
  moves STANDBY→ACTIVE, but with no voice session and no end-of-interaction
  rules nothing ever moves it back; restart to reset.
- Smoke-tested: app boots, camera opens, gesture model auto-downloads on first
  run (`~/.cache/mediapipe/`), orchestrator goes starting→standby, dashboard
  serves at http://127.0.0.1:8787 and streams heartbeats.
- Note for whoever is reworking gestures: this session observed
  [harp/vision/gestures.py](harp/vision/gestures.py) being rewritten
  concurrently (wave-tracker → canned Open_Palm categories);
  [tests/test_gestures.py](tests/test_gestures.py) still imports the old
  `_WaveTracker` and fails collection. app.py only relies on the stable
  `GestureRecognizer(bus, camera).run()` surface, which both versions share.
  All other tests pass (48).

### 2026-07-02 — Dashboard implemented + tested

- Built [harp/dashboard/](harp/dashboard/) end to end: `server.py` (bus →
  browser), `static/` (vanilla HTML/CSS/JS, no build step), `__main__.py`
  (`python -m harp.dashboard` runs it against a fresh, empty bus so it's
  testable standalone before anything real feeds it).
- Server reuses `websockets` (already a dependency) instead of adding
  FastAPI/aiohttp: `process_request` serves the static page over plain HTTP,
  `/ws` streams every bus event as JSON (`{type, server_ts, fields}`) tagged
  with its class name. The frontend renders known types specially (state,
  presence/identity, transcript grouped into speaker turns from streamed
  deltas, tool calls paired by id, heartbeat-with-staleness, errors) and
  falls back to a generic raw-event log for anything else — confirmed this
  works with zero dashboard changes when `WakeRequested`/
  `EndOfInteractionDetected` (added concurrently, see below) showed up.
- Found and fixed a real bug via an end-to-end smoke test (real `Orchestrator`
  + real dashboard server sharing one `Bus`, not just synthetic events): the
  per-connection handler needs `connection.wait_closed()` racing against the
  event-forwarding loop, not just relying on the next `send()` to raise on a
  dead connection — otherwise a client that disconnects while the bus is idle
  leaves its handler task (and bus subscription) stuck forever, which then
  deadlocks server shutdown (`Server.close()` waits for all handlers to
  exit). Second bug from the same smoke test: cancelling that forwarding task
  and calling `stream.aclose()` concurrently raced on the same async
  generator ("already running") — fixed by fully awaiting the cancelled task
  before closing the stream ourselves.
- Confirms the design intent from `dashboard/server.py`'s original docstring
  holds in practice: it is a pure observer (never publishes), and it's
  decoupled enough from the rest of the app to be built, tested, and proven
  end-to-end before `app.py` exists to wire it into the real agent.
- Also fixed a latent typing bug this surfaced in `core/bus.py`:
  `subscribe()` was annotated `-> AsyncIterator[Event]`, but the returned
  object is an async generator with `.aclose()`, which `AsyncIterator` doesn't
  declare. Retyped to `AsyncGenerator[Event, None]` — no behavior change.
- Tests: [tests/test_dashboard.py](tests/test_dashboard.py) (5: event
  delivery, per-client isolation, subscriber cleanup on disconnect, static
  page served, 404 on unknown path).

### 2026-07-02 — Status voice clips generated (Kokoro TTS)

- Unblocked `status_voice`'s "offline TTS" question: Kokoro-82M was found
  installed in a separate venv (`/home/mani/Repos/Latex/.venv-tts`). New
  one-shot utility [scripts/generate_status_voice.py](scripts/generate_status_voice.py)
  renders the canned status phrases and is meant to be re-run only when the
  phrase set changes — run it with that venv's interpreter, not harp's.
- Output committed as assets: `assets/status_voice/en/` — 14 lines covering
  boot (`starting_up`, `connection_established`, `ready`), wake (`listening`),
  connectivity/errors (`no_internet`, `connection_lost`, `retrying`,
  `error_recoverable`, `error_fatal`, `mic_problem`), filler (`one_moment`),
  and lifecycle (`going_standby`, `session_ended`, `shutting_down`) — plus
  `assets/status_voice/manifest.json` (id → lang → text/file/voice/duration).
  24 kHz mono 16-bit WAV, voice `af_heart`, ~1.5 MB total.
- Manifest is keyed id → lang because Kokoro has **no Urdu voice**; the open
  idea for the `ur` set is Kokoro's Hindi voice over transliterated text, or
  another engine. English only for now.
- `orchestrator/status_voice.py` is still a stub — wiring `play(line_id)` to
  these clips is the remaining half of that chunk.

### 2026-07-02 — Always-on wake listener + harp.yaml settings + liveness file

- New subsystem [harp/listener/](harp/listener/): HARP's ears while asleep. Two
  wake rules — loudness ≥ `wake_level` wakes immediately; sound ≥
  `transcribe_level` (lower) captures a phrase, transcribes it locally with
  faster-whisper (CPU int8, lazy-loaded, model auto-downloads on first use) and
  wakes if it contains a configured wake word (whole-word match, works for
  multi-word phrases and Urdu script). Publishes `WakeRequested(reason,
  context)`; `context` is model-facing text the orchestrator will deliver to
  Gemini Live / OpenAI Realtime at session open so the model knows why it woke.
  Listens only while STANDBY (releases the mic to the live session otherwise);
  standalone it listens forever, which the calibration meter uses:
  **`uv run python -m harp.listener`** (shows live level vs both thresholds and
  what Whisper heard). Pure detector logic tested with synthetic audio.
- New [harp.yaml](harp.yaml): the user-tweakable settings file (thresholds, wake
  words, whisper model size, heartbeat file/interval), loaded via
  `config.load_settings()` with per-key defaults and warnings on typo'd keys.
  Secrets stay in `.env`. Deps added: `pyyaml`, `faster-whisper`.
- `WakeRequested` and `InteractionStarted` gained a `context` field; the
  orchestrator forwards it and stores it for the upcoming voice bridge.
- Orchestrator heartbeat now also touches a liveness file (`.harp/heartbeat`,
  gitignored) so the future watchdog process can detect death/hang by mtime.
- PLAN.md "Open" section updated: web-realtime = ongoing sandbox whose proven
  OpenAI settings get consolidated into `harp/`; speech-wake no longer open.
- status_voice: decided NOT obvious (espeak-ng isn't installed; offline TTS with
  decent Urdu is the sticking point) — options presented to the user instead.
- Tests: 33 pass (bus 5, retry 4, orchestrator 12, listener 8, config 4).

### 2026-07-02 — Orchestrator skeleton implemented + tested

- Added the two bus events the orchestrator consumes but that were missing from
  the vocabulary: `WakeRequested` and `EndOfInteractionDetected`
  (core/events.py). The triggers/end-rules stubs now name them explicitly.
- Implemented `orchestrator/retry.py` (pure policy: exponential backoff capped
  at 30s; give up after 5 consecutive failures or 120s of failing) and
  `orchestrator/orchestrator.py`: boots STARTING→STANDBY, opens a session on
  `WakeRequested` (ignored unless STANDBY), closes on `EndOfInteractionDetected`
  (publishing `InteractionEnded` for the future memory summarizer), non-fatal
  errors → close session, ERROR, backoff, back to STANDBY; fatal or exhausted
  error budget → STOPPING; graceful shutdown; periodic `Heartbeat`. The error
  budget resets once a session opens successfully.
- `_open_session`/`_close_session` are still seams: right state flow + right
  events, but no real voice session yet — that bridge is the next chunk, along
  with `status_voice` narration at boot/error points.
- Tests: [tests/test_orchestrator.py](tests/test_orchestrator.py) (10, all
  bus-driven — publish events in, assert state/interaction events out) and
  [tests/test_retry.py](tests/test_retry.py) (4). 19 pass including the bus tests.

### 2026-07-01 — core/bus.py implemented + tested

- Implemented `Bus`: `subscribe()` registers a per-subscriber `asyncio.Queue`
  eagerly (so events published before the caller's first read aren't missed)
  and returns an async-generator stream over it; `publish()` fans out to every
  matching subscriber and drops the oldest queued event on a full queue rather
  than blocking (a stuck consumer can't stall the rest of the bus).
  - Caveat worth knowing: cleanup runs in the stream's `finally`, which only
    executes once the stream has actually started. Closing a subscription
    before ever reading from it won't unregister it until GC finalizes the
    generator — read at least once before closing early.
- Added `tests/test_bus.py` (5 tests, all passing) covering: basic pub/sub,
  type filtering, multiple independent subscribers, drop-oldest back-pressure
  under a slow/absent consumer, and subscriber cleanup on close.
- Added `pytest` + `pytest-asyncio` as dev dependencies (`uv add --dev`),
  `asyncio_mode = "auto"` in `pyproject.toml` so async tests need no marker.
- This unblocks the rest of PLAN.md's build order: every other subsystem can
  now be built and tested against a real bus instead of a stub.

### 2026-07-01 — Architecture skeleton

- Scaffolded the full subsystem layout from PLAN.md: `core/` (bus, events,
  state), `app.py`, and stub packages for orchestrator, presence, vision,
  knowledge, memory, interaction, triggers, dashboard. All import cleanly; bodies
  raise `NotImplementedError`, type contracts (events, `AppState`) are real.
- Design decision: subsystems never import each other — they communicate only
  through the `core` event bus, so each is independently buildable/testable.

### 2026-07-01 — Voice core + OpenAI provider

- Built the provider abstraction and the two live backends (Gemini Live, OpenAI
  Realtime) behind it; `python -m harp` runs either.
- OpenAI transcription steered to render spoken Urdu in Urdu (Perso-Arabic)
  script instead of Hindi/Devanagari, via a `prompt` (not a pinned `language`, so
  EN/Urdu switching still works). Overridable with `OPENAI_TRANSCRIBE_PROMPT`.
- Shared the realtime model/voice with the `web-realtime/` sandbox through `.env`.

### 2026-07-01 — Gemini voice spike (verified)

- Real-time voice verified working; spoken Urdu and English both sound good.
  Model `gemini-3.1-flash-live-preview`, `v1beta` channel. Now superseded by the
  voice core above; the spike file is kept only as reference.
