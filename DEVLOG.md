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

- **Real-time voice core.** `python -m harp [--provider gemini|openai]` — talk
  into the mic, HARP answers through the speakers. Bilingual EN/Urdu.
  - Provider abstraction: [harp/voice/provider.py](harp/voice/provider.py) — one
    interface (`SessionConfig`, normalized `VoiceEvent`s, `VoiceConnection`).
  - Backends: [harp/voice/gemini.py](harp/voice/gemini.py) and
    [harp/voice/openai.py](harp/voice/openai.py).
  - Runner + audio I/O: [harp/voice/session.py](harp/voice/session.py),
    [harp/voice/audio_io.py](harp/voice/audio_io.py).
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
  (faster-whisper, lazy-loaded; model downloads on first use), and wakes if a
  configured wake word is in it. Publishes `WakeRequested(reason, context)`
  where `context` is model-facing text (e.g. the transcript) the orchestrator
  passes to the live session at open. Calibrate levels with the live meter:
  `uv run python -m harp.listener`. Detector logic is pure and unit-tested
  ([tests/test_listener.py](tests/test_listener.py)).
- **User-tweakable settings file.** [harp.yaml](harp.yaml) (thresholds, wake
  words, whisper model, heartbeat) loaded by `load_settings()` in
  [harp/config.py](harp/config.py); missing file/keys fall back to defaults,
  typo'd keys warn instead of crash ([tests/test_config.py](tests/test_config.py)).
  Secrets stay in `.env`.
- **Status voice audio assets.** `assets/status_voice/en/*.wav` — 14 canned
  status lines (boot, connectivity, error narration, sleep/wake, shutdown)
  rendered offline with Kokoro TTS (voice `af_heart`, 24 kHz mono PCM), plus
  `assets/status_voice/manifest.json` mapping stable line ids → text/file/
  duration, keyed id → lang so Urdu can be added later. Generated ONCE by
  [scripts/generate_status_voice.py](scripts/generate_status_voice.py) — note
  it must run under the separate Kokoro venv:
  `/home/mani/Repos/Latex/.venv-tts/bin/python scripts/generate_status_voice.py`.
  Playback (`status_voice.play()`) is NOT wired yet — that's the remaining half.
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
  - `face_id.py` — InsightFace (`buffalo_l`, CPU) detects + embeds faces,
    picks the most prominent one by bbox area, matches against memory/store
    via memory/matcher. Now a **continuous slow loop** (`run()`, ~1 pass /
    1.5 s, detection off-thread) that publishes `PersonIdentified` only when
    who-is-in-frame *changes* (appears / different person / came back after
    absence) — not every pass. **Unknown faces are report-only**
    (`person_id="unknown"`, never stored — decision from the 2026-07-02
    memory log entry). Exposes `current` (latest identification, None when
    nobody's there) for the voice bridge's identity line, and `overlay()`
    (name + similarity + face box) for the dashboard camera view, same
    pattern as gestures. Wired into app.py.
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
- **app.py — the full supervised agent.** `uv run python -m harp.app
  [--provider gemini|openai]` wires ONE shared `Bus` into everything built so
  far — orchestrator + VoiceBridge (with the search_knowledge tool and the
  face-ID identity line composed here, in the composition root), wake
  listener (if `listener.enabled` in harp.yaml), camera + gesture recognizer
  + face-ID slow loop, dashboard at http://127.0.0.1:8787 — and runs them
  concurrently. A missing/busy webcam logs a warning and disables the
  camera-fed subsystems for that run instead of blocking the voice side.
  First task to exit wins: orchestrator reaching STOPPING is a clean
  shutdown; any other task finishing means a subsystem crashed and the crash
  is re-raised. **A wake now opens a real conversation.** Remaining caveat:
  nothing *ends* one automatically yet (interaction/end_rules is still a
  stub) — a session stays open until the provider closes it, an error closes
  it, or Ctrl+C.

**Scaffolded only (skeletons — import cleanly, but raise `NotImplementedError`):**

The full architecture from PLAN.md exists as stubs so subsystems can be built
independently. Each file has a docstring + a "To build" list. Nothing below is
functional yet, except where noted.

- **core/** — the spine everything else depends on: `bus.py` (async pub/sub,
  ✅ **implemented + unit-tested**, see [tests/test_bus.py](tests/test_bus.py)),
  `events.py` (the shared event vocabulary), `state.py` (the app state machine).
- **orchestrator/** — `orchestrator.py` + `retry.py` are ✅ implemented (see
  above), including running the voice bridge; `watchdog` and `status_voice`
  are still stubs.
- **presence/** — webcam "is anyone here" → sleep/wake.
- **vision/** — `camera.py`, `face_id.py`, and `gestures.py` are all ✅
  implemented and wired (see above).
- **knowledge/** — `retriever` + `tools` are ✅ implemented (BM25 port of the
  sandbox, see above); `indexer` (future vector store) and `web_search`
  (internet fallback) are still stubs.
- **memory/** — `store.py` + `matcher.py` are ✅ implemented + tested (JSON
  per person in `.harp/memory/people/`, cosine matcher, enrollment via
  [scripts/enroll_people.py](scripts/enroll_people.py) over `people/` — see
  the log entry); `logger` and `summarizer` are still stubs (the voice bridge
  now exists, so these are unblocked).
- **interaction/** — `end_rules` (when a conversation is over).
- **triggers/** — proactive `engine` (wave / follow-up).

**Reference only (not the product, don't build on directly):**

- [spike_gemini_voice.py](spike_gemini_voice.py) — the original throwaway spike.
- [web-realtime/](web-realtime/) — the user's sandbox for tuning OpenAI features
  and RAG/tool-calling before they land in `harp/`.

## Suggested next steps

1. **Live verification (user):** run `uv run python -m harp.app --provider
   openai`, wake it (wake word / loud sound / wave), and check: a real
   conversation happens; the transcript + tool calls appear on the dashboard;
   asking an expo question triggers `search_knowledge`; standing in frame as
   an enrolled person shows the name on the camera overlay and the model
   greets you by name at session open. Also worth confirming with
   `--provider gemini`.
2. **End-of-interaction rules** (`interaction/end_rules`) — now the biggest
   gap: sessions only end via provider close / error / Ctrl+C. PLAN's sketch:
   left frame + silent for a while (needs presence, or a first version from
   transcript silence alone).
3. `status_voice` playback (`play(line_id)` over the generated
   `assets/status_voice/` clips + manifest) + `watchdog` complete PLAN phase 2.
4. Then PLAN.md's remaining phase order: presence/sleep-wake → web-search
   fallback → memory's remaining half (`logger` + `summarizer`, now unblocked
   by the voice bridge) → proactive triggers. Build each subsystem so it runs
   and is testable **on its own** before wiring it into `app.py`.

---

## Log

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
