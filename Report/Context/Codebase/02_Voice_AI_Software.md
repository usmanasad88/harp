# HARP Voice-AI Software Brain (`harp` repository)

*A from-scratch, modular rewrite of HARP's conversational intelligence: a
continually-running bilingual voice agent that gates an expensive cloud session,
answers grounded in your documents (RAG), sees and recognises people, and remembers
who it has spoken to. Grounded in the code under `harp/harp/`.*

> **Positioning:** this runs on an internet-connected laptop and treats **robot
> motion as out of scope**. It is the "smart brain" track, distinct from the
> deployed robot in `harpcontrol`. Much of it is working; a few subsystems are
> deliberately scaffolded as testable skeletons (marked *[stub]* below).

## 1. Real-time bilingual voice — provider-agnostic

- **Full-duplex, interruptible, audio-to-audio** conversation. The end user just
  **talks** — there is no screen in the end-user flow.
- **English *and* Urdu**, verified working, including natural code-switching. The
  language lever is the **system instruction** ("mirror the user's language"), not a
  pinned language code, so the native-audio model switches languages mid-conversation.
- **One interface, two cloud back-ends** (`voice/provider.py`): **Google Gemini
  Live** (`gemini-3.1-flash-live-preview`, native audio, 90+ languages, image/video
  input, 128K context) and **OpenAI Realtime** (`gpt-realtime-2`), selectable at
  runtime (`--provider gemini|openai`). The rest of the app is written once against
  a normalized event stream (`VoiceEvent`), so RAG, vision, memory, and the UI don't
  care which vendor is live.
- The **persona** ("Laila", a warm, voice-first expo host) lives in a prompt file,
  with explicit voice-first rules (short replies, no markdown, say numbers aloud,
  yield on interruption).

## 2. The orchestrator — a supervisor, not a demo (`orchestrator/`)

The heart of the design: a **state machine** that keeps the assistant alive and
only spends money on the cloud session when someone is actually there.

- **States:** `STARTING → STANDBY ⇄ ACTIVE`, plus `ERROR` and `STOPPING`, with
  legal transitions enforced.
- **Session gating:** the costly live voice session opens on a wake condition and
  closes at end-of-interaction — it does **not** hold a 12-hour connection open.
- **Self-narrating errors + retry:** on failure it goes to `ERROR`, backs off with
  **capped exponential backoff** (≤30 s), and returns to `STANDBY`; a fatal error or
  an exhausted budget (5 consecutive failures / 120 s) triggers a clean `STOPPING`.
- **Liveness:** publishes a periodic `Heartbeat` **and** touches a heartbeat file
  (mtime = last beat) so a future watchdog process can detect a hang or crash.

## 3. Wake — an always-on local listener (`listener/`)

- Owns the microphone **while idle** and releases it to the live session on wake.
- **Two wake rules**, both tuned in `harp.yaml`: a **loudness threshold** wakes
  immediately; a lower threshold captures a phrase, transcribes it **locally** with
  **faster-whisper** (runs offline, on CPU), and wakes if a configured **wake word**
  is present (works for multi-word phrases and Urdu script).
- Passes the transcript forward as **context** the model receives at session open,
  so HARP knows *why* it woke.
- A built-in **calibration meter** (`python -m harp.listener`) shows live level vs.
  both thresholds and what Whisper heard.

## 4. Knowledge — retrieval-augmented answering (`knowledge/`)

- **`search_knowledge` is live:** a **BM25 keyword search** over the `data/*.md`
  corpus, chunked at markdown headings, with English + Urdu-script tokenization. No
  embeddings — a deliberate, right-sized choice for a small corpus.
- Exposed to **both** providers as a **tool/function call** (OpenAI GA function
  format with `tool_choice: auto`; Gemini `function_declarations`), so the model
  **retrieves before answering** questions about specific documents, then grounds
  its reply on what came back. Returns structured "no matches" / "error" payloads
  rather than raising.
- **Context-agnostic by design:** drop any documents into `data/`; nothing is
  hard-coded to one topic.
- *[stub]* `web_search` (internet fallback) and `indexer` (a future vector store,
  the reserved seam for scaling beyond keyword search).

## 5. Vision — camera, face-ID, gesture cue (`vision/`)

- **Camera** (`camera.py`): one shared `cv2.VideoCapture` captured on a background
  thread (so the async event loop never blocks), with auto-reconnect on device
  drop-out. Verified against a real webcam.
- **Face identity** (`face_id.py`): **InsightFace** (`buffalo_l`, CPU) detects and
  embeds faces (512-d ArcFace vectors), picks the most prominent one, and matches it
  against stored people. Runs as a **continuous slow loop** (~1 pass / 1.5 s) that
  publishes `PersonIdentified` **only when who-is-in-frame changes**. **Unknown faces
  are report-only** — never silently stored.
- **Gesture cue** (`gestures.py`): **MediaPipe's** pretrained `GestureRecognizer`; a
  **raised open palm**, held then released, is the proactive greeting cue (a "wave"),
  debounced with hold-frames + release-cooldown so one gesture fires exactly one
  event. Verified live.
- Both face-ID and gestures also produce **overlays** (name + similarity, or gesture
  label + hand box) drawn onto the dashboard's live camera view.

## 6. Memory — who it has met, and what was said (`memory/`)

- **Per-person store** (`store.py`): one human-editable JSON per person under
  `.harp/memory/people/` — id, name, role, model-facing notes, face embeddings, and
  accumulating conversation summaries. Atomic writes; enrollment fields are replaced
  on re-enrollment while summaries survive.
- **Matcher** (`matcher.py`): brute-force **cosine similarity** over stored ArcFace
  embeddings (right-sized for dozens of people; a vector index would be overkill),
  best match above a threshold wins, similarity returned as confidence.
- **Enrollment** by convention: drop photos + `info.yaml` into `people/<id>/` and run
  a one-shot script; real people's photos are kept out of git.
- *[stub]* `logger` + `summarizer` — logging every interaction and summarising it
  into per-person memory (now unblocked by the working voice bridge).

## 7. The voice bridge — the agent talks end-to-end (`voice/bridge.py`)

- Runs one live session (provider + mic + speaker) and **translates its events onto
  the shared bus**: `UserSaid` / `AgentSaid` (with final markers so turns close),
  `ToolRequested` / `ToolCompleted` around each tool dispatch, and provider errors
  into the orchestrator's error path.
- At session open it injects the **wake context** and a composed identity line —
  *"you are talking to &lt;name&gt;"* from face-ID — so HARP can greet a known person in
  context. This is the seam that makes RAG, face-ID, and memory reach the model.

## 8. Developer dashboard (`dashboard/`)

- A **read-only web view** of the live system (`websockets`, `http://127.0.0.1:8787`):
  state changes, the grouped **transcript**, **tool calls**, presence / identity /
  gesture events, heartbeats and errors, a **"heard while idle"** panel, and a **live
  camera view** with the vision overlays.
- **Not part of the end-user flow** (the user only ever talks) — it is an
  observability tool. One deliberate exception: a **mic-mute button** that performs a
  real OS-level mute (`pactl`), kept in sync across every open tab via the bus.
- **LAN-reachable** (configurable bind) so it can be watched from a phone.

## 9. Configuration & operational polish

- **`harp.yaml`** — user-tweakable behaviour knobs (wake thresholds, wake words,
  Whisper model size, heartbeat, dashboard bind/port). **`.env`** — secrets only.
- **Status voice:** 14 canned status lines (boot / connectivity / error / sleep /
  wake / shutdown) rendered **offline** with Kokoro TTS, so HARP can speak "starting
  up", "no internet", etc. **without** a live model. *(Playback wiring is the
  remaining half.)*

## 10. Runnable entry points (evidence of working capability)

```bash
uv run python -m harp                 # bare bilingual voice core (Gemini default)
uv run python -m harp --provider openai
uv run python -m harp.app             # full supervised agent + dashboard
uv run python -m harp.listener        # wake-listener calibration meter
uv run python -m harp.dashboard       # dashboard alone
```

`python -m harp.app` wires **one shared event bus** into the orchestrator, voice
bridge (with the `search_knowledge` tool and the face-ID identity line), wake
listener, camera + gesture + face-ID loops, and the dashboard — and runs them
concurrently. A wake now opens a **real conversation** that shows up live on the
dashboard.
