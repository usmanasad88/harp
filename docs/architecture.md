# Architecture overview

[← Back to index](index.md)

This page explains how HARP hangs together: the event bus that decouples everything, the state
machine that gates the cloud session, the process/thread model, and the full life of a
conversation from wake to long-term memory.

## 1. The decoupling spine: one async event bus

Every subsystem holds exactly two shared things: the `Bus` (in `harp/core/bus.py`) and the event
dataclasses (in `harp/core/events.py`). When the wake listener hears a wake word it does not call
the orchestrator — it publishes a `WakeRequested` event; the orchestrator, subscribed to that
type, reacts. When face-ID notices everyone left the frame it publishes `PresenceChanged`; the
end-rules monitor reacts. Nobody holds a reference to anybody else.

Properties of the bus that shape the whole system:

- **In-process, asyncio-based.** One `Bus` instance is created in `app.py` and passed to every
  subsystem constructor. Publishing fans the event out to every subscriber's private
  `asyncio.Queue`.
- **Never blocks on a slow subscriber.** Each subscriber queue is capped at 64 events; when a
  queue is full, `publish()` drops that subscriber's *oldest* queued event rather than waiting.
  The reasoning: for a real-time agent a stale event (an old presence frame) is worthless, and one
  stuck consumer must not stall the world.
- **Type-filtered subscriptions.** `bus.subscribe(WakeRequested, ErrorRaised)` yields only those
  types; `bus.subscribe()` with no arguments yields everything (used by the dashboard and the
  session log, which is how *new* event types automatically show up in both with no code change).
- **No replay.** A new subscriber only sees events published after it subscribed. Every consumer
  that needs "what is true right now" at startup (a fresh dashboard tab, the end-rules at session
  open) gets it via an injected getter callable instead — this "seed, then follow the bus"
  pattern appears in half a dozen places.

The full event vocabulary (about 25 dataclasses: lifecycle, presence/identity, wake/end,
conversation, memory, tools, controls, health) is catalogued on the [Core layer](core.md) page.

## 2. The state machine and the orchestrator

The app-level state machine is deliberately tiny and pure (`harp/core/state.py`):

```
STARTING ──▶ STANDBY ⇄ ACTIVE
    │           │         │
    └──▶ ERROR ─┘         │        (ERROR → STANDBY on retry)
    └──────────▶ STOPPING ◀┘
```

- **STARTING** — booting: the canned "Starting up" clip plays, connectivity is probed.
- **STANDBY** — alive but idle. No cloud session exists (no money is being spent). The wake
  listener owns the microphone; the camera keeps watching.
- **ACTIVE** — a live voice session is open; someone is being helped.
- **ERROR** — narrating a problem and deciding whether to retry (backoff) or give up.
- **STOPPING** — clean shutdown; a terminal state.

`state.py` only encodes *which transitions are legal* (`can_transition`). All behavior lives in
the **orchestrator** (`harp/orchestrator/orchestrator.py`), which is the single owner of wake
policy: detectors merely publish `WakeRequested`, and the orchestrator honors it only while in
STANDBY (and, in exclusive push-to-talk mode, only when the reason is "button"). This keeps the
expensive decision — opening a cloud session — in exactly one place. Details on the
[Orchestrator](orchestrator.md) page.

## 3. Process and concurrency model

The full agent is **one Python process** running **one asyncio event loop**, plus a handful of
worker threads for things that genuinely block:

| Thread | Created by | Why |
|---|---|---|
| `harp-camera` | `vision/camera.py` | Reading a frame blocks on real hardware; the thread keeps only the newest frame under a lock |
| PortAudio callback threads | `voice/audio_io.py` | sounddevice invokes mic/speaker callbacks on its own threads; bytes are handed to asyncio via `call_soon_threadsafe` |
| pynput key listener | `interaction/push_to_talk.py` | Global keyboard hooks run on their own thread |
| `harp-base-motors` | `motion/base_motors.py` | A 20 Hz serial writer that enforces the deadman timeout independently of everything else |
| Ad-hoc `asyncio.to_thread` calls | many modules | Whisper transcription, InsightFace detection, HTTP web search, WAV playback, serial port opening, the patrol/follow loops — every blocking call is pushed off the loop so voice and bus dispatch never stall |

At startup `app.run_app()` collects one coroutine per enabled subsystem into a dict of named
"runners", spawns them all as tasks, and waits with `FIRST_COMPLETED` semantics: the orchestrator
reaching STOPPING ends the app normally, while *any other* task finishing first is treated as a
crash and re-raised. On the way out, all tasks are cancelled, motors are zeroed, the camera is
released, and the session log is closed last so teardown warnings are still captured.

## 4. The life of a conversation

This is the central flow of the whole system, end to end.

### 4.1 While idle (STANDBY)

- The **always-on listener** (`harp/listener/`) owns the mic. A pure detector classifies audio
  chunks: a sound above `wake_level` is an immediate `LoudSound` wake; speech-level audio is
  captured as a phrase, transcribed locally with faster-whisper, and matched against the
  configured wake words. Either path publishes `WakeRequested(reason, context)` — where `context`
  is a *model-facing* sentence like *"You just woke from standby because someone said 'hello
  harp'..."*, loaded from a template in `prompts/`.
- The **camera stack** keeps watching: the gesture recognizer publishes `GestureDetected("wave")`
  for a held-up open palm, which the trigger engine (`harp/triggers/`) converts into a
  `WakeRequested(reason="wave")`. Face-ID publishes `PersonIdentified` and `PresenceChanged`.
- The **context writer** (`harp/memory/context.py`) exploits idle time: the moment face-ID sees
  someone, it asks the parallel Gemini Flash Lite helper to fuse the camera frame with that
  person's stored memories into a *wake briefing* — cached so that when the wake comes, the
  session opens with zero added latency already knowing who it is talking to.
- With push-to-talk armed, pressing the talk key publishes `WakeRequested(reason="button")`.
- With push-to-talk + status voice, the **idle prompt** replays "Please hold the green button to
  talk to me" every 45 s.

### 4.2 Opening (STANDBY → ACTIVE)

The orchestrator receives `WakeRequested`, checks state and wake policy, transitions to ACTIVE,
publishes `InteractionStarted`, and starts the **voice bridge** as a task. The bridge:

1. builds a fresh `SessionConfig` (persona from `prompts/system_instructions.md`, the current
   dashboard VAD/noise tuning, and the tool declarations for every capability enabled this run);
2. connects to the provider (Gemini Live or OpenAI Realtime), opens mic and speaker;
3. sends one opening text message into the session: the wake context ("why you woke") plus the
   identity context ("who you're talking to" — the pre-computed briefing if fresh, else the
   static face-ID line);
4. then pumps in both directions: mic chunks (through the push-to-talk gate and the loudness
   gate) go up; the provider's events come down and are translated onto the bus —
   `UserTranscript` → `UserSaid`, `AgentTranscript` → `AgentSaid`, `ToolCall` →
   `ToolRequested`/`ToolCompleted` (running the tool dispatcher in between), `AudioOut` → the
   speaker, `Interrupted` → clear the speaker, `ProviderError` → `ErrorRaised`.

Meanwhile the **interaction logger** (`harp/memory/logger.py`) opens a per-conversation
transcript file and records every turn, tool call, and face sighting.

### 4.3 During the session

The model can call tools: `search_knowledge` (BM25 over `data/`), `web_search` (DuckDuckGo
fallback), `search_memory` (its own history), `describe_scene` (a fresh camera look via the
helper model), `move_around` / `follow_person` (the wheeled base, when motion is enabled), and
`end_session` (hang up on itself). See [Agent tools](agent-tools.md).

### 4.4 Ending (ACTIVE → STANDBY)

Four independent things can end a session, all by publishing `EndOfInteractionDetected(reason,
cause)`:

| Cause | Published by | Trigger |
|---|---|---|
| `walked_off` | `EndOfInteractionMonitor` | No face in frame for `absence_timeout_seconds` (default 10 s) |
| `silence` | `SilenceMonitor` | Nothing said in either direction for `silence_timeout_seconds` (default 15 s) |
| `agent` | the `end_session` tool | The visitor said goodbye and the model hung up on itself |
| `provider` | the orchestrator | The provider closed the stream (session limit, network) |

The orchestrator cancels the bridge task, transitions back to STANDBY, publishes
`InteractionEnded`, and plays the end-of-session clip the status **rule book** maps to that cause
("Goodbye." vs "Going on standby.").

### 4.5 After the session

`InteractionEnded` makes the logger finalize the transcript (rename `.part` → `.jsonl`), and the
**summarizer** sweeps: it mechanically digests the transcript (who, when, what was searched),
makes ONE rate-limited Flash Lite call producing `{summary, follow_up, person_facts}`, and
attaches the memory to every enrolled participant's record — or, if nobody was recognized, to the
guestbook file (strangers' faces are never stored). Failed calls leave the transcript pending for
the next sweep or the next boot; crashed runs' `.part` files are rescued at boot. The next time
that person appears on camera, the context writer folds these memories into the wake briefing —
closing the loop.

## 5. Voice-path variants

The mic-to-model path has three configurations, all presenting the same `run(context)` interface
to the orchestrator:

1. **Single agent (default)** — `VoiceBridge`: mic → provider → speaker, with optional
   push-to-talk gating and loudness gating.
2. **Push-to-talk / exclusive** — same bridge, but the mic gate substitutes digital silence for
   real audio unless the talk key is held (silence, not nothing, so the provider's VAD still ends
   turns cleanly). In exclusive mode the wake listener isn't even started and the orchestrator
   vetoes all non-button wakes.
3. **Two-agent filter (experimental)** — `TwoAgentBridge`: a *filter* agent hears the room and
   relays only messages intended for HARP, as text, to a *responder* session that never hears raw
   audio. Half-duplex, ~1–2 s extra latency, one extra live session. See [Voice](voice.md).

## 6. Observability

Three layers, all fed by the same bus:

- **The dashboard** (`http://127.0.0.1:8787`) — live view of everything: state, transcript, tool
  calls, wake phrases, camera with overlays, plus its few narrow controls (mic mute, voice
  tuning, camera source, patrol button). The `/user` page is the visitor-facing kiosk view.
- **The session log** (`.harp/logs/session-<ts>.jsonl`) — one flushed-per-line JSONL timeline per
  run: a settings header, every bus event, and every Python log record, for post-hoc debugging.
- **Status voice** — canned local audio clips so a running robot is audible-legible ("Starting
  up", "I can't reach the internet", "Goodbye") even with no screen and no cloud.

## 7. Where policy lives (edit here, not there)

| To change... | Edit |
|---|---|
| Thresholds, timeouts, toggles, ports | [`harp.yaml`](configuration.md) |
| What any model/tool is told | [`prompts/*.md`](configuration.md#the-prompts-system) |
| Which clip plays at which life-cycle moment | `harp/orchestrator/status_rules.py` |
| The words of the clips themselves | `scripts/generate_status_voice.py` (then re-run it) |
| What HARP knows (RAG corpus) | `data/*.md` |
| Who HARP recognizes | `people/` + `scripts/enroll_people.py` |
| Which subsystems exist at all | `harp/app.py` (the composition root) |
