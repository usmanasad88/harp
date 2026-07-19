# Core layer (`harp/core/`)

[← Back to index](index.md)

The core package is the small set of things *every* subsystem is allowed to depend on: the event
bus, the event vocabulary, the state machine, and the per-run developer log. Nothing in here
knows about voice, vision, or any other subsystem.

## `core/bus.py` — the async pub/sub bus

One class, `Bus`, shared by the whole process (constructed once in `app.py`).

- `publish(event)` — delivers the event to every current subscriber whose type filter matches.
  It **never blocks**: each subscriber has a private `asyncio.Queue` capped at 64 entries
  (`_QUEUE_MAXSIZE`), and if a queue is full the *oldest* queued event is discarded to make room
  (`queue.get_nowait()` then `put_nowait`). The rationale, stated in the module docstring: for a
  real-time agent a late event is worthless, and one stuck consumer must not stall the world.
- `subscribe(*types)` — returns an async generator of events. With arguments, only those event
  types are yielded; with none, everything is (this is how the dashboard and session log capture
  event types invented after they were written). The queue is registered *eagerly*, before the
  caller first awaits, so events published between subscribing and first iteration are not lost.
  Cleanup happens in the generator's `finally`, which only runs once iteration has started — the
  code comments warn that a subscription abandoned before reading any event lingers until GC.

That's the entire bus: ~60 lines. There is no cross-process transport, no persistence, no topic
strings — types *are* the topics.

## `core/events.py` — the shared event vocabulary

These dataclasses are the **only** contract between subsystems. They are deliberately small,
provider-agnostic, and behavior-free. (They intentionally overlap with `voice/provider.py`'s
`VoiceEvent` types — those are the low-level events of one live voice session; the ones here are
the system-wide vocabulary. The voice bridge translates one into the other.)

All inherit from an empty `Event` base class (which is what lets `subscribe()` filter by type).

### Lifecycle / state

| Event | Fields | Meaning |
|---|---|---|
| `StateChanged` | `old`, `new` (state names as strings) | The orchestrator moved between app states |
| `ShutdownRequested` | `reason` | Something asked HARP to shut down cleanly |

### Presence & identity (published by vision)

| Event | Fields | Meaning |
|---|---|---|
| `PresenceChanged` | `present: bool`, `count: int` | Is anyone in frame, and roughly how many. Published by face-ID only on *changes*, not every pass |
| `PersonIdentified` | `person_id`, `name`, `is_known`, `confidence` | Face-ID resolved who is in frame; strangers get `person_id="unknown"`, `is_known=False` |
| `GestureDetected` | `kind` (e.g. `"wave"`) | A debounced, recognized gesture — a proactive cue |

### Wake / end requests (what the orchestrator acts on)

| Event | Fields | Meaning |
|---|---|---|
| `PhraseHeard` | `text`, `wake_word` (matched word or `None`) | The idle listener transcribed a phrase — published whether or not it matched, so the dashboard can show what the ears picked up |
| `WakeRequested` | `reason` (loud sound / wake word / wave / button), `context` | Please open a session. `context` is the model-facing explanation delivered into the session at open. Only honored while STANDBY |
| `EndOfInteractionDetected` | `reason` (human-readable), `cause` (machine key: `walked_off` / `silence` / `agent` / `provider` / `""`) | The conversation is judged over; the orchestrator should close the session. `cause` keys the status rule book |

### Interaction / conversation

| Event | Fields | Meaning |
|---|---|---|
| `InteractionStarted` | `reason`, `context` | A live session opened |
| `InteractionEnded` | `reason` | The session closed (the memory summarizer's cue) |
| `UserSaid` | `text`, `final` | What HARP heard the user say. Streaming: non-final deltas carry pieces, a final marks the turn closed |
| `AgentSaid` | `text`, `final` | What HARP said back, same streaming shape |
| `TalkKeyChanged` | `held` | The push-to-talk key's *effective* hold flipped (debounce-bridged, so a hardware button's tap train reads as one hold). The kiosk page renders this |

### Memory

| Event | Fields | Meaning |
|---|---|---|
| `MemoryWritten` | `person_ids`, `summary`, `follow_up` | The summarizer stored a memory; empty `person_ids` means it went to the guestbook |
| `ContextPrepared` | `people`, `text` | The context writer pre-computed a wake briefing for whoever is currently in frame |

### Tools

| Event | Fields | Meaning |
|---|---|---|
| `ToolRequested` | `id`, `name`, `arguments` | The live model asked for a tool (mirrored onto the bus for the dashboard) |
| `ToolCompleted` | `id`, `output` | The tool finished; its output is on its way back to the model |

### Controls (dashboard-related state changes)

| Event | Fields | Meaning |
|---|---|---|
| `MicMuteChanged` | `muted` | The OS-level mic mute flipped (mutes the physical device via the system mixer) |
| `CameraSourceChanged` | `source` (auto / realsense / webcam / usb_webcam), `backend` (what's actually driving frames) | The selected camera source changed |
| `MoveAroundChanged` | `active`, `note` | The stall patrol started/stopped. Published **only** by the `MoveAroundController`, never by the dashboard server |
| `FollowChanged` | `active`, `person`, `note` | Follow mode started/stopped. Published only by the `FollowController` |
| `VoiceTuningChanged` | `near_field_level`, `vad_threshold`, `vad_silence_ms`, `noise_reduction` | The live noise/VAD tuning changed; carries the full snapshot so every open dashboard tab stays in sync |

### Health

| Event | Fields | Meaning |
|---|---|---|
| `Heartbeat` | `ts` | Emitted every `heartbeat.interval_seconds` so observers know the agent is alive |
| `ErrorRaised` | `where`, `message`, `fatal` | A subsystem failed. Non-fatal → the orchestrator narrates and retries with backoff; fatal → STOPPING |

## `core/state.py` — the state machine

A five-value string enum `AppState` (`starting`, `standby`, `active`, `error`, `stopping`) plus a
transition table and one function:

- `can_transition(current, target) -> bool` — is the move legal?

Legal moves: STARTING → {STANDBY, ERROR, STOPPING}; STANDBY → {ACTIVE, ERROR, STOPPING};
ACTIVE → {STANDBY, ERROR, STOPPING}; ERROR → {STANDBY, STOPPING}; STOPPING → nothing (terminal).

The module is pure (no I/O, no bus) so it is unit-testable on its own. All *behavior* attached to
states lives in the orchestrator; only the shape of the machine lives here.

## `core/session_log.py` — the per-run developer log

Writes one append-only JSONL timeline per run of the full agent (default `.harp/logs/
session-<timestamp>.jsonl`, configured by `session_log:` in `harp.yaml`). Its consumer is whoever
must reconstruct, after the fact, what a run actually did — a human or an AI agent reading it
cold. Three sources merge into one time-ordered file:

1. **A `session_start` header** — the *effective* settings (merged `harp.yaml` + defaults), the
   resolved provider/model/voice, platform and Python version. So the file states what this run's
   knobs actually were, not what `harp.yaml` says today. No secrets: API keys never appear.
2. **Every bus event, verbatim** — via a no-filter subscription (`run()` iterates
   `bus.subscribe()` and calls `log_event` on each). Event types added later are captured
   automatically.
3. **Every Python `logging` record** — `handler()` returns a `logging.Handler`
   (`_RecordHandler`) that `app.py` attaches to the root logger, so camera warnings, mic retries,
   and "filter heard (asr)" lines land in the same timeline they used to lose when the terminal
   closed. Records with exception info get the full traceback under an `"exc"` key.

### The `SessionLog` class

- `open(header)` — prunes old runs (keeps the newest `keep_runs`, default 30; timestamp-named
  files sort chronologically), creates this run's file (a `-2` suffix disambiguates two runs in
  the same second), and writes the header. Must be called before attaching the handler.
- `run()` — the bus-persisting coroutine (one of app.py's runner tasks, registered *first* so
  its catch-all subscription exists before the orchestrator's first boot events).
- `log_event(event)` — serializes a dataclass event; if `dataclasses.asdict` fails on an exotic
  field it degrades to `{"repr": repr(event)}`.
- `close(reason)` — writes a `session_end` record and stops accepting lines; anything logged
  after close (late teardown threads) is silently dropped.
- `_write(record)` — stamps every line with epoch (`t`) and ISO (`iso`) time, JSON-encodes with
  `default=repr`, and **flushes after every line** so a crash preserves the record up to the
  moment it happened. Thread-safe via a lock, because log records arrive from any thread.

Fail-safety is the design theme: a value JSON can't encode falls back to `repr`; a circular
reference drops just that line; a write error drops just that line; the `_RecordHandler` never
logs from inside `_write` so it can't recurse. Logging can never take the agent down — important
because `app.py` treats any exited task as a crash.

**Do not confuse this with the interaction transcript** (`harp/memory/logger.py`): that is the
per-*conversation* record the memory summarizer consumes, with a different consumer and lifetime.
This is the per-*run* debugging record.
