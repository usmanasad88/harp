# Orchestrator (`harp/orchestrator/`)

[← Back to index](index.md)

The supervisor. It owns the state machine, gates the cloud voice session (wake policy lives
here and nowhere else), handles errors with narration + backoff, proves liveness, and speaks
canned status lines through a policy "rule book".

| File | Role |
|---|---|
| `orchestrator.py` | The supervisor loop itself |
| `retry.py` | Pure backoff/give-up policy |
| `status_voice.py` | Plays canned status clips by stable id |
| `status_rules.py` | THE RULE BOOK: which clip plays at which life-cycle moment |
| `watchdog.py` | Reserved stub: an external keep-alive process (not implemented) |

## `orchestrator.py` — the supervisor

`Orchestrator(bus, provider_name, ...)` takes everything optional-and-injected: heartbeat
interval and file, the voice bridge, the status voice, a connectivity probe, and a `wake_allowed`
predicate. Constructed with all of them `None` (as tests and partial wirings do) it still drives
states and events correctly, silently — that seam is deliberate.

### The main loop (`run()`)

Subscribes to `WakeRequested`, `EndOfInteractionDetected`, `ErrorRaised`, `ShutdownRequested`;
starts the heartbeat task; runs `_boot()`; then reacts to events until STOPPING:

- `ShutdownRequested` → `_shutdown()`: close any open session (without the normal end-of-session
  narration), play the "shutdown" moment, transition to STOPPING.
- `WakeRequested` → honored only while STANDBY. If a `wake_allowed` predicate was injected
  (exclusive push-to-talk injects `reason == "button"`), a non-passing wake is logged and
  dropped — the veto that makes button-only mode real.
- `EndOfInteractionDetected` → only meaningful while ACTIVE: `_close_session(reason, cause)`.
- `ErrorRaised` → `_handle_error()` (below).

### State transitions

`_to(target, reason)` validates against `core.state.can_transition` (raising on an illegal move —
a programming error, not a runtime condition), swaps the state, logs, and publishes
`StateChanged`. The `state` property is also read directly by `app.py` to seed fresh dashboard
connections.

### Boot

`_boot()` plays the "boot" moment ("Starting up."), then — only if a connectivity probe was
injected (app.py wires `_internet_reachable`, a real TCP connect to 8.8.8.8:53 run off-thread) —
plays "boot.online" ("Connection established.") or "boot.offline" ("I can't reach the internet
right now."), then transitions to STANDBY. The probe is best-effort and never blocks reaching
STANDBY.

### Session lifecycle

- `_open_session(reason, context)` — STANDBY → ACTIVE, reset the error counters (a successful
  open proves recovery), start `voice_bridge.run(context)` as a task, publish
  `InteractionStarted`.
- `_run_session(context)` — the task body. It never mutates orchestrator state directly; it
  translates the bridge's fate into bus events the main loop reacts to like anyone else's:
  a crash publishes `ErrorRaised(where="voice.session")`; a *clean return* (the provider closed
  the stream — server-side session limit, clean network drop) while still ACTIVE publishes
  `EndOfInteractionDetected(cause="provider")`.
- `_close_session(reason, cause, narrate=True)` — cancel the session task, ACTIVE → STANDBY,
  publish `InteractionEnded` (the memory summarizer's cue), then — on a normal close — play the
  `session_end.<cause>` moment. Error and shutdown paths pass `narrate=False` because they speak
  their own line instead.

### Error handling

`_handle_error(ev)`:

1. If ACTIVE, close the session (no narration — the error line is coming).
2. Fatal → play "error.fatal", transition to STOPPING.
3. Non-fatal → transition to ERROR; play the closest error moment, chosen by `_error_line(where)`
   from the `where` string ("mic"/"audio" → `error.mic`; "voice"/"session"/"provider"/"network" →
   `error.connection`; else `error.generic`); count consecutive errors; if
   `retry.should_give_up(attempts, elapsed)` → STOPPING; otherwise sleep
   `retry.backoff_seconds(attempt)` and transition back to STANDBY ("retrying after error").

### Heartbeat

`_heartbeat()` proves liveness two ways, forever: a `Heartbeat(ts)` event on the bus for
in-process observers (the dashboard's health panel), and touching the heartbeat file
(`.harp/heartbeat`) — for the watchdog, which as a separate process cannot see the bus and judges
liveness by the file's mtime.

## `retry.py` — the retry policy

Pure functions, no I/O, unit-testable without real failures:

- `backoff_seconds(attempt)` — exponential 1 s, 2 s, 4 s, 8 s... capped at 30 s.
- `should_give_up(attempt, elapsed)` — True after 5 consecutive failures or 120 s of continuous
  failing.

## `status_voice.py` — canned speech that works without the cloud

`StatusVoice(assets_dir, lang)` plays pre-rendered WAV clips from `assets/status_voice/` by
**stable id** (`play("starting_up")`), resolved through `manifest.json` (id → lang → file). This
is speech for exactly the moments the cloud voice is unavailable: boot, outages, errors,
shutdown.

Design rules, enforced here so no caller can violate them:

- **Serialized**: an asyncio lock ensures one clip at a time — two quick transitions can't talk
  over each other, and the idle prompt can't talk over a goodbye.
- **Fail-safe**: a missing manifest, unknown id, missing file, or dead audio device is a logged
  no-op, never an exception — callers may `await play(...)` on any code path unguarded.
- **Language fallback**: requested lang → default lang → English, so a partially translated
  manifest still speaks. (Only `en` ships today.)
- The audio sink is injected (default: a blocking WAV playback via sounddevice, run off-thread),
  so tests need no sound card and importing the module needs no audio stack.

The clips are regenerated by `scripts/generate_status_voice.py` (Kokoro offline TTS, run from a
separate venv because torch would bloat HARP's). The words live in that script's `LINES` dict.

## `status_rules.py` — THE RULE BOOK

The one file to edit to change what HARP says (or keep it silent) at a life-cycle moment. The
orchestrator, the error handler, follow-mode announcements, and the idle prompt all look their
line up here by a stable **moment key** — none of them hardcode a clip id.

`RULES` maps moment → clip id (or `None` = stay silent). `line_for(moment)` resolves dotted keys
with parent fallback: `session_end.surprise` falls back to `session_end`, so an unmapped variant
still says something sensible.

Current policy (see the file itself for the fully commented map):

| Moment | Clip | Notes |
|---|---|---|
| `boot` | starting_up | first thing a run says |
| `boot.online` / `boot.offline` | connection_established / no_internet | after the connectivity probe |
| `session_end.walked_off` | session_ended ("Goodbye.") | person left frame |
| `session_end.silence` | session_ended | nobody spoke for the timeout |
| `session_end.agent` | going_standby | the model already said its own goodbye |
| `session_end.provider` | going_standby | provider closed the stream; keep it neutral |
| `session_end` | going_standby | fallback for unmapped causes |
| `error.mic` / `error.connection` / `error.generic` | mic_problem / connection_lost / error_recoverable | non-fatal narration before retry |
| `error.fatal` | error_fatal | narrated once, then the app stops |
| `shutdown` | shutting_down | orderly shutdown |
| `idle_prompt` | hold_green_button | the push-to-talk invite |
| `follow.no_person` / `follow.started` / `follow.stopped` | follow_no_person / follow_started / follow_stopped | follow-mode announcements |

To re-word a line: edit the text in `scripts/generate_status_voice.py`, re-run it (ids stay
stable, so this file doesn't change). To add a line: add id+text there, re-run, then reference
the new id here.

## `watchdog.py` — reserved stub

The intended external supervisor: a separate, dependency-free process that spawns the agent,
watches the heartbeat file's mtime, and restarts the agent on death or hang with backoff.
Currently just a documented `NotImplementedError` — the heartbeat file it would consume is
already being written.
