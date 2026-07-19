# Interaction (`harp/interaction/`)

[← Back to index](index.md)

Everything about how a conversation session starts by hand and ends on its own: the two
automatic end rules, push-to-talk, the idle invite, and the model's own `end_session` tool.

| File | Role |
|---|---|
| `end_rules.py` | `EndOfInteractionMonitor` (walk-off rule) and `SilenceMonitor` (nobody-spoke rule) |
| `push_to_talk.py` | Hold-a-key-to-talk, including exclusive mode and hardware-button debounce |
| `idle_prompt.py` | The periodic "hold the green button" invite while idle |
| `session_tools.py` | The `end_session` tool the live model can call to hang up |

## `end_rules.py` — when a session closes on its own

Both monitors are pure bus consumers, armed only between `InteractionStarted` and
`InteractionEnded` (so neither can ever fire while HARP is idle), and both publish the same
`EndOfInteractionDetected` — distinguished by `cause` so the status rule book narrates each
differently.

### `EndOfInteractionMonitor` — the person walked off

Rule: while a session is open, **no face in frame** for a continuous `absence_timeout` (default
10 s, `interaction.absence_timeout_seconds`) ends it with `cause="walked_off"`. Face-ID doubles
as the presence signal (it publishes `PresenceChanged` while watching for faces anyway), so no
separate presence detector is needed.

Mechanics: it tracks `_active` (session open) and `_present` (latest presence), and on every
relevant event re-evaluates: active AND absent → arm a countdown task; anything else → disarm.
A returning face cancels the countdown, and a fresh absence starts the *full* countdown over —
deliberately erring toward not cutting people off (brief detection dropouts, a turned head, a bad
frame are absorbed by the timeout).

The subtle part — **presence at session start**: the bus never replays, and face-ID publishes
only on *changes*. A session that opens with nobody in frame (woken by voice or a loud sound from
across the room) would never receive an "absent" event and would stay open forever. So at each
`InteractionStarted` the monitor reads the current truth directly through the injected
`is_present` getter (wired by `app.py` to `face_id.current is not None`). With no camera this run
(`is_present=None`) it assumes present and simply never auto-closes on absence — the silence rule
below still applies.

### `SilenceMonitor` — nobody said anything

The independent second rule, closing the hole the face rule leaves: a person standing in frame
who never talks (a false wake, or push-to-talk with the button never pressed) keeps face-presence
alive indefinitely. Rule: **nothing said in either direction** for `silence_timeout` (default
15 s, `interaction.silence_timeout_seconds`; 0 disables the monitor entirely — `app.py` then
doesn't run it) ends the session with `cause="silence"`.

The countdown starts the moment the session opens (so a session where nothing is *ever* said
still closes) and restarts on every sign of life: `UserSaid` or `AgentSaid` — including streaming
fragments, which prove someone is mid-utterance — and `TalkKeyChanged`, so someone who just
pressed the talk button isn't cut off before their words are transcribed.

## `push_to_talk.py` — hold a key to talk

The answer to loud rooms: the mic reaches the model **only while a key is held**. It is a
*per-session mode*, not a global switch:

- At startup push-to-talk is inactive; the wake listener and wave trigger work normally.
- Pressing the key **while STANDBY** publishes `WakeRequested(reason="button")` with a context
  telling the model to listen first rather than deliver a long welcome
  (`prompts/wake_context_push_to_talk.md`) — and marks *this* session as push-to-talk: for its
  whole duration, `mic_open` follows the key.
- When that session ends (any way), a `StateChanged` back to STANDBY clears the mark; hands-free
  wakes resume. A session that was woken hands-free is *not* gated — `mic_open` stays True
  throughout it.

**Exclusive mode** (`push_to_talk.exclusive: true`) makes the button the entire interface. This
class itself only changes `mic_open` (held-only in *every* session); the rest is wired in
`app.py`: the wake listener is not started at all, and the orchestrator's `wake_allowed`
predicate vetoes every non-button wake.

### The gate

`mic_open` is what the voice bridge consults per chunk (via the `mic_gate` callable app.py
wires). When False, the bridge sends same-length digital silence instead of room audio — a
continuous stream, so the provider VAD still closes turns. State is kept in bare bools/floats
written from the key-listener thread and read from the asyncio side; the code notes that
single-word reads/writes are atomic under the GIL and a one-chunk (~64 ms) lag is inaudible.

### Release debounce (hardware buttons)

`release_debounce_seconds` exists for HARP's physical arcade button: an ESP32 BLE keyboard whose
firmware cannot hold keys — it re-TAPS the whole combo ~2.5×/s while pressed (~95 ms down,
~300 ms up, measured with `spike_ptt_gate.py --debug-keys`). With a debounce longer than the tap
gap (0.7 s in `harp.yaml`), a key-up re-pressed within the window counts as one continuous hold,
so the tap train becomes the hold the firmware can't send. Consequences handled in the code:

- `held` is True while the key is down *or* within the debounce window after a key-up.
- A press inside the window is a tap-train continuation, not a new button press — it must not
  trigger a second wake (`new_hold` logic in `press()`).
- `TalkKeyChanged` publishing is deduped (`_publish_held`) so the kiosk page's green "Listening"
  screen never flickers mid-train; on a debounced release, `_settle_release` re-arms itself on
  the event loop (`call_later`) until the window truly expires, then publishes the close.
- The cost: the mic stays open ~a debounce-worth after each real release.

### Keyboard backend

The global key listener is pynput, injected via `listener_factory` so the class is testable by
driving `press()`/`release()` directly and importing the module never requires pynput or a
display. The default factory parses the `harp.yaml` key spec — one key (`space`, `m`) or a
`+`-combo (`ctrl+shift+m`) — into a set of pynput keys; `_combo_handlers` implements
hold-the-whole-combo semantics (activate when the last member lands, release when any member
lifts, edge-triggered so OS auto-repeat can't re-fire), with `listener.canonical()` folding
left/right modifier variants and stripping modifier effects (Ctrl+M's `\r` → `m`). An
unrecognized spec falls back to space with a warning. A listener that fails to start (missing
display, permissions) logs and disables push-to-talk for the run rather than crashing.

## `idle_prompt.py` — the periodic invite

With push-to-talk armed (especially exclusive mode at an expo), a passer-by has no way to know
HARP is voice-driven. `IdlePrompt` replays one status clip ("Please hold the green button to talk
to me", clip id resolved through the rule book's `idle_prompt` moment) every
`push_to_talk.idle_prompt_seconds` (default 45; 0 disables) **while the app is in STANDBY**.

Mechanics: it watches `StateChanged` and keeps a repeating timer task alive only during STANDBY —
leaving STANDBY cancels it immediately, so the prompt never opens over a live conversation. It
plays through the same `StatusVoice` instance the orchestrator uses, whose internal lock
serializes clips (the prompt can never talk over a boot/error/goodbye line). The first prompt
comes one full interval *after* standby starts, never instantly — so the "going on standby" line
gets airtime and a person whose conversation just ended isn't immediately lectured.

## `session_tools.py` — the `end_session` tool

Lets the realtime model hang up on itself: when the visitor says goodbye or asks it to stop, the
model (taught by `prompts/end_session_tool.md`: say a short spoken goodbye FIRST, then call) calls
`end_session`. The handler publishes
`EndOfInteractionDetected(reason="agent ended the session (...)", cause="agent")` — which the
orchestrator handles exactly like the automatic end rules — and returns
`{"ok": True, "note": "Ending the session now. Goodbye."}` to the model.

The module's shape (`declarations(provider)` returning the OpenAI flat-function or Gemini
`function_declarations` format, plus an async handler) is the template all other tool modules
follow; see [Agent tools](agent-tools.md). Because the handler needs the bus, `app.py`'s
dispatcher routes `end_session` here explicitly.
