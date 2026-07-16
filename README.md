# HARP

A bilingual (English/Urdu) **real-time voice assistant** for the Humanoid
Assistant Robotic Platform. The end-user just talks to it — no screen. It will
answer grounded in any knowledge you give it (RAG), and see through a camera.

This repo is a from-scratch rewrite. See [PLAN.md](PLAN.md) for the vision and
locked decisions, and [DEVLOG.md](DEVLOG.md) for what's been built so far.

## Run it

Uses [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env          # then add your OPENAI_API_KEY (and/or GEMINI_API_KEY)
uv run python -m harp                     # full agent + dashboard (OpenAI default)
uv run python -m harp --provider gemini   # Gemini Live instead
```

`uv run` installs dependencies from the lockfile on first use (first run also
downloads the vision + wake-word models, so give it a minute).

This is the **full supervised agent**: the orchestrator, the real cloud voice
session (grounded in `data/` via `search_knowledge`), the always-on wake
listener (threshold-based session start, tuned in [harp.yaml](harp.yaml)), the
camera + gesture + face-ID vision stack, and the developer dashboard — all
sharing one event bus. Wake it with a wake word, a loud sound, or a wave, then
talk (English or Urdu); it replies through your speakers. `Ctrl+C` to quit.

Open <http://127.0.0.1:8787> for the developer dashboard — a live read-only view
of the agent: state, heartbeats, wake requests, every phrase the idle listener
heard (and whether it matched a wake word), the transcript and tool calls of the
live conversation, detected gestures, face-ID, and the camera view with a live
box + label drawn over whatever the recognizer currently sees.

The same server also serves <http://127.0.0.1:8787/user> — the **end-user
screen**, meant for a display facing the visitor (put it full-screen with F11).
It shows one thing at a time: "Hold the green button to talk" (EN + Urdu) while
idle, a green full-screen "Listening" while the push-to-talk key is held,
pulsing dots while the reply is on its way, then the agent's reply streamed in
as it speaks — and back to the prompt when the turn is over.

### Just the voice core

To smoke-test the mic → provider → speaker path on its own — no orchestrator,
wake listener, vision, or dashboard:

```bash
uv run python -m harp --voice-only                     # bare voice session
uv run python -m harp --voice-only --provider gemini
```

Speak into your mic; it replies through your speakers, grounded in `data/`.
(`python -m harp.app` still runs the same full agent as the default above.)

### Robot body (standalone, no ROS)

The physical robot — the servo gimbal head and the RMD-X8 wheel motors from
the old `harpcontrol` repo — now runs from this repo as plain Python, no
ROS 2. Phase 1 is a **separate entry point**, not yet wired into the agent:

```bash
uv run python -m harp.motion --list-ports        # find your serial ports first
uv run python -m harp.motion --gimbal-port COM5 --left-port COM6 --right-port COM7
```

Face tracking (RealSense if present — nearest face by depth — else any
webcam) drives the head; a PS4/PS5 controller drives the wheels: **hold** the
D-pad to move (release to stop), Cross = stop, Square / Circle = speed up /
down. Add `--preview` for a live detection window, `--test-controller` to
verify the button mapping. Every piece of hardware is optional — whatever
isn't plugged in is skipped with a warning and the rest runs. The wheels have
a mandatory deadman stop: if teleop stops refreshing for 0.25 s (crash, hang,
controller loop dead), both motors are zeroed.

## Settings

Behavior knobs (wake thresholds, wake words, heartbeat) live in
[harp.yaml](harp.yaml) — edit and restart. Secrets stay in `.env`. To calibrate
the wake listener's sound levels against your own mic and room:

```bash
uv run python -m harp.listener   # live level meter + wake-word test
```

### Agent prompts

Everything the cloud agents are actually *told* — Laila's persona, the
two-agent filter's persona, every tool description, and the "you just woke up
because..." / "you're talking to `<name>`" context sent at the start of a
conversation — lives in [prompts/](prompts/) as plain markdown, not buried in
Python. Edit a file and restart HARP to change how an agent behaves; see
[prompts/README.md](prompts/README.md) for the full index of files and which
agent/tool each one drives.

### Session logs (developer)

Every run of the full agent writes one timeline file to `.harp/logs/`
(`session-<timestamp>.jsonl`): the settings the run actually used, every event
on the bus (state changes, wakes, the conversation, tool calls, detections,
errors), and every internal log line — one JSON object per line, flushed as
written, so even a crashed run keeps its record up to the moment it died. Read
it (or hand it to an agent) to debug a past run cold. Old runs are pruned
automatically; configure or disable with `session_log:` in
[harp.yaml](harp.yaml).

### Long-term memory

HARP remembers the people it talks to. A **parallel Gemini Flash Lite
helper** (separate from the live voice session, capped at 14 calls/min to
stay inside the free tier) does three jobs:

- **After every conversation** it writes a memory — what was discussed, any
  open follow-up ("they were looking for hall B"), and facts the visitor
  shared — attached to every recognized (enrolled) participant in
  `.harp/memory/people/`. Conversations with strangers go to
  `.harp/memory/guestbook.jsonl` instead; no face is ever stored for them.
  Raw turn-by-turn transcripts live in `.harp/memory/interactions/`.
- **Before a conversation**, when someone appears on camera while HARP is
  idle, it pre-writes a briefing from the live camera frame plus that
  person's stored memories — so the moment HARP wakes, it already knows who
  it's talking to and what's unfinished, with no added wake latency. The
  briefing refreshes when the people in frame change or every 2 minutes.
- **During a conversation** the live agent can call `describe_scene` (a fresh
  look through the camera) and `search_memory` (its own history: "do you
  remember me?").

Configure with `memory:` in [harp.yaml](harp.yaml) (`enabled`, `model`,
`calls_per_minute`, `context_ttl_seconds`). Needs `GEMINI_API_KEY` in `.env`
(even when the voice provider is OpenAI); without it, transcripts are still
recorded and get summarized by a later run that has the key.

### Status voice

HARP speaks short pre-recorded status lines — "Starting up", "Connection
established" / "I can't reach the internet", "Going on standby", error notices,
"Shutting down" — so you can *hear* what it's doing at boot, on errors, and when
a conversation ends, even without the dashboard. They play from local clips
(`assets/status_voice/`), so they work when the cloud voice can't. Toggle with
`status_voice:` in [harp.yaml](harp.yaml). If the clips aren't present HARP just
stays silent about status; regenerate them once with
[scripts/generate_status_voice.py](scripts/generate_status_voice.py).

### Push-to-talk (for noisy places)

In a loud room (e.g. an expo hall) an always-listening mic false-wakes on crowd
noise and gets its turn-taking wrecked by background speech. **Push-to-talk**
gives you an on-demand, clean conversation — the mic reaches the model *only
while you hold a key*:

```yaml
# harp.yaml
push_to_talk:
  enabled: true     # arms the key; HARP still boots hands-free
  key: ctrl+shift+m # hold this to talk: a key (space, enter, m) or a '+'-combo (ctrl+shift+m)
  exclusive: false  # true = push-to-talk ONLY: the model never hears the mic
                    # unless the key is held — even hands-free-woken sessions
                    # stay gated (they get silence until you hold the key)
```

It runs *alongside* the normal hands-free wake (wake word / loud sound / wave).
**Press the key while HARP is idle** to start a session whose mic is gated:
it hears you only while the key is down, so crowd noise can't false-wake it or
cut into its turn. When that session ends — you walk away, or you tell it to
close — HARP returns to hands-free listening. The live agent can also **end the
conversation itself**: say goodbye or ask it to stop and it hangs up.

### Two-agent noise filter (experimental)

Another approach to loud rooms: put a **filtering agent** in front of the
assistant. A first real-time agent listens to the mic, throws away background
chatter, crowd noise, and anything not meant for HARP, and passes on **only the
message the visitor intends** (as text) to the normal responder — which never
hears the raw room, just "silence, then a clean request", and replies by voice:

```yaml
# harp.yaml
filter_agent:
  enabled: true       # off by default; single agent runs when false
  provider: ""        # "" = same provider as the responder; or gemini | openai
```

It costs a second live session and adds ~1–2 s of latency, and it's **half-duplex
(you can't interrupt HARP mid-reply)**, so it's a knob rather than the default —
push-to-talk above stays the guaranteed fallback. Watch the dashboard transcript
to see what the filter chose to relay vs. drop.

### Voice/noise tuning (dashboard)

Open <http://127.0.0.1:8787> and you'll find a **Voice tuning** panel with live
sliders for a loudness/proximity gate, VAD threshold, VAD silence, and noise
reduction — the levers for fighting a real-time model committing (or
hallucinating) a turn from room noise. It applies to whichever agent currently
owns the microphone: the plain single-agent session by default, or the
two-agent filter's session when `filter_agent.enabled` above is on. Defaults
live in `harp.yaml`'s `voice_tuning:` section; the dashboard sliders override
them at runtime, and copying good values back into `harp.yaml` persists them.
Calibrate the loudness gate against your own mic/room with
`uv run python -m harp.listener`.

## Layout

- `harp/voice/` — the real-time voice core (working). Provider-agnostic: Gemini
  Live and OpenAI Realtime behind one interface.
- `harp/` (core, orchestrator, presence, vision, knowledge, memory, …) — the rest
  of the architecture, currently scaffolded as skeletons. See
  [DEVLOG.md](DEVLOG.md) for status and [PLAN.md](PLAN.md) for the design.
- `spike_gemini_voice.py`, `web-realtime/` — throwaway references, not the
  product.
