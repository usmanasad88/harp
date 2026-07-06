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

### Just the voice core

To smoke-test the mic → provider → speaker path on its own — no orchestrator,
wake listener, vision, or dashboard:

```bash
uv run python -m harp --voice-only                     # bare voice session
uv run python -m harp --voice-only --provider gemini
```

Speak into your mic; it replies through your speakers, grounded in `data/`.
(`python -m harp.app` still runs the same full agent as the default above.)

## Settings

Behavior knobs (wake thresholds, wake words, heartbeat) live in
[harp.yaml](harp.yaml) — edit and restart. Secrets stay in `.env`. To calibrate
the wake listener's sound levels against your own mic and room:

```bash
uv run python -m harp.listener   # live level meter + wake-word test
```

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
  key: space        # hold this to talk (space | enter | a single character)
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

## Layout

- `harp/voice/` — the real-time voice core (working). Provider-agnostic: Gemini
  Live and OpenAI Realtime behind one interface.
- `harp/` (core, orchestrator, presence, vision, knowledge, memory, …) — the rest
  of the architecture, currently scaffolded as skeletons. See
  [DEVLOG.md](DEVLOG.md) for status and [PLAN.md](PLAN.md) for the design.
- `spike_gemini_voice.py`, `web-realtime/` — throwaway references, not the
  product.
