# HARP

A bilingual (English/Urdu) **real-time voice assistant** for the Humanoid
Assistant Robotic Platform. The end-user just talks to it — no screen. It will
answer grounded in any knowledge you give it (RAG), and see through a camera.

This repo is a from-scratch rewrite. See [PLAN.md](PLAN.md) for the vision and
locked decisions, and [DEVLOG.md](DEVLOG.md) for what's been built so far.

## Run the voice core

Uses [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env          # then add your GEMINI_API_KEY (and OPENAI_API_KEY)
uv run python -m harp                     # Gemini (default)
uv run python -m harp --provider openai   # OpenAI Realtime
```

`uv run` installs dependencies from the lockfile on first use. Speak into your
mic (English or Urdu); it replies through your speakers. `Ctrl+C` to quit.

## Run the full agent (with dashboard)

```bash
uv run python -m harp.app     # orchestrator + wake listener + camera gestures
```

Then open <http://127.0.0.1:8787> — the developer dashboard, a live read-only
view of the agent: state, heartbeats, wake requests, every phrase the idle
listener heard (and whether it matched a wake word), detected gestures, and
the camera view — with a live box + label drawn over whatever hand gesture
the recognizer currently sees. Note the actual cloud voice session is **not** bridged into
the orchestrator yet, so waking it (say a wake word) flips it to `active` but
no conversation starts.

## Settings

Behavior knobs (wake thresholds, wake words, heartbeat) live in
[harp.yaml](harp.yaml) — edit and restart. Secrets stay in `.env`. To calibrate
the wake listener's sound levels against your own mic and room:

```bash
uv run python -m harp.listener   # live level meter + wake-word test
```

## Layout

- `harp/voice/` — the real-time voice core (working). Provider-agnostic: Gemini
  Live and OpenAI Realtime behind one interface.
- `harp/` (core, orchestrator, presence, vision, knowledge, memory, …) — the rest
  of the architecture, currently scaffolded as skeletons. See
  [DEVLOG.md](DEVLOG.md) for status and [PLAN.md](PLAN.md) for the design.
- `spike_gemini_voice.py`, `web-realtime/` — throwaway references, not the
  product.
