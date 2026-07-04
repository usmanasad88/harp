# HARP — Rewrite Plan

A bilingual (English/Urdu) **continually-running** voice assistant for the
Humanoid Assistant Robotic Platform. It runs unattended for long stretches,
manages its own uptime, and only spins up an expensive cloud voice session when
someone is actually there to talk to. Grounded in **any** knowledge you give it
(RAG over arbitrary documents), able to see through a camera (image Q&A + face
recognition), and able to remember the people it has spoken to.

## The vision

HARP is not a one-shot voice demo. It is a **robust orchestrator** around a
continually-running chatbot — a supervisor that keeps the assistant alive,
healthy, and responsive over many hours, and that calls on a set of independent
subsystems as the situation demands.

Picture how it behaves through a day:

- **On start-up** it speaks pre-recorded / pre-programmed status lines —
  "Starting up", then "Connection established" once it reaches the cloud, or
  "No internet" if it can't. These canned outputs work without a live model.
- **It narrates its own problems.** When something goes wrong it articulates the
  error out loud in plain terms and **retries** as appropriate, rather than
  failing silently.
- **It survives crashes.** A secondary **watchdog/monitor** process checks that
  the main agent is alive; if the process dies, it restarts it.
- **It does not hold a cloud session open for 12 hours.** That is too costly and
  fragile, so the orchestrator gates the live session: it has **standby / sleep
  modes** and wakes only when needed.
- **It knows when nobody's there.** A separate **presence** subsystem detects
  humans in the webcam. With no one around and no one talking, HARP goes to
  sleep; a person appearing (or speaking) wakes it.
- **It recognizes who it's talking to.** When an interaction begins, a
  **face-recognition** subsystem logs the user and, on a new interaction, checks
  **past memories for a match** so HARP can greet them in context.
- **It remembers conversations.** Everything is **logged**; when an interaction
  ends, a separate memory-creation step reads the logs and writes a **memory
  summary** for that individual.
- **It knows when a conversation is over.** There are **defined end-of-interaction
  rules** — e.g. the person leaves the frame and stays silent for some time.
- **It can start conversations on its own.** Alongside reacting to people, HARP
  has **rule-based proactive triggers** that let the orchestrator open a session
  unprompted. Two kinds:
  - **Gesture / cue based.** While idle, if someone waves at the robot (we have
    palm-gesture recognition), HARP greets them and starts the conversation —
    no wake word needed.
  - **Memory based / follow-up.** A past conversation can leave a **follow-up
    intent** in memory — e.g. someone was looking for something and HARP gave
    them directions. Later, if that person is detected in frame again, a
    subsystem looks up their memory log and **re-opens the session to follow
    up** ("Did you find what you were looking for?").
  These are the *kind* of rules we want, not the exact set — the trigger
  conditions and cool-downs are to be designed so HARP is helpful, not pushy.
- **It can look things up.** It **retrieves from the local `data/` folder** and,
  when needed, can fall back to **internet search**.

Everything above is direction, not locked design. Exact mechanisms and
implementation choices are to be settled through brainstorming in later sessions.

## Decisions (locked)

- **Interaction:** cloud real-time voice (full-duplex, interruptible).
- **Provider:** **pluggable** — support both **Gemini Live** and **OpenAI Realtime**,
  selectable at runtime behind one common interface. Capabilities differ per
  provider (e.g. native video input is stronger on Gemini); the app degrades
  gracefully.
- **Knowledge:** **context-agnostic.** Drop any documents into `data/`; they're
  indexed into a vector store. Nothing is hardcoded to a specific corpus.
- **Deployment:** internet-connected laptop/cloud (not offline robot — yet).
- **UX (phase 1):** the **end-user interacts by voice only — no screen.** The core
  is a **headless Python voice agent** (mic + speaker + camera on the device). A
  separate **developer dashboard/monitor** observes the agent (live transcripts,
  retrieved context, face-ID, latency) but is not part of the end-user flow.
  One deliberate exception (2026-07-02): a mic-mute button on the dashboard,
  reachable from the same LAN (see `dashboard.bind` in harp.yaml) — the one
  write action the dashboard is allowed; it's a physical/OS-level mute
  (`harp/audio_control.py`, `pactl`), not a new agent behavior.
- **Architecture style:** **incremental, modular.** Build it as a set of
  **independently functional subsystems that talk to each other** — never a
  single-shot monolith. Each subsystem should be runnable and testable on its
  own, with the orchestrator wiring them together.
- **Stack:** Python core; web-based dev dashboard added later.
- **Scope (v1):** orchestrated voice chat + RAG, bilingual EN/Urdu, vision (image
  Q&A + face-ID), presence-driven sleep/wake, crash recovery, and per-person
  conversation memory.
- **Out of scope (v1):** robot motion commands (deferred).

## Architecture

```
                 ┌──────────────────────────────────────────────┐
   watchdog ────►│            ORCHESTRATOR (supervisor)          │
  (restarts)     │  state machine: starting / standby / active  │
                 │  pre-programmed voice, error narration, retry │
                 └───┬───────────┬───────────┬──────────┬────────┘
                     │           │           │          │
              presence      voice session  memory    knowledge
            (webcam: is   (Gemini Live OR  (logs →   (RAG over data/
             anyone here) OpenAI Realtime) summaries) + web search)
                     │           │           │
                 face-ID ◄───────┘     end-of-interaction rules
            (who is it? match    (left frame + silent for a while)
             against memories)
```

Subsystems publish data (presence, who's present, transcripts, retrieved
context) that the orchestrator — or the cloud agent itself — consumes. A thin
**provider abstraction** isolates Gemini vs OpenAI so the rest of the app (RAG,
vision, memory, UI) is written once.

## Verified facts (don't re-litigate)

- **Spike works — Urdu and English both sound good** in real time (verified
  2026-07-01).
- **Model:** `gemini-3.1-flash-live-preview` — native audio-to-audio, 90+ languages
  (Urdu covered), accepts image/video/text input, 128K context.
- **Client must use** `http_options={"api_version": "v1beta"}` to reach the Live API.
- **Language is auto-detected** by the native-audio model. The reliable lever for
  language is the **system instruction** ("reply in the same language the user
  speaks"), not `speech_config.language_code` (which the native-audio path ignores).
- Spike code: [spike_gemini_voice.py](spike_gemini_voice.py). Throwaway scaffolding,
  not the final architecture (uses `sounddevice`, no VAD/tools/vision).

## Proven reference: aura `SoundMonitor`

`~/Repos/aura/src/aura/monitors/sound_monitor.py` is the user's existing,
battle-tested Gemini Live integration. **Reference, not a dependency** — harvest
its patterns when building HARP's real voice core:

- VAD energy-gating (only send speech + trailing silence — saves tokens, kills
  spurious replies).
- Robust audio I/O: `pyaudio` + scipy resample + by-name device select with
  ALSA / PulseAudio / `arecord` / `parec` fallbacks (needed for real robot mics).
- **Tool-calling loop** (`tool_call` → handler → `FunctionResponse`) — this is the
  delivery mechanism for **RAG** (`search_knowledge`) and **web search**.
  Declaration format: `~/Repos/aura/src/aura/interfaces/voice_action_bridge.py`.
- `send_image()` — already-solved path for **vision** (camera frames).
- `ContextWindowCompressionConfig` sliding window for long sessions.
- Native-audio quirk handling: thinking-text filter + repetition suppression.

## Open / in-progress

- `web-realtime/` is a **sandbox**, not the product — experiments are still
  ongoing there (OpenAI realtime tuning, RAG/tool-calling). The product is the
  Python headless agent + dashboard described above. Settings and findings that
  prove out in the sandbox get **consolidated into `harp/`**; the OpenAI session
  settings are the next candidates (port into `harp/voice/openai.py`).
- Many mechanisms above are still open: exactly how the watchdog supervises, how
  end-of-interaction is judged, and how memories are stored and matched. **To be
  worked out through brainstorming, not assumed.** (Speech-based wake is now
  settled: the always-on listener in `harp/listener/`, tuned via `harp.yaml`.)

## Build phases

Built incrementally — each subsystem stands on its own before it's wired in.
Nothing here is a single-shot build, and every phase needs its own testing.

1. ✅ **Spike** — realtime voice + Urdu verified. DONE.
2. **Orchestrator skeleton:** start/standby/active state machine, pre-programmed
   status voice lines, error narration + retry, and a watchdog that restarts the
   agent if it dies.
3. **Presence + sleep/wake:** webcam human-detection drives standby↔active so the
   cloud session only runs when someone is there.
4. **RAG:** index documents in `data/` ONCE into a vector store; expose
   `search_knowledge(query)` as a function/tool call (use aura's tool-calling
   loop). Add internet-search fallback.
5. **Vision + face-ID:** stream camera frames (`send_image`); recognize and log
   the user, inject "you are talking to <name>" into context. Add **palm-gesture
   recognition** as a proactive cue.
6. **Memory:** log every interaction; on a defined end-of-interaction, summarize
   the logs into a per-person memory; match new visitors against past memories.
   Capture **follow-up intents** so a later sighting can be acted on.
7. **Proactive triggers:** rule-based engine that lets the orchestrator open a
   session unprompted — gesture/cue based (wave → greet) and memory based (known
   person reappears with an open follow-up → re-engage). Includes the guard rails
   (cool-downs, opt-out) that keep it helpful rather than pushy.
8. **UI + polish:** dev dashboard (transcripts, retrieved context, face-ID,
   latency, agent health), provider + language toggle. OpenAI Realtime as the
   second provider behind the abstraction.
