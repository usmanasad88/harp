# HARP — Codebase Documentation

HARP (Humanoid Assistant Robotic Platform) is a bilingual (English/Urdu) **real-time voice
assistant robot** built for a reception/expo setting. A visitor walks up, wakes it with a wake
word, a loud sound, a wave, or by holding a talk button, and has a spoken conversation with
"Laila" — a persona running on a cloud realtime voice model (OpenAI Realtime or Gemini Live).
Answers are grounded in a local document corpus (`data/`), the robot can see through a camera
(gestures, face recognition, scene description), remembers the people it has talked to, can
narrate its own status through pre-recorded clips, and can physically move (patrol its stall or
follow a recognized person on its wheeled base).

This documentation describes **how the code is organized and how every module works**, in plain
language. It is written to be a stand-in for reading the source: each page walks through a
package file-by-file, explaining what each class and function does, what events it consumes and
produces, and how it fails.

## The one-paragraph mental model

Everything is a set of small, independent **subsystems** that never import each other. They
communicate only by publishing and subscribing to typed events on one shared in-process **event
bus**. A single **orchestrator** owns the app's state machine (starting → standby ⇄ active →
stopping) and is the only thing that decides when a paid cloud conversation opens or closes.
One file — [`harp/app.py`](app.md) — is the *composition root*: the only place that knows all the
subsystems exist, constructs them, wires them to the bus, and runs them concurrently as asyncio
tasks. Because of this, any subsystem can be disabled (in `harp.yaml` or by commenting it out in
`app.py`) and the rest keeps running.

## Documentation contents

### Foundations

| Page | Covers |
|---|---|
| [Architecture overview](architecture.md) | The big picture: event bus, state machine, process/thread model, the life of a conversation from wake to memory |
| [Entry points & running HARP](entry-points.md) | `python -m harp` and every other runnable module and script |
| [Configuration](configuration.md) | `harp.yaml` section-by-section, `.env` variables, and the `prompts/` system |
| [Core layer](core.md) | `harp/core/` — the bus, the full event vocabulary, the state machine, the per-run session log |

### Subsystems (one page per package)

| Page | Package | Role |
|---|---|---|
| [Voice](voice.md) | `harp/voice/` | The realtime conversation core: provider abstraction, Gemini + OpenAI backends, mic/speaker I/O, the supervised bridge, the two-agent noise filter |
| [Wake listener](listener.md) | `harp/listener/` | Always-on idle listening: loudness detection, phrase capture, local Whisper wake-word matching |
| [Orchestrator](orchestrator.md) | `harp/orchestrator/` | The supervisor state machine, error/retry policy, canned status voice and its rule book |
| [Interaction](interaction.md) | `harp/interaction/` | How sessions end (walk-off, silence, the model hanging up), push-to-talk, the idle invite prompt |
| [Vision](vision.md) | `harp/vision/` | The shared camera, gesture recognition, face identification, camera snapshots, the `describe_scene` tool |
| [Memory](memory.md) | `harp/memory/` | Long-term memory of people and conversations: transcripts, summaries, wake briefings, `search_memory` |
| [Knowledge](knowledge.md) | `harp/knowledge/` | RAG over `data/`: the BM25 retriever, the `search_knowledge` tool, the web-search fallback |
| [Motion](motion.md) | `harp/motion/` | The robot body: wheel motors with deadman safety, the stall patrol, follow-me mode, the servo head, controller teleop |
| [Dashboard](dashboard.md) | `harp/dashboard/` | The developer web dashboard and the visitor-facing kiosk page |

### Reference

| Page | Covers |
|---|---|
| [The composition root (`app.py`)](app.md) | Line-by-line walkthrough of how everything is wired together |
| [Agent tools](agent-tools.md) | Every tool the live model can call, in one place: declarations, handlers, and behavior |
| [Data, prompts & on-disk state](data-and-state.md) | The `data/` corpus, `prompts/`, `people/`, `assets/`, and everything written under `.harp/` |
| [Tests](tests.md) | What the test suite covers and how it is able to test hardware/cloud code |

## Repository layout at a glance

```
harp/                     the application package
  __main__.py             CLI: `python -m harp` (full agent) / --voice-only
  app.py                  composition root — wires every subsystem to the bus
  config.py               harp.yaml loading, prompt loading, provider defaults
  audio_control.py        OS-level mic mute (pactl)
  core/                   bus, event vocabulary, state machine, session log
  voice/                  realtime voice: providers, audio I/O, bridge, filter
  listener/               always-on wake listener (loudness + Whisper)
  orchestrator/           supervisor, retry policy, status voice + rule book
  interaction/            end rules, push-to-talk, idle prompt, end_session tool
  vision/                 shared camera, gestures, face-ID, snapshots, describe
  memory/                 person store, matcher, Gemini helper, summarizer, ...
  knowledge/              BM25 retriever, search_knowledge / web_search tools
  motion/                 wheel motors, patrol, follow, gimbal head, teleop
  dashboard/              websocket dashboard server + static frontend
  presence/, triggers/    thin/reserved: wave→wake rule engine, presence seam
harp.yaml                 user-editable behavior settings (see Configuration)
prompts/                  every piece of text sent to a model, as markdown
data/                     the RAG corpus search_knowledge retrieves from
people/                   enrollment folders (photos + info.yaml) for face-ID
assets/status_voice/      pre-rendered status clips + manifest
assets/models/            the YOLOv8n-face ONNX model for motion tracking
scripts/                  enrollment, TTS generation, previews, site scraper
tests/                    the pytest suite
.harp/                    runtime state (logs, memory, heartbeat) — gitignored
```

## Key design rules (recur on every page)

1. **Subsystems talk only through the bus.** No subsystem imports another; they agree only on
   the event dataclasses in `harp/core/events.py`. This is why each can be built and tested in
   isolation with fake events.
2. **Policy is centralized.** Detectors report facts (`WakeRequested`, `PresenceChanged`);
   only the orchestrator decides to open/close a paid session. What HARP *says* at each life-cycle
   moment lives in one rule book (`orchestrator/status_rules.py`). What models are *told* lives in
   `prompts/` as markdown, not in Python.
3. **Everything degrades, nothing crashes.** A missing camera, API key, prompt file, audio
   device, or motor port disables that capability for the run with a logged warning; the rest
   keeps working. Tools return `{"error": ...}` payloads instead of raising, so a failed tool
   becomes the model apologizing rather than a dead session.
4. **Hardware and cloud are injected.** Providers, microphones, speakers, serial ports, key
   listeners, and the Gemini helper's caller are all constructor parameters with real defaults,
   so tests inject fakes and never need hardware.
5. **The bus never replays history.** Anything that needs current state at connect/open time
   (dashboard tabs, end rules, the interaction logger) is *seeded* through an injected getter —
   a pattern you will see repeatedly.
