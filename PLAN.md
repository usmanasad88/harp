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
- **It can run push-to-talk in noisy places.** Real-time voice is fragile in
  loud environments (an expo hall): background noise causes false wakes and
  bleeds into the model's turn-taking. A **push-to-talk mode** gives an on-demand
  clean conversation — pressing a key **while idle opens a session** whose
  **microphone is live only while the key is held**. It's *per-session* and runs
  alongside the hands-free wakes: only a press-started session is gated, and when
  it ends HARP returns to normal listening. The mic is gated inside the voice
  bridge (cross-platform, not OS-level). Implemented as a keyboard hotkey
  (harp.yaml `push_to_talk:`); a dashboard hold-button or physical switch could
  drive the same gate later.
- **It knows when a conversation is over — including on request.** Beyond the
  automatic end rules (below), the live agent has an **`end_session` tool** so it
  can hang up itself when the visitor says goodbye or asks it to stop.
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
- **Out of scope (v1):** robot motion commands as part of the core voice/RAG/
  vision/memory loop. **Now planned as its own phase** — see "Motion / robot
  body" below — and ships fully optional: HARP with no gimbal/motors/controller
  attached must keep running exactly as v1 does today.

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
- **Noise robustness for the expo.** The Realtime model is *native audio* — the
  input transcriber is only a side-channel for the dashboard, so noise hurts via
  the audio, not the transcript. Ranked levers, best combined: (1) a
  **volume/proximity gate** — `harp/voice/loudness_gate.py`'s `LoudnessGate`,
  shared by the single-agent bridge and the two-agent filter's mic pump, sends
  silence below a calibrated `near_field_level` so only clear close-mic speech
  commits a turn (automatic push-to-talk keyed on loudness; pre-roll + hangover
  so word onsets aren't clipped) — **built**, live-tunable on the dashboard
  (`harp.yaml` `voice_tuning:`, defaulted off); (2) OpenAI **near-field noise
  reduction** in the session config (near-free) — **built**, same panel; (3)
  raise `server_vad.threshold` (~0.7–0.8) and/or lengthen `silence_duration_ms`
  — **built**, same panel; `semantic_vad` for incomplete sentences not tried;
  (4) a **persona line** to ignore background/fragmentary speech (complement
  only — with auto-response every committed turn still replies) — not built.
  Push-to-talk stays the guaranteed fallback. The `near_field_level` threshold
  is a live calibration against the real mic/room
  (`uv run python -m harp.listener`).
- **Two-agent filter (built 2026-07-06, opt-in, `filter_agent.enabled`,
  default off).** A heavier, LLM-based lever complementing the above: a first
  realtime agent hears the room, discards noise / background / speech not meant
  for HARP, and relays only the intended message (as text) to the responder,
  which never hears raw room audio. Text relay, half-duplex (no barge-in). Costs
  a second live session + ~1-2 s latency, so it's a knob, not the default; the
  cheaper levers and push-to-talk remain first-line. See DEVLOG 2026-07-06;
  `harp/voice/two_agent.py`.

## Motion / robot body (harpcontrol port)

### Findings (investigation 2026-07-09, don't re-litigate)

- `harpcontrol` (Raspberry Pi 5, Ubuntu 24.04, ROS 2 Jazzy) drives a servo
  gimbal (ESP32 over serial, ASCII protocol `Y{yaw}P{pitch}\n` @ 9600 baud) and
  two RMD-X8 base motors (raw byte protocol over **plain serial** @ 115200 —
  **not CAN bus**, no python-can/SocketCAN involved), teleoperated by a PS5
  controller read via `evdev` (Linux-only).
- ROS 2's *entire* role in that repo is passing tiny strings between local
  processes on one machine (`/target_face`, `/face_detected`, `/cmd_input`)
  plus rosbridge as a websocket bridge for a PyQt5 face-animation view. No TF,
  no `ros2_control`, no Nav2, no multi-machine DDS discovery, no cross-language
  message codegen — none of ROS 2's actual value is exercised anywhere in that
  codebase.
- Every piece is Windows-portable as plain Python: `pyserial` (motors + gimbal
  just become COM ports), `pyrealsense2` (RealSense has first-class Windows
  support, arguably better tooling than Linux), `onnxruntime`/`opencv` (no
  issue). The only Linux-only piece is `evdev` for the controller — replace
  with `pygame.joystick` (current code only reads D-pad + 3 digital buttons,
  so no need for anything heavier like `pydualsense` yet).
- **Decision: no ROS 2, no new message-bus protocol.** `harp/core/bus.py`
  (already built, in-process asyncio pub/sub) replaces the ROS 2 topics
  directly — this is structurally the same thing ROS 2 was doing, minus the
  network layer and Linux-only bindings. `harp/dashboard/server.py` (already
  broadcasts every bus event as JSON over `websockets` to any connected
  browser) replaces rosbridge for anything that needs to reach a GUI — e.g. a
  ported face-animation view can open a plain `WebSocket` to HARP's own `/ws`,
  no rosbridge, no `roslib.min.js`. Data rates here are tiny (≤20 Hz PID loop,
  button-rate teleop), so there's no case for pulling in ZeroMQ or an MQTT
  broker.

### Decisions for this phase (locked 2026-07-09)

- Fold the hardware in as new subsystems on the **existing** bus — same shape
  as camera/gestures/face-ID/listener in `app.py`, not a new architecture.
- **Hardware is fully optional at every level**, mirroring how `Camera` already
  works in `app.py` (`camera: Camera | None`; dependent subsystems just don't
  start without it). HARP with no gimbal, no motors, and no controller plugged
  in must run exactly as it does today — voice, RAG, memory, and vision
  unaffected. Each hardware subsystem opens its device at startup and disables
  itself (log a warning, don't crash the app) if the port/device isn't found,
  matching the existing "camera unavailable → gestures + face-ID disabled this
  run" pattern in `run_app()`.
- **Camera is pluggable**: RealSense or a plain webcam, selectable via
  `harp.yaml` (new `camera.backend: auto | realsense | webcam`, default `auto`
  = try RealSense, fall back to webcam). Depth-based nearest-face selection is
  a RealSense-only enhancement layered on top of the same face-position
  signal; a plain webcam still drives head tracking, just without depth
  prioritization among multiple faces.
- **Motor safety stop is mandatory, not optional.** The base-motor subsystem
  owns a deadman timeout: if no teleop command (PS5 or voice) refreshes it
  within a short window (e.g. 250 ms), or the subsystem is shutting down or
  crashing, it writes a zero-speed frame to both motors. This did not exist in
  `harpcontrol/gender/rmd2.py` — ROS 2 wasn't providing it either; it's a
  straight safety gap being fixed here, independent of the platform change.
- **Two teleop sources feeding one command path**: the PS5 controller
  (`pygame.joystick`, replacing `evdev`) AND voice, via new tool calls the live
  model can invoke (`move_forward`, `move_backward`, `turn_left`,
  `turn_right`, `stop_base` — same shape as the existing `end_session` tool in
  `harp/interaction/session_tools.py`). Both sources publish the same bus
  event; the base-motor subsystem doesn't care which one it came from.

### Sketch for the implementing agent (not locked — verify against the code as it lands)

- New package `harp/motion/` (sibling to `harp/vision`, `harp/voice`):
  `gimbal.py` (PID head tracking + ESP32 serial), `base_motors.py` (RMD-X8
  serial writer + deadman timeout), `teleop_ps5.py` (pygame controller → bus),
  plus move/turn/stop tool declarations alongside `session_tools.py`.
- New events in `harp/core/events.py`: something like `FacePosition` (cx, cy,
  frame_w, frame_h, source) for whichever face-detector is active to drive the
  gimbal (see open question below); `TeleopCommand` (direction, speed,
  source: `"ps5"` | `"voice"`) for both teleop sources; consider a hardware-
  presence event so the dashboard can show what's actually connected
  (gimbal/motors/controller present or not).
- `harp.yaml` new sections, following the existing dataclass + `_section()`
  pattern in `config.py`: `camera:` (backend, device index/serial),
  `gimbal:` (enabled, serial port or auto-detect by VID/PID, PID gains, angle
  limits), `base_motors:` (enabled, ports or auto-detect, speed limits,
  deadman timeout seconds), `teleop:` (ps5 enabled, voice enabled).
- Identify serial devices by USB VID/PID/serial number (via
  `serial.tools.list_ports`), not fixed COM port letters — the Linux side
  solved this with udev rules (`/dev/motor_left`); Windows COM letters aren't
  stable across replug, so the equivalent lookup is needed here.

### Open question left to the implementing agent

- `harpcontrol` has a RealSense+ONNX face detector (`depth_detect.py`)
  entirely separate from `harp/vision/face_id.py`'s existing face detection/
  recognition. Decide whether to unify these (face-ID's detector also drives
  the gimbal) or keep them separate — running two face detectors against the
  same camera is likely wasteful, but face-ID may not currently expose
  per-frame pixel position in the form the PID loop needs. Check
  `harp/vision/face_id.py` before assuming either way.

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
9. **Motion / robot body:** port `harpcontrol`'s gimbal + base-motor hardware
   onto the bus as new `harp/motion/` subsystems (no ROS 2) — see "Motion /
   robot body" above for the full findings and scope. Pluggable camera
   (RealSense | webcam), PS5 + voice-tool-call teleop, and a mandatory
   motor deadman stop. Fully optional: HARP runs unchanged with no hardware
   attached.
