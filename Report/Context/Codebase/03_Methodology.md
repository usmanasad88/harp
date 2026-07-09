# HARP — Engineering Methodology

*How the two codebases were designed and built. Useful for the "Methodology" and
"System Design" sections of the reports. The two tracks share an event-driven,
decoupled philosophy but apply it in different technology stacks.*

## 1. Distributed, message-passing architecture (both tracks)

Both HARP implementations refuse the single-file monolith. Each is a **set of
independent components that communicate only through messages**, never by importing
one another:

- **`harpcontrol`** uses **ROS 2 topics** as the message bus. Perception, head
  control, mobility, teleop, voice, and the animated face are **separate ROS nodes /
  processes**, coupled only by named topics (`/target_face`, `/face_detected`,
  `/cmd_input`, `/user_emotions`). A single `ros2 launch` file brings the whole
  system up. A web UI is bridged in over **rosbridge** (websocket), so even the
  browser front-end is "just another node."
- **`harp`** uses an in-process **async publish/subscribe event bus**
  (`core/bus.py`) as the spine. Subsystems (orchestrator, listener, vision, voice
  bridge, dashboard) publish and consume **typed events** (`WakeRequested`,
  `PersonIdentified`, `UserSaid`, `ToolRequested`, `Heartbeat`, …) and **never import
  each other**. A single composition root (`app.py`) wires them onto one shared bus.

**Why it matters for the write-up:** the same design principle — *loose coupling via
messages* — appears at two scales (a robot-wide ROS graph and an in-process event
bus). Each subsystem is independently runnable and testable, which is exactly what
lets the project grow without rewrites.

## 2. Control & signal-processing strategies (`harpcontrol`)

- **Classical PID control** for head tracking, one independent loop per axis (yaw,
  pitch), operating on normalized image error, with integral clamping and rate
  limiting on the serial output. Chosen for smoothness and tunability on an edge CPU.
- **Sensor fusion:** RGB **detection** (YOLOv8n-face in ONNX) fused with **aligned
  depth** to pick the *nearest* person as the tracking target — a distance prior that
  a 2-D detector alone can't give.
- **Robustness details** worth citing: NMS to dedupe detections; **median depth** over
  a patch (rejects depth-map holes); a **staged, interruptible search behaviour** when
  no face is present; hardware limit clamping and guaranteed motor-zeroing on exit.
- **Direct hardware protocols:** hand-built serial byte frames for the RMD-X8 motors
  (header + signed speed + checksum) and a compact ASCII command to the ESP32 head —
  i.e. the team implemented the device protocols themselves rather than relying on
  vendor middleware.

## 3. Software-engineering methodology (`harp`)

- **Spike first, then build.** The real-time voice loop (and Urdu quality) was proven
  in a throwaway spike **before** any architecture was committed — de-risking the
  hardest unknown first.
- **Incremental, subsystem-at-a-time.** The full architecture was scaffolded as
  importable **skeletons** (docstring + "to build" list + `NotImplementedError`), then
  each subsystem was filled in, verified, and only then wired into `app.py`. Nothing
  is delivered as one big-bang commit.
- **Provider abstraction.** A thin interface isolates Gemini vs. OpenAI, so a vendor
  swap doesn't ripple through RAG, vision, memory, or the UI.
- **Test-driven with fakes for heavy dependencies.** ~100 automated tests run with **no
  camera, GPU, model download, or API key** — the webcam, InsightFace, MediaPipe, the
  voice providers, and the mic/speaker are all faked. Hardware- and model-dependent
  paths are verified separately with small manual preview scripts.
- **Graceful degradation & self-narration.** A missing webcam disables only the
  camera-fed features (not voice); the agent speaks canned status lines with no live
  model; errors are narrated and retried rather than failing silently.
- **Sandbox → consolidation.** A separate web sandbox (`web-realtime/`) is used to tune
  provider settings and tool-calling; proven settings are then **ported into the
  product** (e.g. the OpenAI realtime config and the `search_knowledge` tool).
- **Reference harvesting.** Patterns from an existing, battle-tested Gemini Live
  integration were reused deliberately (VAD gating, robust audio I/O, the
  tool-calling loop) rather than reinvented.

## 4. AI / model methodology (both tracks)

- **Cloud audio-to-audio** conversation (Gemini Live native audio; OpenAI Realtime as
  the second provider in `harp`) rather than a chained STT → LLM → TTS pipeline — lower
  latency and more natural turn-taking, including barge-in.
- **Bilingual by prompt, not by config.** Language mirroring is steered through the
  **system instruction**, which the native-audio path honours, so EN/Urdu switching is
  automatic and mid-conversation. (OpenAI transcription is additionally nudged to
  render spoken Urdu in Perso-Arabic script.)
- **Grounding via tool calls.** In `harp`, knowledge retrieval is a **function/tool** the
  model chooses to call before answering — the standard, provider-portable way to do
  RAG in a realtime session — with vision (camera frames) and identity ("you are
  talking to &lt;name&gt;") injected the same way.
- **Local models at the edge for the always-on parts.** The wake listener transcribes
  **offline** with faster-whisper, and face embeddings run **on-device** with
  InsightFace — the expensive cloud call is reserved for the actual conversation.

## 5. Design decisions worth citing in the report

- **Cost-aware session gating.** The orchestrator's whole reason to exist: a
  continually-running assistant that only opens the paid cloud session when a person
  is present (wake word, loud sound, or a wave), and closes it at end-of-interaction.
- **Privacy stance.** Unknown faces are **reported but never stored**; real people's
  enrollment photos and face fingerprints are kept out of version control.
- **Observability without coupling.** The dashboard is a **pure observer** of the event
  bus — added, tested, and proven **before** the full app existed to feed it — with a
  single, deliberate write exception (OS-level mic mute).
- **Right-sized choices over cargo-cult scale.** BM25 keyword search instead of a
  vector DB for a small corpus; brute-force cosine matching instead of a vector index
  for dozens of people — with the seams (`indexer.py`) left in place to scale later.

## 6. Suggested framing for the reports

- Present `harpcontrol` as the **realised prototype** (perception, tracking, mobility,
  voice, expressive face on real hardware) and `harp` as the **software-engineering and
  AI contribution** (a robust, modular, testable, provider-agnostic agent with RAG,
  recognition, and memory).
- Emphasise the **shared architectural thesis** — decoupled components communicating by
  messages — as the through-line that makes HARP extensible rather than a fixed demo.
- Be precise about **status**: distinguish *working* subsystems from *scaffolded* ones
  (voice, wake, RAG, vision, dashboard work; end-of-interaction rules, proactive
  triggers, memory summarisation, status-voice playback, and web-search fallback are
  in progress) so the reports claim exactly what the code delivers.
