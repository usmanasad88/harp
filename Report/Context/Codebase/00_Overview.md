# HARP — Codebase Capabilities & Methodology (Overview)

*Source-of-truth notes distilled directly from the two HARP code repositories, for
use in the FYP / completion reports and presentations. Everything here is grounded
in the actual code, not the marketing slides — where a slide claims something the
code does not (yet) implement, that is flagged.*

## Two codebases, two systems

HARP exists in the code as **two distinct implementations** that share a philosophy
(a bilingual, Gemini-powered assistive robot) but almost no code:

| | `harpcontrol` | `harp` |
|---|---|---|
| **Role** | The **physical robot** as built and demonstrated | A **from-scratch rewrite of the software "brain"** |
| **Runs on** | Raspberry Pi 5, Ubuntu 24.04, ROS 2 Jazzy | Internet-connected laptop (Python, `uv`) |
| **Stack** | ROS 2 nodes + serial hardware + PyQt face UI | Async event-bus Python app + web dashboard |
| **Voice** | Gemini Live (`gemini-2.5-flash-native-audio`) | Provider-agnostic: **Gemini Live *or* OpenAI Realtime** |
| **Vision** | RealSense + YOLOv8-face + depth, head tracking | Webcam + InsightFace face-ID + MediaPipe gestures |
| **Motion** | Differential drive (RMD-X8), PS4 teleop, PID head | **Out of scope** (deferred by design) |
| **Knowledge** | None (persona only) | **RAG** over documents + web-search fallback |
| **Memory** | None | Per-person face memory + conversation summaries |
| **Maturity** | Working, deployed prototype | Modular skeleton; voice/vision/RAG/dashboard working |

**In one sentence:** `harpcontrol` is *the robot that moves, tracks faces, and talks*
today; `harp` is *the smarter, more robust conversational agent* being engineered to
eventually become its brain. The reports should keep the two clearly separated.

## What each file in this folder covers

- **[01_Robot_Control_Hardware.md](01_Robot_Control_Hardware.md)** — the `harpcontrol`
  repository: perception, head tracking, mobility, voice, and the animated face,
  all on ROS 2 / Raspberry Pi. This is the physical demonstrated platform.
- **[02_Voice_AI_Software.md](02_Voice_AI_Software.md)** — the `harp` repository:
  the orchestrated bilingual voice agent with RAG, face-ID, memory, and a dev
  dashboard. This is the software methodology showcase.
- **[03_Methodology.md](03_Methodology.md)** — the engineering methodology common
  to and distinct between the two: architecture styles, control strategies,
  testing, and design decisions worth writing up.

## System-level capability summary (both combined)

Reading the two together, the HARP project as a whole demonstrates:

- **Real-time bilingual (English/Urdu) spoken conversation** via a cloud
  audio-to-audio model, full-duplex and interruptible.
- **Face detection with depth**, and **active head tracking** that keeps a person
  centred, with autonomous "look-around" search when no one is present.
- **Autonomous and remote-controlled mobility** on a differential-drive base.
- **An expressive animated face** that reacts to emotion and to whether a person
  is present.
- **(In the software brain)** retrieval-augmented answering over arbitrary
  documents, recognition of *who* is being spoken to, and per-person memory —
  i.e. a path from a scripted demo to a genuinely context-aware assistant.

> **Accuracy note for report writers:** project slides list "detects age and gender."
> The reviewed code in `harpcontrol` implements **face detection + depth estimation**
> (YOLOv8n-face ONNX) and face *identity* (in `harp`, via InsightFace), but no
> explicit age/gender **classifier** appears in these repositories. Describe age/gender
> as a project objective/earlier module rather than crediting it to this code unless
> you have the corresponding source.
