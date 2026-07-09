# HARP Robot Control & Hardware (`harpcontrol` repository)

*The physical, demonstrated robot: a ROS 2 system on a Raspberry Pi 5 that sees,
tracks, talks, moves, and shows an animated face. Grounded in the code under
`harpcontrol/gender/`.*

## 1. Deployment platform

- **Compute:** Raspberry Pi 5 (8–16 GB recommended).
- **OS / runtime:** Ubuntu 24.04, Python 3.11, **ROS 2 Jazzy Jalisco**.
- **Sensors & actuators:**
  - Intel RealSense depth camera (D435 / D455) — RGB + depth.
  - **ESP32** over USB serial — drives the two head servos (yaw + pitch).
  - **2× RMD-X8** brushless motors over serial — the differential-drive base.
  - USB microphone + USB speaker/soundbar — voice I/O.
  - Display / tablet — the animated face.
  - PS4 wireless controller — teleoperation.

The whole system is launched together by one ROS 2 launch file
(`launch/harp_system.launch.py`), which spawns rosbridge, the face UI, the voice
assistant, the drive-motor controller, PS4 teleop, depth detection, and face
tracking as separate processes.

## 2. Perception — face detection + depth (`depth_detect.py`)

- Runs a **YOLOv8n-face** model in **ONNX Runtime** (CPU) on the RealSense colour
  stream (640×480), with standard letterboxing, confidence thresholding, and
  **non-maximum suppression** hand-implemented in NumPy.
- Aligns the **depth** stream to the colour frame (`rs.align`) so every detected
  face has a real-world distance; distance is taken as the **median depth** over a
  small patch at the face centre (robust to holes/noise in the depth map).
- Selects the **nearest face** (smallest depth) as the tracking target and
  publishes its pixel centre `"{cx} {cy}"` on the ROS topic **`/target_face`**.
- Renders a live debug window with the bounding box, distance in metres, and a
  smoothed FPS counter.

**Capability in report terms:** real-time, distance-aware detection of the closest
person, at ~30 FPS on an edge device, feeding a control loop.

## 3. Active head tracking — PID → ESP32 (`facetracking.py`)

- A ROS 2 node subscribes to **`/target_face`** and runs an independent **PID
  controller per axis** (yaw and pitch) on the normalized pixel error between the
  face centre and the frame centre, with integral wind-up clamping and derivative
  terms.
- Converts the PID output into servo angles (clamped to safe mechanical limits —
  yaw 20–90°, pitch 40–70°) and streams them to the **ESP32** over serial as a
  compact `"Y{yaw}P{pitch}"` command at ~20 Hz.
- **Autonomous idle behaviour** when no face is seen:
  - < 0.5 s lost → keep tracking;
  - > 2 s → return to a neutral rest pose;
  - > 3 s → begin a **smooth "look-around" sweep** across the yaw range, staged
    and interruptible the instant a face reappears.
- Publishes a boolean on **`/face_detected`** so the animated face UI can react to
  a person appearing or leaving.

**Capability in report terms:** the robot **holds eye contact**, follows the nearest
person smoothly, gracefully returns to rest, and **searches for people on its own**
when the scene is empty — all with classical, tunable PID control.

## 4. Voice assistant — Gemini Live (`harp.py`, `cameraharp.py`)

- Full **real-time, audio-to-audio** conversation using Google **Gemini Live**
  (`gemini-2.5-flash-native-audio-preview`) over the `v1beta` API, voice "Aoede".
- Async I/O pipeline (`asyncio.TaskGroup`): capture mic → stream to Gemini →
  receive audio → play back, plus optional **camera streaming** (JPEG frames at a
  fixed cadence) so the model can *see* while it talks.
- **Robust audio device handling:** auto-detects the correct mic and speaker by
  name from the device list, queries native sample rates, and **resamples**
  Gemini's 24 kHz mono output to whatever the speaker supports, up-mixing mono→stereo.
- A fixed **HARP persona** in the system instruction: identity (NUST CEME
  Mechatronics, funded by Pro Rector RIC NUST), a warm/polite/concise tone, and
  guardrails (never claims emotions/consciousness, fixed location answer, scripted
  goodbye).
- `cameraharp.py` is the same assistant with camera vision enabled by default and
  its own camera auto-detection (`cv2.CAP_V4L2`, probing indices).

**Capability in report terms:** natural spoken dialogue, bilingual-capable via the
native-audio model, optionally grounded in live camera view, running on the robot.

## 5. Mobility — differential drive + teleop (`rmd2.py`, `ps4control.py`)

- **Motor driver (`rmd2.py`):** talks to two **RMD-X8** motors on dedicated serial
  ports (`/dev/motor_left`, `/dev/motor_right`, 115200 baud) by building the
  motor's **raw speed-command byte frame** (header + little-endian signed RPM×100 +
  checksum). Provides forward/reverse/left/right/stop and live speed ramping, and
  always zeroes the motors on exit for safety.
- **Two input paths into the same driver:**
  - Direct **keyboard** teleop (non-blocking raw-terminal key reads).
  - **PS4 controller** via `evdev` (`ps4control.py`): finds the wireless controller,
    maps D-pad and face buttons to movement/speed commands, and publishes them on
    the ROS topic **`/cmd_input`**, which the motor node consumes.

**Capability in report terms:** the platform is **both autonomously-capable and
remote-controlled** — the same low-level driver is fed either by an operator's
gamepad/keyboard or (by design) by higher-level ROS commands.

## 6. Expressive animated face (`face_animations/`)

- A **PyQt5 + QtWebEngine** full-screen window hosts an HTML/CSS/JS animated face
  (`index.html`, `index.js`) with procedurally-animated eyes and eyelids
  (blinking, per-emotion keyframes).
- Driven **over ROS via rosbridge**: the web page opens a websocket to
  `ws://localhost:9090` (roslib) and subscribes to:
  - **`/user_emotions`** — one of `happy / sad / angry / focused / surprise /
    neutral`, switching the facial expression;
  - **`/face_detected`** — reacts when a person appears or leaves.
- A ROS node (`main.py`) bridges the same emotion topic on the Python side.

**Capability in report terms:** a **socially expressive front-end** decoupled from the
control code — emotion and presence are just ROS messages, so any subsystem can
change HARP's "face" without touching the UI.

## 7. How it all fits together (data flow)

```
RealSense ─► depth_detect.py ─(/target_face)─► facetracking.py ─(serial)─► ESP32 head servos
                   │                                  │
                   │                          (/face_detected)
                   ▼                                  ▼
             (nearest person)                 face_animations (PyQt+Web, via rosbridge :9090)
                                                      ▲
PS4 pad ─► ps4control.py ─(/cmd_input)─► rmd2.py ─(serial)─► RMD-X8 drive motors
                                                      │
Mic/Cam ─► harp.py (Gemini Live) ─► speaker      (/user_emotions drives expression)
```

Everything is glued by **ROS 2 topics** and brought up by a single launch file — a
clean, distributed, node-per-responsibility design (see
[03_Methodology.md](03_Methodology.md)).
