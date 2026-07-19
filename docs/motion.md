# Motion (`harp/motion/`)

[← Back to index](index.md)

The robot body, ported from the old `harpcontrol` ROS repo to plain Python: two RMD-X8 wheel
motors on serial ports (differential drive), a servo gimbal head on an ESP32, face tracking to
drive the head, PS4/PS5 teleop, and two agent-drivable behaviors — the bounded **stall patrol**
(`move_around` tool / dashboard button) and **follow-me** (`follow_person` tool).

| File | Role |
|---|---|
| `base_motors.py` | The wheel motors: serial protocol + deadman-guarded writer thread |
| `patrol.py` | The reusable drive → look-around → turn lap (blocking, thread-oriented) |
| `controller.py` | `MoveAroundController` — bus-wired owner of the patrol |
| `follow.py` | `FollowController` + the pure `steer()` policy — follow a known person |
| `gimbal.py` | The servo head: PID face tracking + idle look-around over ESP32 serial |
| `face_tracker.py` | YOLOv8n-face ONNX detection + RealSense/webcam cameras for tracking |
| `teleop_ps5.py` | Hold-to-drive D-pad teleop via pygame |
| `__main__.py` | `python -m harp.motion` — the standalone body runner |
| `autonomous_patrol.py` | Standalone patrol CLI with a controller E-STOP |
| `tools.py` | The `move_around` and `follow_person` tool declarations/handlers |

## Safety model (read this first)

Layered, outermost first:

1. **Bounded behaviors** — the patrol runs a fixed number of laps and stops by itself; follow
   stops when it loses its person.
2. **Cooperative stop** — every motion primitive checks a `threading.Event` at 20 Hz and writes
   an immediate zero-speed command on the way out.
3. **The deadman timeout** (`base_motors.py`) — a dedicated writer thread re-sends the current
   command at 20 Hz; if nobody has refreshed the command within 0.25 s (teleop thread died,
   script hung, detector stalled), it writes zero-speed frames on its own. Teleop and the
   follow/patrol loops therefore *continuously* re-send commands (hold-to-drive), never latch a
   speed and walk away.
4. **Mutual exclusion** — patrol and follow refuse to start over each other via `conflict`
   callbacks wired in `app.py`, so the two can never fight over the serial ports; the app's
   shutdown path stops whichever is running and zeroes the wheels.
5. Follow-specific: face-ID must keep vouching for the target every pass, or the wheels zero.

## `base_motors.py` — the wheels

`make_rmdx8_speed_command(rpm)` builds one raw RMD-X8 speed-control frame (header + little-endian
rpm×100 + checksum, at 115200 baud — the protocol proven in harpcontrol). `BaseMotors(left_port,
right_port)` owns both ports:

- `start()` — zero both motors and start the `harp-base-motors` writer thread.
- `command(left_rpm, right_rpm)` — set speeds *and refresh the deadman clock*. Must be called
  again within the deadman window.
- The writer loop, every 50 ms: if the last `command()` is older than `deadman_seconds` (0.25),
  send zeros (logging a warning on the transition), else re-send the current command. Its final
  act — even if `stop()` never runs because the main thread died — is a zero frame.
- `stop()` — halt the writer, zero both motors, close the ports. Serial write failures are
  logged, never raised.

Sign convention (established by the mirror-mounted right wheel, used everywhere): **forward is
`(+rpm, -rpm)`; same-sign commands spin in place**.

## `patrol.py` — the reusable lap

Blocking, thread-oriented functions run in a worker thread; everything checks `stop_event`.

- `PatrolParams` — the geometry/calibration (from `harp.yaml` `motion:`): side length, segments
  per side, speeds, and the two dead-reckoning constants `sec_per_meter` / `sec_per_90_turn`.
  Motion is **timed dead reckoning**, so laps drift with battery and floor — calibrate in place
  and keep the boundary comfortably inside the stall. `lap_seconds()` estimates a lap for the
  user/model-facing notes.
- `execute_command(motors, l, r, duration, stop_event)` — feed one command at 20 Hz for the
  duration; on stop, write an immediate zero instead of coasting into the deadman.
- `humanoid_scan()` — the lifelike pause: swivel left, look (1.5 s), swivel right, look,
  recenter, settle. The pause lengths are module constants (choreography, not calibration).
- `patrol_lap()` — four sides; per side, `segments` × (drive one segment → settle → scan), then
  a 90° corner turn. Returns False if stopped early.
- `run_patrol(motors, params, stop_event, laps)` — `laps` laps (`laps <= 0` = forever, the
  standalone CLI's mode); always leaves the motors on a zero command.

## `controller.py` — `MoveAroundController`

The bus-wired owner of the patrol. Both entry points — the live model's `move_around` tool and
the dashboard's button — land on this one controller, so a patrol can never be started twice and
everyone learns of changes the same way: a `MoveAroundChanged(active, note)` event, published
**only here** (start, stop, the bounded lap finishing on its own, or a failure).

- `start()` — under an asyncio lock (a tool call and a dashboard click arriving together can't
  both open the ports): refuse if already active ("already moving"); refuse with the `conflict`
  reason if follow mode holds the motors; open the motors off-thread (failure → `ErrorRaised` +
  `{"error": ...}`); launch `run_patrol` via `asyncio.to_thread`; publish the change; return a
  note with the estimated duration and how to stop early.
- `stop(note)` — idempotent; trips the stop event and *waits* for the patrol task, whose
  `finally` zeroes the wheels, closes the ports (releasing them for the standalone CLIs), and
  publishes `MoveAroundChanged(active=False, note=...)`.
- `snapshot()` — `{"active": bool}`, the seed for fresh dashboard connections.

The serial ports are opened when a patrol starts and released when it ends, so the standalone
motion CLIs can use the same adapters whenever a patrol isn't running.

## `follow.py` — follow-me

Drive toward a person HARP *knows*. `start()` refuses unless face-ID currently recognizes an
enrolled person, and the loop keeps requiring face-ID to vouch for that same person — the moment
an unrecognized face is what's in front (target left, or a stranger stepped closer), the wheels
zero within a pass; unseen past `follow_lost_seconds` (default 6 s) ends follow by itself.

### The steering policy — `steer()` (pure, tested)

One face box → `(left_rpm, right_rpm, driving')`, deliberately a pure function because the sign
conventions are exactly the kind of silent bug a test should pin down:

1. **Recenter first**: while the face center is outside the central box (±`follow_center_frac`
   of frame width), only spin in place toward it.
2. **Distance by face size**: the shared camera has no depth stream, so the face box *height* as
   a fraction of frame height is the distance proxy. Below `follow_far_frac` → too far, drive
   forward; at/above `follow_near_frac` → close enough, stop. The wide band between the two is
   **hysteresis** — keep doing whatever the previous tick did, so the base doesn't hunt around a
   single threshold.

### `FollowController`

Vision arrives as injected callables (`latest_frame`, `current_person`,
`person_in_front(person_id)`) so it's testable without hardware. The loop (worker thread,
~10 Hz): confirm the target via face-ID (its own 1.5 s cadence); detect the largest face with the
fast YOLOv8n-face detector (shared with the head tracker; the ONNX session is built lazily on
first start and reused); `steer()`; `motors.command()`. Every start/stop/auto-stop publishes
`FollowChanged(active, person, note)` and fires a canned status clip through the injected
`announce` callable ("I'm now following you. Please ensure a safe distance..." /
"Follow mode stopped." / "No known person detected in frame."), resolved via the status rule
book's `follow.*` moments. A `conflict` callback refuses to start while the patrol runs.

## `gimbal.py` — the servo head

Port of harpcontrol's face-tracking head node minus ROS. The ESP32 expects ASCII
`Y{yaw}P{pitch}\n` at 9600 baud. `Gimbal.track(cx, cy)` runs two PID axes (gains tuned on the
real head — "don't retune blind") on the normalized pixel error and sends clamped servo angles
(yaw 20–90°, pitch 40–70°, with a +15° pitch bias compensating the mounting angle), rate-limited
to 20 Hz. `tick()` (call ~10 Hz) runs the idle logic: after 0.5 s without a face → "face lost";
after 2 s → return to rest; after 3 s → a staged look-around sweep (max yaw → min → center →
5 s pause → repeat) until a face reappears.

## `face_tracker.py` — detection for tracking

The YOLOv8n-face ONNX model (`assets/models/yolov8n-face-lindevs.onnx`, CPU) with the classic
letterbox → decode → confidence filter → NMS pipeline. `FaceDetector.detect(frame)` returns
pixel boxes. `pick_face(boxes, depth_frame)` chooses the face to track: with RealSense depth,
the **nearest** face (median depth over a 5×5 patch at the box center); without, the largest box.
Two camera classes (`RealSenseCamera` with color+aligned depth — including a 15 fps fallback
profile for USB 2 links, with a warning — and `WebcamCamera`) plus `open_camera(backend)`
(auto → RealSense first, then webcam, None if nothing opens). Distinct from `vision/camera.py`:
this camera serves the standalone motion process and *does* use depth.

## `teleop_ps5.py` — controller teleop

pygame-based hold-to-drive: the D-pad state is polled at 50 Hz and re-sent every tick — even
`(0,0)` — which is what feeds the motors' deadman while the loop is alive; releasing the pad
stops the robot. Cross = stop, Square/Circle = speed up/down by 100 rpm. Handles both SDL
joystick layouts for Sony pads (HIDAPI: D-pad as buttons 11–14; classic DirectInput: D-pad as
hat 0), picked per controller by whether it reports a hat; survives unplug/replug.
`drive_speeds(hat_x, hat_y, ...)` preserves the hardware-proven sign mapping from harpcontrol.
`run_test()` (via `--test-controller`) prints raw button/hat events to verify the mapping.

## `__main__.py` — the standalone body runner

Phase 1 of the robot body: one process, **no event bus**, not wired into the agent. The
camera→detector→gimbal loop runs on a worker thread; PS5 teleop runs on the main thread (pygame
prefers it) feeding the motors. Every piece of hardware is optional — whatever isn't plugged in
is skipped with a warning and the rest runs. `--list-ports` prints serial ports with VID:PID and
serial numbers (Windows COM letters aren't stable across replug; this is how you tell the ESP32
from the two motor adapters). `--preview` shows a live detection window.

## `autonomous_patrol.py` — standalone patrol CLI

Runs `run_patrol` with `laps=0` (forever) in a background thread, with **any button** on a
connected PS4 controller as an E-STOP and Ctrl+C handling. Don't run it while the full agent's
motion subsystem is enabled — both want the same two serial ports.

## `tools.py` — the agent tools

`move_around` and `follow_person`, both taking a single `action: "start" | "stop"` (default
start). The handlers just route to the one controller that owns each behavior and return its
result dict — so the model, the dashboard, and the auto-stop paths can never disagree about
state. Tool wording lives in `prompts/move_around_tool.md` and `prompts/follow_tool.md` (say
something brief first, stop the moment anyone asks, never pretend to follow when the tool
errored). See [Agent tools](agent-tools.md).
