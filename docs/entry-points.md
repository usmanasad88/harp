# Entry points & running HARP

[← Back to index](index.md)

The project uses [uv](https://docs.astral.sh/uv/); `uv run` installs dependencies from the
lockfile on first use. The first run also downloads models: the Whisper wake model (~75–150 MB),
the MediaPipe gesture model (~8 MB), and InsightFace's buffalo_l bundle (~350 MB).

## The main entry points

### `uv run python -m harp` — the full supervised agent

`harp/__main__.py`. Default provider **openai** (override with `--provider gemini` or the
`HARP_PROVIDER` env var). Runs `harp.app.run_app`: orchestrator + real voice session + wake
listener + camera/gestures/face-ID + memory + dashboard (+ motion when enabled), all on one bus.
Prints the knowledge chunk count at startup. Ctrl+C to quit.

The heavy vision imports are deferred: `--voice-only` never touches cv2/insightface/mediapipe.

### `uv run python -m harp --voice-only` — the bare voice core

No bus, orchestrator, dashboard, listener, or vision — just mic → provider → speaker with
`search_knowledge` wired, printing transcripts to the terminal. The fast smoke test of the
provider/audio path (`harp/voice/session.py`).

### `uv run python -m harp.app` — the same full agent, direct

`app.py`'s own `main()` — identical to `python -m harp` except its default provider is
**gemini**.

## Subsystem CLIs

| Command | What it does |
|---|---|
| `uv run python -m harp.listener` | Wake-listener calibration: a live RMS meter with the two thresholds marked, plus the real detector + Whisper so you see exactly what would wake HARP. Tune `listener.wake_level` / `transcribe_level` in `harp.yaml` against it |
| `uv run python -m harp.dashboard` | The dashboard against a fresh empty bus — every panel shows its honest "nothing yet" state; the way to verify the dashboard itself |
| `uv run python -m harp.motion --list-ports` | List serial ports with VID:PID + serial numbers (tell the ESP32 head apart from the two motor adapters) |
| `uv run python -m harp.motion --gimbal-port COM5 --left-port COM6 --right-port COM7` | The standalone robot body: face tracking drives the head, a PS4/PS5 pad drives the wheels (hold D-pad to move, Cross = stop, Square/Circle = speed). Every piece of hardware optional. `--preview` shows the detection window; `--test-controller` prints raw button events |
| `uv run python -m harp.motion.autonomous_patrol --left-port COM4 --right-port COM5 ...` | Standalone perimeter patrol, laps forever, any controller button = E-STOP. Don't run while the agent's motion subsystem is enabled (same ports) |

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `enroll_people.py` | Build the face-ID store from `people/<person-id>/` folders (info.yaml + 3–5 photos, exactly one face per photo; group shots skipped). Re-running re-enrolls from current photos but preserves accumulated interaction summaries. `--only <id>` for one person |
| `generate_status_voice.py` | One-shot render of the canned status lines to `assets/status_voice/en/*.wav` + `manifest.json`, using the Kokoro offline TTS. Runs from a **separate venv** (`.venv-tts` — torch would bloat HARP's); output is committed to the repo. Edit its `LINES` dict to re-word clips |
| `scrape_site.py` | Crawl a website's same-domain content pages into clean markdown files in `data/` for the RAG corpus. Polite: one domain, skips asset/API/form paths, pauses between requests |
| `preview_camera.py` | Grab one frame from the real webcam and save it — capture sanity check |
| `preview_face_id.py` | One frame + detection + recognition against the enrolled store, boxes labeled with name + similarity — the tool for calibrating the 0.4 match threshold |
| `preview_gestures.py` | Run the real MediaPipe recognizer against the webcam and print what would have been published |

## Spikes and references (not the product)

- `spike_gemini_voice.py` — the original throwaway Gemini Live proof-of-concept.
- `spike_ptt_gate.py` — records exactly what a gated session would send (used to measure the
  ESP32 button's tap timing with `--debug-keys`).
- `web-realtime/` — the browser sandbox where the OpenAI Realtime session shape was proven; the
  Python OpenAI backend mirrors its `server.js` config, and they share the `REALTIME_*` env vars.

## A typical expo-day run

1. `.env` has `OPENAI_API_KEY` and `GEMINI_API_KEY`.
2. `harp.yaml` (as committed): push-to-talk **exclusive** on `ctrl+shift+m` with 0.7 s debounce
   (the ESP32 arcade button), idle invite every 45 s, motion enabled on COM4/COM5, dashboard
   bound to the network.
3. `uv run python -m harp` — the dashboard opens; the robot says "Starting up." then
   "Connection established."; a second screen shows `http://<ip>:8787/user` full-screen.
4. Visitors hold the green button to talk; sessions end via goodbye/silence/walk-off; memories
   accumulate under `.harp/memory/`; every run leaves a replayable timeline in `.harp/logs/`.
