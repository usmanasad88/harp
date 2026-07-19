# Configuration

[← Back to index](index.md)

HARP has three configuration surfaces, each with a different audience:

1. **`.env`** — secrets (API keys) and developer overrides. Loaded once at import of
   `harp/config.py`; values already in the shell environment win over the file.
2. **`harp.yaml`** — behavior knobs a non-developer can edit (thresholds, timeouts, toggles,
   ports). Loaded by `load_settings()`; **every field has a default**, so a missing file or a
   missing key never breaks anything, and unknown keys produce a warning, not a crash.
3. **`prompts/`** — every piece of text sent to any model, as plain markdown files.

All three are implemented in `harp/config.py`, which is also home to the per-provider session
defaults and two small live-state classes the dashboard mutates at runtime.

## `.env` — environment variables

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | OpenAI Realtime provider (required when `--provider openai`) |
| `GEMINI_API_KEY` | Gemini Live provider AND the memory helper (needed even when the voice provider is OpenAI, if you want summaries/briefings/describe_scene) |
| `HARP_PROVIDER` | default provider for the CLI (`openai` for `python -m harp`, `gemini` for `python -m harp.app`) |
| `OPENAI_MODEL` / `REALTIME_MODEL` | override the OpenAI realtime model (default `gpt-realtime-2`). The `REALTIME_*` names are shared with the `web-realtime/` sandbox so one `.env` switches both |
| `OPENAI_VOICE` / `REALTIME_VOICE` | override the OpenAI voice (default `marin`) |
| `GEMINI_MODEL` / `GEMINI_VOICE` | override the Gemini model (default `gemini-3.1-flash-live-preview`) / voice (default `Kore`) |
| `OPENAI_INPUT_RATE`, `OPENAI_OUTPUT_RATE`, `GEMINI_INPUT_RATE`, `GEMINI_OUTPUT_RATE` | override the audio sample rates |
| `HARP_LANGUAGE` | pin a BCP-47 language; unset = native-audio auto-detect (recommended) |
| `OPENAI_TRANSCRIBE_MODEL` | the OpenAI input-transcription model (default `gpt-4o-mini-transcribe`) |
| `OPENAI_TRANSCRIBE_PROMPT` | override the romanization steering prompt for OpenAI transcription |
| `HARP_WHISPER_PROMPT` | override the local Whisper `initial_prompt` for wake-word transcription |

`require_key(name)` is the accessor that raises a clean, traceback-free error telling you to add
the key to `.env`.

## `harp.yaml` — section by section

Each section maps to a dataclass in `config.py`; `_section()` builds it keeping defaults for
missing keys and warning on unknown ones. Edit and restart to apply.

### `listener:` → `ListenerSettings`
The always-on wake listener. `enabled`; `wake_level` (0–1 RMS — this loud wakes by itself; 1.0
disables loudness-only wake); `transcribe_level` (this loud starts phrase capture); `wake_words`
(romanized list); `max_phrase_seconds` / `silence_seconds` (phrase capture bounds);
`whisper_model` (tiny/base/small). Calibrate with `python -m harp.listener`.

### `push_to_talk:` → `PushToTalkSettings`
`enabled` (arms the key — hands-free wake still works); `key` (one key or a `+`-combo);
`exclusive` (the button becomes the whole interface: listener not started, non-button wakes
vetoed, mic heard only while held); `release_debounce_seconds` (for hardware buttons that re-tap
instead of holding — HARP's ESP32 arcade button needs ~0.7; 0 for a real keyboard);
`idle_prompt_seconds` (how often the idle invite clip replays; 0 = never).

### `camera:` → `CameraSettings`
`backend`: `auto` (RealSense color stream if plugged in, else webcam), `realsense` (only), or
`webcam` (pin it here when the standalone motion process should keep the RealSense — one process
owns it). `webcam_index` / `usb_webcam_index` tell the dashboard dropdown which OS device index
is "the laptop's own" vs "the USB one". This is only the startup default — the dashboard switches
sources live.

### `motion:` → `MotionSettings`
`enabled` (false on a dev machine = no tool, no button, ports never touched); `left_port` /
`right_port` (find with `python -m harp.motion --list-ports`); patrol geometry and calibration
(`base_speed`, `turn_speed`, `side_length`, `segments`, `sec_per_meter`, `sec_per_90_turn`,
`laps`); and the follow-me knobs (`follow_speed`, `follow_turn_speed`, `follow_far_frac` /
`follow_near_frac` — the face-size distance band, with the gap between them acting as
hysteresis, `follow_center_frac` — the no-turn central box, `follow_lost_seconds`).

### `interaction:` → `InteractionSettings`
`absence_timeout_seconds` (no face this long ends the session; face-ID doubles as presence) and
`silence_timeout_seconds` (nothing said in either direction this long also ends it; 0 disables).

### `filter_agent:` → `FilterAgentSettings`
The experimental two-agent noise filter. `enabled`; `provider` ("" = same as the responder);
`response_tail_seconds` (how long the filter's mic stays muted after a reply finishes, covering
audio still draining from the speaker).

### `voice_tuning:` → `VoiceTuningSettings`
Boot defaults for the live-tunable noise/VAD knobs (the dashboard sliders override at runtime;
copy good values back here to persist): `near_field_level` (the loudness gate, 0 = off, same RMS
scale as the listener tool; takes effect instantly), `vad_threshold`, `vad_silence_ms`, and
`noise_reduction` (none / near_field / far_field — OpenAI only). The VAD/noise fields are baked
into the session config at open, so a change applies to the *next* conversation.

### `status_voice:` → `StatusVoiceSettings`
`enabled` (false = run silently) and `lang` (only `en` ships). Which clip plays when is the rule
book (`orchestrator/status_rules.py`); the words are `scripts/generate_status_voice.py`.

### `heartbeat:` → `HeartbeatSettings`
`file` (touched every beat, for the external watchdog) and `interval_seconds`.

### `memory:` → `MemorySettings`
`enabled`; `model` (default `gemini-3.1-flash-lite`); `calls_per_minute` (the hard cap shared by
summarizer + briefing + describe_scene — the free-tier budget); `context_ttl_seconds` (wake
briefing refresh period while someone stays in frame).

### `session_log:` → `SessionLogSettings`
`enabled`, `dir` (default `.harp/logs`), `keep_runs` (older run files pruned at startup).

### `dashboard:` → `DashboardSettings`
`bind` (`localhost` or `network` — mapped by `dashboard_bind_host()` to 127.0.0.1 / 0.0.0.0, an
unrecognized value warns and falls back to localhost); `port`; `open_browser`.

## The prompts system

Every piece of text sent to a model — personas, tool descriptions, and the "you just woke up
because…" context — lives in `prompts/` as markdown, loaded through `load_prompt(path,
fallback)`. Design points:

- **Authoring notes are stripped**: the markdown `# title` line and any `>` blockquote lines are
  removed before the model sees the text — they are notes for humans.
- **Graceful degradation**: a missing, unreadable, or emptied-out file (fewer than 40 chars
  kept) falls back to a hardcoded `FALLBACK_*` constant next to its loader in `config.py`, so a
  typo degrades instead of failing to start.
- **Templates**: the context prompts contain `{placeholders}` (`{name}`, `{text}`, `{level}`,
  `{facts}`, `{transcript}`, `{people}`, `{focus}`…). `format_prompt(template, fallback,
  **kwargs)` fills them; if someone edits a file and removes a placeholder the code expects, it
  falls back to the built-in default for that one message rather than crashing the subsystem.

There is one loader function per file (`load_persona`, `load_filter_persona`,
`load_search_tool_description`, `load_end_session_description`, the wake-context loaders, the
memory-prompt loaders, …). The full file-by-file index — which agent/tool each file drives —
is in `prompts/README.md` and summarized in [Data, prompts & on-disk state](data-and-state.md).

Also defined here: `IGNORE_SENTINEL = "[[ignore]]"`, the token the filter agent's persona must
output for audio that should not be relayed (stripped by `filter_agent.clean_relay`).

## Provider session defaults — `build_session_config()`

The `_DEFAULTS` table holds, per provider: default model and voice, which env vars override them
(in priority order), and the audio rates (Gemini 16 kHz in / 24 kHz out; OpenAI 24/24).
`build_session_config(provider, tuning)` assembles a `SessionConfig` from persona + resolved
model/voice/rates, and — when a `VoiceTuning` is passed — stamps the current VAD/noise knobs onto
it. It is called fresh at every session open so the dashboard's current values apply to the next
conversation. `build_filter_config()` produces the filter agent's variant: same defaults, the
filter persona, and **no tools** (it only relays; it never retrieves or answers).

## Live-state classes (mutated by the dashboard)

- **`VoiceTuning`** — the runtime copy of the tuning knobs, seeded from `VoiceTuningSettings` in
  `app.py`. `apply(field, value)` validates and clamps each knob (levels to 0–1, silence to
  100–3000 ms, noise reduction to the fixed choice list) and returns the full new snapshot —
  raising `ValueError` on garbage, so a bad dashboard message can never wedge a nonsense setting.
  `near_field_level` is read per mic chunk (instant); the VAD fields apply at next session open.
- **`CameraSourceState`** — the runtime camera-source selection. `select(source)` validates one
  of `auto | realsense | webcam | usb_webcam` and maps it to the `(backend, device)` pair for
  `Camera.request_switch` — `usb_webcam` is HARP-side sugar for "webcam, but the other device
  index" (the camera itself has no notion of laptop vs USB). It never touches the camera itself,
  keeping the class testable without hardware.

## Paths

`config.py` also centralizes the repo-anchored paths: `PROMPTS_DIR`, `SETTINGS_FILE`
(`harp.yaml`), `DATA_DIR` (`data/`), `PEOPLE_DIR` (`people/`), `PEOPLE_STORE`
(`.harp/memory/people/`), `INTERACTIONS_DIR`, `GUESTBOOK_FILE`, and `STATUS_VOICE_DIR`
(`assets/status_voice/`).
