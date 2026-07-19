# Wake listener (`harp/listener/`)

[← Back to index](index.md)

The always-on listener owns the microphone while HARP is idle and decides when to wake it. Two
wake paths: a **loud sound** (level above `wake_level` — a clap, a raised voice) wakes
immediately with no words needed; **speech** (level above `transcribe_level`) is captured as a
phrase, transcribed locally, and matched against the configured wake words.

| File | Role |
|---|---|
| `detector.py` | Pure decision logic: PCM chunks in, `LoudSound` / `Phrase` / `None` out; plus wake-word matching |
| `transcriber.py` | Local faster-whisper wrapper for phrase transcription |
| `listener.py` | The bus-wired subsystem: mic + detector + transcriber + wake publishing |
| `__main__.py` | `python -m harp.listener` — the calibration tool |

## `detector.py` — pure wake decisions

No microphone, no bus, no model — just arithmetic on PCM chunks, so it is unit-testable with
synthetic audio. Timing derives from sample counts, not wall clocks, keeping tests deterministic.

- `rms_level(chunk)` — the loudness of one 16-bit chunk, normalized to 0.0–1.0 of int16 full
  scale. This same function defines the scale used by `wake_level`, `transcribe_level`, and the
  voice loudness gate's `near_field_level`, so one calibration session covers them all.
- `WakeDetector.feed(chunk)` — the state machine:
  - level ≥ `wake_level` → reset and return `LoudSound(level)` immediately;
  - idle + level ≥ `transcribe_level` → start capturing, prepending up to 3 chunks of kept
    **pre-roll** so the first syllable of "hello" isn't clipped;
  - while capturing, accumulate; when consecutive quiet reaches `silence_seconds` or total
    capture reaches `max_phrase_seconds`, return `Phrase(pcm)` (the joined buffer) and reset;
  - otherwise return `None`.
- `match_wake_word(transcript, wake_words)` — returns the first configured wake word found in
  the transcript, or `None`. Whole-word matching only ("hi" cannot match inside "this"),
  punctuation/case-insensitive, supports multi-word entries and non-Latin script (it tokenizes
  with a Unicode `\w+` regex and does a padded-substring check).

## `transcriber.py` — local Whisper

`PhraseTranscriber(model_name)` wraps faster-whisper, loaded **lazily** — importing the module is
cheap; the model (~75–150 MB depending on `listener.whisper_model`: tiny/base/small) is pulled in
on the first real transcription. Runs on CPU in int8; a 2–4 s phrase transcribes in well under a
second.

Two deliberate details:

- **Offline-first load**: once cached, the model is loaded with `local_files_only=True`, because
  huggingface_hub otherwise makes a HEAD request to re-validate the cache on *every* load — which
  stalls for tens of seconds on a flaky connection and looks like a hang. Only a genuinely-first
  use falls back to a networked download.
- **Romanized-Urdu steering**: the `initial_prompt` (from `prompts/transcription_whisper.md`,
  overridable with `HARP_WHISPER_PROMPT`) nudges Whisper to emit spoken Urdu in Latin script —
  otherwise it produces Perso-Arabic, which the romanized wake words in `harp.yaml` ("salam",
  "assalam") could never match. The prompt intentionally contains **no wake word**, so if Whisper
  ever regurgitates its prompt on silence it cannot cause a false wake.

## `listener.py` — the subsystem

`AlwaysOnListener(bus, settings)` ties it together. `run()`:

- Lazily builds the real transcriber (injectable for tests).
- Watches `StateChanged` on the bus: it listens **only while the app is in STANDBY**. On any
  other state it closes its mic stream (the live voice session needs the device) and reopens it
  when STANDBY returns. Run standalone with no orchestrator publishing states, it simply listens
  forever — which is what the calibration tool relies on.
- The mic loop: for each chunk, feed the detector and handle the decision:
  - `LoudSound` → format the loud-sound wake context template (`{level}` filled in) and publish
    `WakeRequested(reason="loud sound", context=...)`.
  - `Phrase` → transcribe **off-thread** (`asyncio.to_thread`), match wake words, publish
    `PhraseHeard(text, wake_word)` either way (so the dashboard shows what the ears picked up and
    why it did or didn't wake), and if a word matched, publish
    `WakeRequested(reason="wake word", context=...)` with the transcribed phrase folded into the
    template.
- After requesting a wake it sleeps 2 s (`_COOLDOWN_SECONDS`) so it can't fire twice before the
  orchestrator's `StateChanged` pauses it properly.
- **Mic resilience**: a `PortAudioError`/`OSError` (another app holds the device, OS privacy
  settings block microphone access — common on Windows — or a USB device dropped) is caught,
  warned about, and retried every 5 s rather than killing the agent.

Note the layering: in exclusive push-to-talk mode this subsystem is *not started at all* — that
decision is made in `app.py`, since every wake it could request would be vetoed anyway.

## `__main__.py` — the calibration tool

`uv run python -m harp.listener` shows a live level meter (a bar with `T` and `W` markers at the
two thresholds from `harp.yaml`) and runs the real detector + Whisper pipeline, printing exactly
what would wake HARP: `>> LOUD SOUND (level 0.31) → would wake HARP` or
`heard: "hello there" → would wake HARP (wake word "hello")`. Adjust `wake_level` /
`transcribe_level` in `harp.yaml` until: quiet room ≈ below both, normal speech ≈ above
`transcribe_level`, raised voice/clap ≈ above `wake_level`. The same 0–1 RMS scale applies to the
voice-tuning loudness gate, so this tool doubles as its calibrator.
