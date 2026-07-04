"""One-shot generator for HARP's canned status voice lines.

Renders the pre-programmed status phrases (boot, connectivity, error
narration — see orchestrator/status_voice.py) to WAV files with the Kokoro
offline TTS engine, plus a manifest.json that maps stable line ids to files.
`status_voice.play()` reads the manifest at runtime; this script runs ONCE
(or whenever the phrase set changes) and its output is committed to the repo.

Kokoro lives in a separate venv, so run with THAT interpreter, not harp's:

    /home/mani/Repos/Latex/.venv-tts/bin/python scripts/generate_status_voice.py

Output:
    assets/status_voice/en/<line_id>.wav   (24 kHz mono 16-bit PCM)
    assets/status_voice/manifest.json

Kokoro has no Urdu voice; the manifest is keyed id -> lang so an "ur" set
(e.g. Kokoro's Hindi voice over transliterated text, or another engine) can
be added later without touching callers.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24_000  # Kokoro's fixed output rate
LANG = "en"
LANG_CODE = "a"  # Kokoro pipeline code: American English
VOICE = "af_heart"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "assets" / "status_voice"

# id -> spoken text. Ids are the stable API; text can be re-worded freely.
# Chosen to cover every point where the orchestrator speaks without a live
# model: boot, connectivity, error narration + retry, sleep/wake, shutdown.
LINES: dict[str, str] = {
    "starting_up": "Starting up.",
    "connection_established": "Connection established.",
    "ready": "I'm ready.",
    "listening": "I'm listening.",
    "one_moment": "One moment, please.",
    "no_internet": "I can't reach the internet right now.",
    "connection_lost": "I lost my connection. Let me try again.",
    "retrying": "Something went wrong. Retrying.",
    "error_recoverable": "I ran into a problem. Give me a moment.",
    "error_fatal": "I couldn't recover from an error, and I have to shut down.",
    "mic_problem": "I'm having trouble with my microphone.",
    "going_standby": "Going on standby.",
    "session_ended": "Goodbye.",
    "shutting_down": "Shutting down. Goodbye.",
}


def main() -> None:
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=LANG_CODE)
    lang_dir = OUT_DIR / LANG
    lang_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines: dict[str, dict] = {}
    for line_id, text in LINES.items():
        chunks = [audio for _, _, audio in pipeline(text, voice=VOICE)]
        audio = np.concatenate(chunks)
        wav_path = lang_dir / f"{line_id}.wav"
        sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
        duration = round(len(audio) / SAMPLE_RATE, 2)
        manifest_lines[line_id] = {
            LANG: {
                "text": text,
                "file": f"{LANG}/{line_id}.wav",
                "voice": VOICE,
                "duration_s": duration,
            }
        }
        print(f"  {line_id:24s} {duration:5.2f}s  {text}")

    manifest = {
        "generated": date.today().isoformat(),
        "generator": "scripts/generate_status_voice.py",
        "engine": "kokoro (hexgrad/Kokoro-82M)",
        "sample_rate": SAMPLE_RATE,
        "format": "wav, mono, 16-bit PCM",
        "lines": manifest_lines,
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"\nWrote {len(LINES)} lines + {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
