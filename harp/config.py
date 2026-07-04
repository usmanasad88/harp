"""Configuration: env keys, the persona, per-provider session defaults, and the
user-tweakable settings file (harp.yaml).

Two layers, two audiences:
  - `.env` — secrets (API keys) and developer overrides.
  - `harp.yaml` — behavior knobs a non-developer can edit (wake thresholds,
    wake words, heartbeat). Loaded by `load_settings()`; every field has a
    default, so a missing file or a missing key never breaks anything.

The persona lives in `prompts/system_instructions.md` (single source of truth,
shared with the web-realtime sandbox). We strip the authoring notes so the model
gets clean instructions.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .voice.provider import SessionConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = REPO_ROOT / "prompts" / "system_instructions.md"
SETTINGS_FILE = REPO_ROOT / "harp.yaml"
# People HARP can recognize: the folder the user curates (photos + info.yaml,
# see people/README.md) and the store enrollment builds from it (gitignored).
PEOPLE_DIR = REPO_ROOT / "people"
PEOPLE_STORE = REPO_ROOT / ".harp" / "memory" / "people"
# The knowledge corpus search_knowledge retrieves from (see harp/knowledge).
DATA_DIR = REPO_ROOT / "data"

# Load harp/.env once, on import. Values already in the shell env win.
load_dotenv(REPO_ROOT / ".env")


# --- harp.yaml: user-tweakable settings ---------------------------------------
@dataclass
class ListenerSettings:
    """The always-on wake listener (harp/listener). Levels are 0.0–1.0 RMS."""

    enabled: bool = True
    wake_level: float = 0.25       # this loud wakes HARP by itself
    transcribe_level: float = 0.04  # this loud starts the wake-word transcriber
    wake_words: list[str] = field(
        default_factory=lambda: [
            "hey", "hi", "hello", "laila", "harp", "salam", "salaam", "assalam",
        ]
    )
    max_phrase_seconds: float = 5.0
    silence_seconds: float = 0.8
    whisper_model: str = "base"


@dataclass
class HeartbeatSettings:
    """Liveness signal for the watchdog (a file whose mtime = last heartbeat)."""

    file: str = ".harp/heartbeat"
    interval_seconds: float = 5.0


@dataclass
class DashboardSettings:
    """The dev dashboard (harp/dashboard). `bind: localhost` keeps it reachable
    only from this machine; `bind: network` exposes it to your phone/other PCs
    on the same Wi-Fi/LAN at this machine's LAN IP (printed at startup)."""

    bind: str = "localhost"
    port: int = 8787


@dataclass
class Settings:
    listener: ListenerSettings = field(default_factory=ListenerSettings)
    heartbeat: HeartbeatSettings = field(default_factory=HeartbeatSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)


def _section(cls, raw: object):
    """Build a settings dataclass from one YAML section, keeping defaults for
    missing keys and warning about (not crashing on) unknown ones."""
    if not isinstance(raw, dict):
        return cls()
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - names)
    if unknown:
        logger.warning("harp.yaml: ignoring unknown %s keys: %s", cls.__name__, unknown)
    return cls(**{k: v for k, v in raw.items() if k in names})


def load_settings(path: Path = SETTINGS_FILE) -> Settings:
    """Read harp.yaml; a missing file or missing keys just mean defaults."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        data = {}
    if not isinstance(data, dict):
        logger.warning("harp.yaml: expected a mapping at the top level; using defaults")
        data = {}
    return Settings(
        listener=_section(ListenerSettings, data.get("listener")),
        heartbeat=_section(HeartbeatSettings, data.get("heartbeat")),
        dashboard=_section(DashboardSettings, data.get("dashboard")),
    )


_DASHBOARD_BIND_HOSTS = {"localhost": "127.0.0.1", "network": "0.0.0.0"}


def dashboard_bind_host(bind: str) -> str:
    """Map harp.yaml's `dashboard.bind` (localhost|network) to a socket bind
    address; an unrecognized value warns and falls back to localhost-only."""
    host = _DASHBOARD_BIND_HOSTS.get(bind)
    if host is None:
        logger.warning(
            "harp.yaml: dashboard.bind=%r not one of %s; defaulting to localhost",
            bind, sorted(_DASHBOARD_BIND_HOSTS),
        )
        return _DASHBOARD_BIND_HOSTS["localhost"]
    return host

FALLBACK_PERSONA = (
    "You are Laila, a friendly robot dolphin at the reception of a robotics expo. "
    "Speak out loud in short, warm, spoken sentences (no markdown). Mirror the "
    "user's language: reply in English to English, Urdu to Urdu, and match a "
    "natural Urdu/English mix. Do not speak other languages. Be honest when you "
    "don't know something. You can talk and listen but cannot move. Keep it "
    "appropriate for all ages."
)


def load_persona(path: Path = PROMPT_FILE) -> str:
    """Read the system prompt, dropping the markdown title and the leading
    '> note' blockquotes (those are notes for us, not for the model)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return FALLBACK_PERSONA
    kept = "\n".join(
        line
        for line in raw.splitlines()
        if not line.lstrip().startswith(">") and not line.lstrip().startswith("# ")
    ).strip()
    return kept if len(kept) > 40 else FALLBACK_PERSONA


# Per-provider defaults. Gemini sends mic at 16 kHz and returns 24 kHz; OpenAI
# Realtime uses 24 kHz both ways.
#
# `model_env` / `voice_env` list the env vars consulted, in order. For OpenAI we
# honor the *same* names the web-realtime sandbox uses (REALTIME_MODEL /
# REALTIME_VOICE) — both read this one harp/.env, so setting REALTIME_MODEL there
# switches the model for the sandbox AND this app at once (handy for testing on
# a mini model). The OPENAI_* names still work and win if both are set.
_DEFAULTS = {
    "gemini": {
        "model": "gemini-3.1-flash-live-preview", "voice": "Kore",
        "model_env": ("GEMINI_MODEL",), "voice_env": ("GEMINI_VOICE",),
        "input_rate": 16000, "output_rate": 24000,
    },
    "openai": {
        "model": "gpt-realtime-2", "voice": "marin",
        "model_env": ("OPENAI_MODEL", "REALTIME_MODEL"),
        "voice_env": ("OPENAI_VOICE", "REALTIME_VOICE"),
        "input_rate": 24000, "output_rate": 24000,
    },
}


def _env_first(names: tuple[str, ...], default: str) -> str:
    """Return the first non-empty env var among `names`, else `default`."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def build_session_config(provider: str) -> SessionConfig:
    """Assemble a provider-agnostic SessionConfig for the chosen backend."""
    try:
        d = _DEFAULTS[provider]
    except KeyError:
        raise ValueError(f"unknown provider: {provider!r}") from None
    up = provider.upper()
    return SessionConfig(
        system_instruction=load_persona(),
        model=_env_first(d["model_env"], d["model"]),
        voice=_env_first(d["voice_env"], d["voice"]),
        # Left unset by default so native-audio auto-detects language.
        language=os.getenv("HARP_LANGUAGE") or None,
        input_rate=int(os.getenv(f"{up}_INPUT_RATE", d["input_rate"])),
        output_rate=int(os.getenv(f"{up}_OUTPUT_RATE", d["output_rate"])),
    )


def require_key(name: str) -> str:
    """Return an env value or raise a clean error (no traceback) if missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to harp/.env (copy .env.example) and try again."
        )
    return value
