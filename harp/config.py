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
# The filter agent's instructions (harp/voice/two_agent). Separate persona from
# the responder's: this one only relays, it never answers.
FILTER_PROMPT_FILE = REPO_ROOT / "prompts" / "filter_instructions.md"
SETTINGS_FILE = REPO_ROOT / "harp.yaml"
# People HARP can recognize: the folder the user curates (photos + info.yaml,
# see people/README.md) and the store enrollment builds from it (gitignored).
PEOPLE_DIR = REPO_ROOT / "people"
PEOPLE_STORE = REPO_ROOT / ".harp" / "memory" / "people"
# The knowledge corpus search_knowledge retrieves from (see harp/knowledge).
DATA_DIR = REPO_ROOT / "data"
# Canned status clips + manifest (see harp/orchestrator/status_voice) that play
# during boot / outages / errors, when the cloud voice is unavailable.
STATUS_VOICE_DIR = REPO_ROOT / "assets" / "status_voice"

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
class PushToTalkSettings:
    """Hold-a-key-to-talk mode (harp/interaction/push_to_talk). Meant for noisy
    rooms: the mic only reaches the model while the key is held, and the first
    press wakes HARP. When enabled, the always-on wake listener is not started.
    Set both in harp.yaml (`push_to_talk:`)."""

    enabled: bool = False
    key: str = "space"  # 'space', 'enter', or a single character


@dataclass
class InteractionSettings:
    """When a live session should close on its own (harp/interaction)."""

    # No face in frame for this many continuous seconds ends the session. Face-
    # ID doubles as presence; a returning face resets the countdown.
    absence_timeout_seconds: float = 10.0


@dataclass
class HeartbeatSettings:
    """Liveness signal for the watchdog (a file whose mtime = last heartbeat)."""

    file: str = ".harp/heartbeat"
    interval_seconds: float = 5.0


@dataclass
class StatusVoiceSettings:
    """Pre-recorded status announcements (harp/orchestrator/status_voice): the
    boot, connectivity, standby, error, and shutdown lines that play without the
    cloud model. Set `enabled: false` in harp.yaml to run silently; `lang` picks
    the manifest language (only 'en' ships today)."""

    enabled: bool = True
    lang: str = "en"


@dataclass
class DashboardSettings:
    """The dev dashboard (harp/dashboard). `bind: localhost` keeps it reachable
    only from this machine; `bind: network` exposes it to your phone/other PCs
    on the same Wi-Fi/LAN at this machine's LAN IP (printed at startup)."""

    bind: str = "localhost"
    port: int = 8787


@dataclass
class FilterAgentSettings:
    """Two-agent noise/intent filtering (harp/voice/two_agent). When enabled, a
    first realtime agent hears the room, discards noise / background chatter /
    speech not meant for HARP, and relays ONLY the intended message (as text) to
    the second agent — the normal responder — which replies with voice. It's
    an opt-in front-end for loud rooms; when disabled, the single-agent
    VoiceBridge runs exactly as before.

    `provider` picks the backend for the filter agent; empty = the same provider
    as the responder. `response_tail_seconds` is how long the filter's mic stays
    muted after the responder finishes a reply (half-duplex: stops it relaying
    the robot's own voice while the tail of the audio is still playing).

    The remaining knobs fight the filter's biggest failure — a native-audio model
    hallucinating a message from room noise — and are all live-tunable on the
    dashboard (see FilterTuning):
      - `near_field_level`: a loudness gate. Only mic audio at least this loud
        (RMS 0..1, same scale as `python -m harp.listener`) reaches the filter;
        quieter room noise is replaced with silence so it never commits a turn.
        0.0 = off. Takes effect instantly.
      - `vad_threshold` / `vad_silence_ms` / `noise_reduction`: the filter
        session's server-VAD tuning (higher threshold + longer silence + noise
        reduction = fewer noise turns). Applied when the next conversation opens.
        `noise_reduction`: none | near_field | far_field (OpenAI; Gemini ignores it)."""

    enabled: bool = False
    provider: str = ""  # "" = same provider as the responder
    response_tail_seconds: float = 0.8
    near_field_level: float = 0.0
    vad_threshold: float = 0.5
    vad_silence_ms: int = 500
    noise_reduction: str = "none"


@dataclass
class Settings:
    listener: ListenerSettings = field(default_factory=ListenerSettings)
    push_to_talk: PushToTalkSettings = field(default_factory=PushToTalkSettings)
    interaction: InteractionSettings = field(default_factory=InteractionSettings)
    heartbeat: HeartbeatSettings = field(default_factory=HeartbeatSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    filter_agent: FilterAgentSettings = field(default_factory=FilterAgentSettings)
    status_voice: StatusVoiceSettings = field(default_factory=StatusVoiceSettings)


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
        push_to_talk=_section(PushToTalkSettings, data.get("push_to_talk")),
        interaction=_section(InteractionSettings, data.get("interaction")),
        heartbeat=_section(HeartbeatSettings, data.get("heartbeat")),
        dashboard=_section(DashboardSettings, data.get("dashboard")),
        filter_agent=_section(FilterAgentSettings, data.get("filter_agent")),
        status_voice=_section(StatusVoiceSettings, data.get("status_voice")),
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


# The filter agent's job in one place, so it works even without the prompt file.
# It must relay ONLY the intended message and emit IGNORE_SENTINEL for anything
# else — see harp/voice/filter_agent.py, which drops the sentinel.
IGNORE_SENTINEL = "[[ignore]]"

FALLBACK_FILTER_PERSONA = (
    "You are a listening filter in front of a voice assistant in a loud, crowded "
    "room. You do not answer or chat. Decide whether the audio is a person "
    "speaking TO the assistant. If it is, repeat ONLY their intended message as a "
    "single clean sentence, in the same language they spoke (English or Urdu; "
    "romanize Urdu). Fix disfluencies but do not add or answer anything. If the "
    "audio is background conversation, crowd noise, someone talking to another "
    "person, the assistant's own voice, or nothing meant for the assistant, "
    f"reply with exactly {IGNORE_SENTINEL} and nothing else. Lines beginning with "
    f"'CONTEXT:' are notes about what the assistant just said — never relay them; "
    f"reply {IGNORE_SENTINEL}."
)


def load_filter_persona(path: Path = FILTER_PROMPT_FILE) -> str:
    """Read the filter agent's instructions (same '> note'/title stripping as
    load_persona); fall back to the built-in text if the file is missing."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return FALLBACK_FILTER_PERSONA
    kept = "\n".join(
        line
        for line in raw.splitlines()
        if not line.lstrip().startswith(">") and not line.lstrip().startswith("# ")
    ).strip()
    return kept if len(kept) > 40 else FALLBACK_FILTER_PERSONA


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


_NOISE_REDUCTION_CHOICES = ("none", "near_field", "far_field")


@dataclass
class FilterTuning:
    """Live-adjustable filter knobs, shared by the two-agent bridge and the
    dashboard. Seeded from FilterAgentSettings; the dashboard mutates it at
    runtime (via `apply`) while you tune against the real room.

    `near_field_level` is read per mic chunk, so it takes effect instantly. The
    VAD/noise fields are baked into the session config at open, so a change
    applies to the NEXT conversation. `apply` validates + clamps every value, so
    a bad message from the dashboard can never wedge a nonsense setting in."""

    near_field_level: float = 0.0
    vad_threshold: float = 0.5
    vad_silence_ms: int = 500
    noise_reduction: str = "none"

    def snapshot(self) -> dict:
        return dataclasses.asdict(self)

    def apply(self, field: str, value: object) -> dict:
        """Set one knob from a dashboard message; return the full new snapshot.
        Raises ValueError on an unknown field or an unusable value."""
        if field in ("near_field_level", "vad_threshold"):
            self.__dict__[field] = _clamp(float(value), 0.0, 1.0)  # type: ignore[arg-type]
        elif field == "vad_silence_ms":
            self.vad_silence_ms = int(_clamp(float(value), 100, 3000))
        elif field == "noise_reduction":
            if value not in _NOISE_REDUCTION_CHOICES:
                raise ValueError(f"noise_reduction must be one of {_NOISE_REDUCTION_CHOICES}")
            self.noise_reduction = str(value)
        else:
            raise ValueError(f"unknown filter tuning field: {field!r}")
        return self.snapshot()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_filter_config(provider: str, tuning: "FilterTuning | None" = None) -> SessionConfig:
    """A SessionConfig for the filter agent (harp/voice/two_agent): the responder
    defaults for `provider`, but with the filter persona and NO tools — it only
    relays what it hears, it never retrieves or answers. `tuning` (when given)
    stamps the current VAD/noise-reduction knobs onto the session, so a fresh
    call at each session open picks up whatever the dashboard set."""
    config = build_session_config(provider)
    config.system_instruction = load_filter_persona()
    config.tools = []
    if tuning is not None:
        config.vad_threshold = tuning.vad_threshold
        config.vad_silence_ms = tuning.vad_silence_ms
        config.noise_reduction = tuning.noise_reduction if tuning.noise_reduction != "none" else None
    return config


def require_key(name: str) -> str:
    """Return an env value or raise a clean error (no traceback) if missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to harp/.env (copy .env.example) and try again."
        )
    return value
