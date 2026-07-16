"""Configuration: env keys, agent prompts, per-provider session defaults, and
the user-tweakable settings file (harp.yaml).

Two layers, two audiences:
  - `.env` — secrets (API keys) and developer overrides.
  - `harp.yaml` — behavior knobs a non-developer can edit (wake thresholds,
    wake words, heartbeat). Loaded by `load_settings()`; every field has a
    default, so a missing file or a missing key never breaks anything.

Every piece of text sent to a model — personas and tool descriptions — is a
markdown file under `prompts/` (see `prompts/README.md` for the full index),
loaded here via `load_prompt()` and handed to whichever module needs it. A
missing/emptied file falls back to the `FALLBACK_*` text below it, so a typo
degrades gracefully instead of failing to start.
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
PROMPTS_DIR = REPO_ROOT / "prompts"
PROMPT_FILE = PROMPTS_DIR / "system_instructions.md"
# The filter agent's instructions (harp/voice/two_agent). Separate persona from
# the responder's: this one only relays, it never answers.
FILTER_PROMPT_FILE = PROMPTS_DIR / "filter_instructions.md"
# Tool descriptions (the text that teaches the model when/how to call each
# tool) and the two transcription-steering prompts — see prompts/README.md.
SEARCH_TOOL_PROMPT_FILE = PROMPTS_DIR / "search_knowledge_tool.md"
END_SESSION_PROMPT_FILE = PROMPTS_DIR / "end_session_tool.md"
OPENAI_TRANSCRIBE_PROMPT_FILE = PROMPTS_DIR / "transcription_openai.md"
WHISPER_PROMPT_FILE = PROMPTS_DIR / "transcription_whisper.md"
# Context delivered into the live session at open, explaining why it woke and
# (for identity) whom face-ID recognizes — see prompts/README.md.
# The parallel memory/vision helper agent (harp/memory): the summarizer's
# instruction, the wake-context writer's instruction, the describe_scene
# vision instruction, and the two tool descriptions — see prompts/README.md.
MEMORY_SUMMARIZER_PROMPT_FILE = PROMPTS_DIR / "memory_summarizer.md"
CONTEXT_WRITER_PROMPT_FILE = PROMPTS_DIR / "context_writer.md"
DESCRIBE_SCENE_PROMPT_FILE = PROMPTS_DIR / "describe_scene.md"
DESCRIBE_SCENE_TOOL_PROMPT_FILE = PROMPTS_DIR / "describe_scene_tool.md"
SEARCH_MEMORY_TOOL_PROMPT_FILE = PROMPTS_DIR / "search_memory_tool.md"
IDENTITY_CONTEXT_FILE = PROMPTS_DIR / "identity_context.md"
IDENTITY_CONTEXT_WITH_NOTES_FILE = PROMPTS_DIR / "identity_context_with_notes.md"
WAKE_CONTEXT_WAVE_FILE = PROMPTS_DIR / "wake_context_wave.md"
WAKE_CONTEXT_PUSH_TO_TALK_FILE = PROMPTS_DIR / "wake_context_push_to_talk.md"
WAKE_CONTEXT_LOUD_SOUND_FILE = PROMPTS_DIR / "wake_context_loud_sound.md"
WAKE_CONTEXT_WAKE_WORD_FILE = PROMPTS_DIR / "wake_context_wake_word.md"
SETTINGS_FILE = REPO_ROOT / "harp.yaml"
# People HARP can recognize: the folder the user curates (photos + info.yaml,
# see people/README.md) and the store enrollment builds from it (gitignored).
PEOPLE_DIR = REPO_ROOT / "people"
PEOPLE_STORE = REPO_ROOT / ".harp" / "memory" / "people"
# Long-term interaction memory (harp/memory): raw per-interaction transcripts
# the logger writes and the summarizer consumes, and the guestbook where
# memories of unrecognized visitors land (no known person to attach them to).
INTERACTIONS_DIR = REPO_ROOT / ".harp" / "memory" / "interactions"
GUESTBOOK_FILE = REPO_ROOT / ".harp" / "memory" / "guestbook.jsonl"
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
    key: str = "space"  # one key ('space', 'enter', 'm') or a '+'-combo ('ctrl+shift+m')
    # Push-to-talk ONLY: the model hears the mic exclusively while the key is
    # held — every session is gated, even ones woken hands-free (wave / wake
    # word), which otherwise stream normally.
    exclusive: bool = False
    # For hardware talk buttons that re-TAP the combo while pressed instead of
    # holding it (HARP's ESP32 BLE arcade button taps ~2.5x/s): a key-up
    # re-pressed within this window counts as one continuous hold. Must exceed
    # the button's tap gap (~0.3s for the ESP32). 0 = off (a plain keyboard).
    release_debounce_seconds: float = 0.0


@dataclass
class CameraSettings:
    """The shared camera (harp/vision/camera) feeding gestures, face-ID, and
    the memory helper's snapshots. `backend: auto` uses a RealSense's color
    stream when one is plugged in, else the webcam; pin it to `webcam` when the
    standalone motion process should keep the RealSense (one process owns it),
    or `realsense` to refuse the webcam fallback. `webcam_index` picks among
    several attached webcams."""

    backend: str = "auto"  # auto | realsense | webcam
    webcam_index: int = 0


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
    VoiceBridge runs exactly as before (still using the noise/VAD knobs below —
    see VoiceTuningSettings — on its own mic).

    `provider` picks the backend for the filter agent; empty = the same provider
    as the responder. `response_tail_seconds` is how long the filter's mic stays
    muted after the responder finishes a reply (half-duplex: stops it relaying
    the robot's own voice while the tail of the audio is still playing)."""

    enabled: bool = False
    provider: str = ""  # "" = same provider as the responder
    response_tail_seconds: float = 0.8


@dataclass
class VoiceTuningSettings:
    """Noise/VAD tuning for whichever agent currently owns the microphone: the
    plain single-agent VoiceBridge, or (when filter_agent.enabled) the two-agent
    filter's mic. Both fight the same problem — a native-audio model committing
    or hallucinating a turn from room noise — with the same knobs, all
    live-tunable on the dashboard while running (see VoiceTuning):

      - `near_field_level`: a loudness gate. Only mic audio at least this loud
        (RMS 0..1, same scale as `python -m harp.listener`) reaches the model;
        quieter room noise is replaced with silence so it never commits a turn.
        0.0 = off. Takes effect instantly.
      - `vad_threshold` / `vad_silence_ms` / `noise_reduction`: the session's
        server-VAD tuning (higher threshold + longer silence + noise reduction =
        fewer noise turns). Applied when the next conversation opens.
        `noise_reduction`: none | near_field | far_field (OpenAI; Gemini ignores it)."""

    near_field_level: float = 0.0
    vad_threshold: float = 0.5
    vad_silence_ms: int = 500
    noise_reduction: str = "none"


@dataclass
class MemorySettings:
    """Long-term interaction memory (harp/memory): a parallel Gemini Flash Lite
    helper agent that (a) summarizes each finished conversation into per-person
    memory, (b) pre-computes a wake briefing (camera frame + stored memories)
    the live session gets at open, and (c) answers the live model's
    describe_scene calls. Needs GEMINI_API_KEY in .env — without it, transcripts
    are still recorded but the helper stays offline. `calls_per_minute` is a
    hard cap shared by all three uses (free-tier quota); `context_ttl_seconds`
    is how often the wake briefing is refreshed while someone stays in frame
    (it also refreshes immediately when WHO is in frame changes)."""

    enabled: bool = True
    model: str = "gemini-3.1-flash-lite"
    calls_per_minute: int = 14
    context_ttl_seconds: float = 120.0


@dataclass
class SessionLogSettings:
    """Per-run developer log (harp/core/session_log): one JSONL timeline per
    run of the full agent — the effective settings, every bus event, every
    internal log line — for post-hoc debugging. `dir` is relative to the repo
    root; the newest `keep_runs` files are kept, older ones pruned at startup."""

    enabled: bool = True
    dir: str = ".harp/logs"
    keep_runs: int = 30


@dataclass
class Settings:
    listener: ListenerSettings = field(default_factory=ListenerSettings)
    push_to_talk: PushToTalkSettings = field(default_factory=PushToTalkSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    interaction: InteractionSettings = field(default_factory=InteractionSettings)
    heartbeat: HeartbeatSettings = field(default_factory=HeartbeatSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    filter_agent: FilterAgentSettings = field(default_factory=FilterAgentSettings)
    voice_tuning: VoiceTuningSettings = field(default_factory=VoiceTuningSettings)
    status_voice: StatusVoiceSettings = field(default_factory=StatusVoiceSettings)
    session_log: SessionLogSettings = field(default_factory=SessionLogSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)


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
        camera=_section(CameraSettings, data.get("camera")),
        interaction=_section(InteractionSettings, data.get("interaction")),
        heartbeat=_section(HeartbeatSettings, data.get("heartbeat")),
        dashboard=_section(DashboardSettings, data.get("dashboard")),
        filter_agent=_section(FilterAgentSettings, data.get("filter_agent")),
        voice_tuning=_section(VoiceTuningSettings, data.get("voice_tuning")),
        status_voice=_section(StatusVoiceSettings, data.get("status_voice")),
        session_log=_section(SessionLogSettings, data.get("session_log")),
        memory=_section(MemorySettings, data.get("memory")),
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

def load_prompt(path: Path, fallback: str) -> str:
    """Read a prompt file from `prompts/` (see `prompts/README.md`), dropping
    the markdown title and leading '> note' blockquotes — those are authoring
    notes for us, not instructions for the model. Falls back to `fallback` if
    the file is missing, unreadable, or emptied out (so a typo degrades
    gracefully instead of failing to start)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return fallback
    kept = "\n".join(
        line
        for line in raw.splitlines()
        if not line.lstrip().startswith(">") and not line.lstrip().startswith("# ")
    ).strip()
    return kept if len(kept) > 40 else fallback


FALLBACK_PERSONA = (
    "You are Laila, a friendly robot dolphin at the reception of a robotics expo. "
    "Speak out loud in short, warm, spoken sentences (no markdown). Mirror the "
    "user's language: reply in English to English, Urdu to Urdu, and match a "
    "natural Urdu/English mix. Do not speak other languages. Be honest when you "
    "don't know something. You can talk and listen but cannot move. Keep it "
    "appropriate for all ages."
)


def load_persona(path: Path = PROMPT_FILE) -> str:
    """The main responder's system instruction (prompts/system_instructions.md)."""
    return load_prompt(path, FALLBACK_PERSONA)


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
    """The two-agent filter's system instruction (prompts/filter_instructions.md)."""
    return load_prompt(path, FALLBACK_FILTER_PERSONA)


FALLBACK_SEARCH_TOOL_DESCRIPTION = (
    "Search the local knowledge base for facts before answering. It contains "
    "everything HARP has been given documents about. The documents are written "
    "in English, so always query with concise English keywords even if the "
    "visitor spoke Urdu. Call this BEFORE answering any factual question about "
    "the venue, event, schedule, or anything else the documents might cover, "
    "and base your spoken reply on what it returns. If it returns nothing "
    "useful, say you are not sure rather than guessing."
)


def load_search_tool_description(path: Path = SEARCH_TOOL_PROMPT_FILE) -> str:
    """The `search_knowledge` tool's description (prompts/search_knowledge_tool.md)."""
    return load_prompt(path, FALLBACK_SEARCH_TOOL_DESCRIPTION)


FALLBACK_END_SESSION_DESCRIPTION = (
    "End the current conversation and put HARP back on standby. Call this when "
    "the visitor says goodbye, says they're done, or asks you to stop or close "
    "the session. Say a short spoken goodbye FIRST, then call this — it hangs "
    "up immediately. Do not call it while the visitor still needs help."
)


def load_end_session_description(path: Path = END_SESSION_PROMPT_FILE) -> str:
    """The `end_session` tool's description (prompts/end_session_tool.md)."""
    return load_prompt(path, FALLBACK_END_SESSION_DESCRIPTION)


FALLBACK_OPENAI_TRANSCRIBE_PROMPT = (
    "The speaker uses only English or Urdu. Write the whole transcript in the "
    "Latin alphabet: leave English as English, and romanize spoken Urdu into "
    'Latin letters (e.g. "aap kaise hain") — never Urdu (Perso-Arabic), Arabic, '
    "or Hindi/Devanagari script."
)


def load_openai_transcribe_prompt(path: Path = OPENAI_TRANSCRIBE_PROMPT_FILE) -> str:
    """The default steering prompt for OpenAI's input transcriber
    (prompts/transcription_openai.md); the OPENAI_TRANSCRIBE_PROMPT env var
    still overrides it at runtime (see harp/voice/openai.py)."""
    return load_prompt(path, FALLBACK_OPENAI_TRANSCRIBE_PROMPT)


FALLBACK_WHISPER_PROMPT = (
    "Yeh baat-cheet Roman Urdu aur English mein likhi gayi hai. "
    "Aap kaisay hain? Main theek hoon, shukriya."
)


def load_whisper_prompt(path: Path = WHISPER_PROMPT_FILE) -> str:
    """The default initial_prompt for the local wake-word Whisper transcriber
    (prompts/transcription_whisper.md); the HARP_WHISPER_PROMPT env var still
    overrides it at runtime (see harp/listener/transcriber.py)."""
    return load_prompt(path, FALLBACK_WHISPER_PROMPT)


def format_prompt(template: str, fallback: str, **kwargs: str) -> str:
    """Fill in a `{placeholder}` template loaded via load_prompt(). If someone
    edits the file and removes/misspells a placeholder the code expects, fall
    back to `fallback` (formatted the same way) instead of crashing the
    subsystem that was about to speak."""
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        logger.warning("prompt template %r is missing a placeholder; using fallback", template)
        return fallback.format(**kwargs)


# --- Context delivered into the live session at open --------------------------
# These explain to the model why it just woke, and (for identity) who face-ID
# says is standing there. See prompts/README.md for the full list.

FALLBACK_IDENTITY_CONTEXT = "(Face recognition: you are talking to {name}.)"


def load_identity_context(path: Path = IDENTITY_CONTEXT_FILE) -> str:
    """Template for the face-ID identity line when the recognized person has
    no notes (prompts/identity_context.md). Format with `name=`."""
    return load_prompt(path, FALLBACK_IDENTITY_CONTEXT)


FALLBACK_IDENTITY_CONTEXT_WITH_NOTES = (
    "(Face recognition: you are talking to {name}. Notes about them: {notes}.)"
)


def load_identity_context_with_notes(path: Path = IDENTITY_CONTEXT_WITH_NOTES_FILE) -> str:
    """Template for the face-ID identity line when the recognized person has
    notes (prompts/identity_context_with_notes.md). Format with `name=`, `notes=`."""
    return load_prompt(path, FALLBACK_IDENTITY_CONTEXT_WITH_NOTES)


FALLBACK_WAKE_CONTEXT_WAVE = (
    "(You just woke from standby because someone waved at you. Greet them "
    "warmly and ask how you can help.)"
)


def load_wake_context_wave(path: Path = WAKE_CONTEXT_WAVE_FILE) -> str:
    """Context sent when a wave woke HARP (prompts/wake_context_wave.md)."""
    return load_prompt(path, FALLBACK_WAKE_CONTEXT_WAVE)


FALLBACK_WAKE_CONTEXT_PUSH_TO_TALK = (
    "(A person started a push-to-talk conversation: they hold a button while "
    "they speak, and are about to say something. Listen for it and answer. A "
    "very short greeting is fine, but don't give a long welcome before they "
    "have spoken.)"
)


def load_wake_context_push_to_talk(path: Path = WAKE_CONTEXT_PUSH_TO_TALK_FILE) -> str:
    """Context sent when a push-to-talk press woke HARP
    (prompts/wake_context_push_to_talk.md)."""
    return load_prompt(path, FALLBACK_WAKE_CONTEXT_PUSH_TO_TALK)


FALLBACK_WAKE_CONTEXT_LOUD_SOUND = (
    "(You just woke from standby because you heard a loud sound nearby "
    "(level {level}) — someone may be trying to get your attention. Greet "
    "them and ask how you can help.)"
)


def load_wake_context_loud_sound(path: Path = WAKE_CONTEXT_LOUD_SOUND_FILE) -> str:
    """Template for the loud-sound wake context (prompts/wake_context_loud_sound.md).
    Format with `level=`."""
    return load_prompt(path, FALLBACK_WAKE_CONTEXT_LOUD_SOUND)


FALLBACK_WAKE_CONTEXT_WAKE_WORD = (
    '(You just woke from standby because someone said: "{text}". Respond to '
    "that naturally and offer to help.)"
)


def load_wake_context_wake_word(path: Path = WAKE_CONTEXT_WAKE_WORD_FILE) -> str:
    """Template for the wake-word wake context (prompts/wake_context_wake_word.md).
    Format with `text=`."""
    return load_prompt(path, FALLBACK_WAKE_CONTEXT_WAKE_WORD)


# --- The parallel memory/vision helper agent (harp/memory) --------------------
# Prompts for the Gemini Flash Lite side-agent: summarizing finished
# conversations, writing the pre-computed wake briefing, and describing the
# camera scene mid-session. See prompts/README.md.

FALLBACK_MEMORY_SUMMARIZER_PROMPT = (
    "You are the memory-keeper for Laila, a robot receptionist. A conversation "
    "just ended; turn it into a memory Laila can use the next time she meets "
    "these visitors.\n\nFacts extracted from the interaction:\n{facts}\n\n"
    "Transcript:\n{transcript}\n\n"
    'Reply with ONLY a JSON object with these keys: "summary" (2-4 plain '
    "sentences: who the visitors were, what they asked or talked about, and how "
    'it was resolved), "follow_up" (one sentence describing anything left open '
    "that Laila should follow up on if she sees them again — e.g. they were "
    'looking for something — or "" if nothing is open), and "person_facts" '
    "(things the visitors said about themselves worth remembering — name, role, "
    'affiliation, interests — or "" if none). Write in English regardless of '
    "the conversation language. Base everything strictly on the transcript; do "
    "not invent details."
)


def load_memory_summarizer_prompt(path: Path = MEMORY_SUMMARIZER_PROMPT_FILE) -> str:
    """Template for the end-of-interaction memory summarizer
    (prompts/memory_summarizer.md). Format with `facts=`, `transcript=`."""
    return load_prompt(path, FALLBACK_MEMORY_SUMMARIZER_PROMPT)


FALLBACK_CONTEXT_WRITER_PROMPT = (
    "You brief Laila, a robot receptionist, just before she starts a "
    "conversation. The attached photo is what her camera sees right now. Below "
    "is what her long-term memory holds about the people her face recognition "
    "identified in frame:\n\n{people}\n\n"
    "Write the briefing Laila will read as the conversation opens: 2-4 short "
    "plain sentences covering who is there (use the photo for how many people "
    "and anything notable about the scene), what she remembers about them, and "
    "any open follow-up she should raise. If nobody is recognized, describe "
    "what the photo shows so she can greet them naturally. Address Laila as "
    '"you". No markdown, no preamble — just the briefing.'
)


def load_context_writer_prompt(path: Path = CONTEXT_WRITER_PROMPT_FILE) -> str:
    """Template for the pre-computed wake briefing (prompts/context_writer.md).
    Format with `people=`."""
    return load_prompt(path, FALLBACK_CONTEXT_WRITER_PROMPT)


FALLBACK_DESCRIBE_SCENE_PROMPT = (
    "The attached photo is what a robot receptionist's camera sees right now, "
    "mid-conversation. Describe it for her in 2-4 short spoken-style sentences "
    "she can relay to the visitor: who and what is visible, and anything "
    "notable. Pay particular attention to: {focus}. No markdown, no preamble."
)


def load_describe_scene_prompt(path: Path = DESCRIBE_SCENE_PROMPT_FILE) -> str:
    """The vision instruction sent (with the camera frame) to the helper model
    when the live agent calls describe_scene (prompts/describe_scene.md).
    Format with `focus=`."""
    return load_prompt(path, FALLBACK_DESCRIBE_SCENE_PROMPT)


FALLBACK_DESCRIBE_SCENE_TOOL_DESCRIPTION = (
    "Look through your camera and get a fresh description of the current "
    "scene: how many people are there, what they look like or are holding, and "
    "anything else visible. Call it when a visitor asks what you can see, who "
    "is around, or about something they are showing you. It takes a moment, so "
    "say something brief first. Base your answer on what it returns."
)


def load_describe_scene_tool_description(
    path: Path = DESCRIBE_SCENE_TOOL_PROMPT_FILE,
) -> str:
    """The `describe_scene` tool's description (prompts/describe_scene_tool.md)."""
    return load_prompt(path, FALLBACK_DESCRIBE_SCENE_TOOL_DESCRIPTION)


FALLBACK_SEARCH_MEMORY_TOOL_DESCRIPTION = (
    "Search your long-term memory of past conversations and the people you "
    "have met. Call it when a visitor asks whether you remember them or "
    "someone else, refers to an earlier visit or conversation, or when knowing "
    "your history with a person would help. Query with a few English keywords "
    "(a name, a topic). If it returns nothing, say honestly that you don't "
    "remember."
)


def load_search_memory_tool_description(
    path: Path = SEARCH_MEMORY_TOOL_PROMPT_FILE,
) -> str:
    """The `search_memory` tool's description (prompts/search_memory_tool.md)."""
    return load_prompt(path, FALLBACK_SEARCH_MEMORY_TOOL_DESCRIPTION)


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


def build_session_config(
    provider: str, tuning: "VoiceTuning | None" = None
) -> SessionConfig:
    """Assemble a provider-agnostic SessionConfig for the chosen backend.
    `tuning` (when given) stamps the current noise/VAD knobs onto the session —
    used both for the plain single-agent bridge and, via build_filter_config,
    the two-agent filter — so a fresh call at each session open picks up
    whatever the dashboard currently has set."""
    try:
        d = _DEFAULTS[provider]
    except KeyError:
        raise ValueError(f"unknown provider: {provider!r}") from None
    up = provider.upper()
    config = SessionConfig(
        system_instruction=load_persona(),
        model=_env_first(d["model_env"], d["model"]),
        voice=_env_first(d["voice_env"], d["voice"]),
        # Left unset by default so native-audio auto-detects language.
        language=os.getenv("HARP_LANGUAGE") or None,
        input_rate=int(os.getenv(f"{up}_INPUT_RATE", d["input_rate"])),
        output_rate=int(os.getenv(f"{up}_OUTPUT_RATE", d["output_rate"])),
    )
    if tuning is not None:
        config.vad_threshold = tuning.vad_threshold
        config.vad_silence_ms = tuning.vad_silence_ms
        config.noise_reduction = tuning.noise_reduction if tuning.noise_reduction != "none" else None
    return config


_NOISE_REDUCTION_CHOICES = ("none", "near_field", "far_field")


@dataclass
class VoiceTuning:
    """Live-adjustable noise/VAD knobs, shared by whichever agent currently owns
    the microphone — the plain single-agent VoiceBridge, or the two-agent
    filter — and the dashboard. Seeded from VoiceTuningSettings; the dashboard
    mutates it at runtime (via `apply`) while you tune against the real room.

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


def build_filter_config(provider: str, tuning: "VoiceTuning | None" = None) -> SessionConfig:
    """A SessionConfig for the filter agent (harp/voice/two_agent): the responder
    defaults for `provider` (including `tuning`, stamped by build_session_config),
    but with the filter persona and NO tools — it only relays what it hears, it
    never retrieves or answers."""
    config = build_session_config(provider, tuning)
    config.system_instruction = load_filter_persona()
    config.tools = []
    return config


def require_key(name: str) -> str:
    """Return an env value or raise a clean error (no traceback) if missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to harp/.env (copy .env.example) and try again."
        )
    return value
