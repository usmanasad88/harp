"""The shared event vocabulary that travels on the bus.

These dataclasses are the ONLY thing subsystems agree on. When two subsystems
need to interact, they do it by publishing/subscribing one of these — never by
importing each other. Keep them small, provider-agnostic, and free of behavior.

Note the deliberate overlap with `voice/provider.py`'s VoiceEvent: those are the
low-level events of a single live voice session; the ones here are the
system-wide bus vocabulary. The orchestrator translates the former into the
latter (a provider `UserTranscript` becomes a bus `UserSaid`, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    """Base type for everything on the bus (lets subscribers filter by type)."""


# --- lifecycle / state -------------------------------------------------------
@dataclass
class StateChanged(Event):
    """The orchestrator moved between app states (see core/state.py)."""

    old: str
    new: str


@dataclass
class ShutdownRequested(Event):
    """Something asked HARP to shut down cleanly."""

    reason: str = ""


# --- presence & identity (presence / vision subsystems) ----------------------
@dataclass
class PresenceChanged(Event):
    """Webcam presence: is anyone in frame, and roughly how many."""

    present: bool
    count: int = 0


@dataclass
class PersonIdentified(Event):
    """Face-ID resolved who is in frame (or flagged them new/unknown)."""

    person_id: str
    name: str | None = None
    is_known: bool = False
    confidence: float = 0.0


@dataclass
class GestureDetected(Event):
    """A recognized gesture, e.g. a wave — a proactive cue for triggers."""

    kind: str


# --- wake / end requests (what the orchestrator acts on) ---------------------
@dataclass
class PhraseHeard(Event):
    """The idle-time wake listener transcribed a phrase. `wake_word` is the
    matched word (a WakeRequested follows) or None (heard, but no wake) —
    published either way so observers can see what HARP's ears picked up."""

    text: str
    wake_word: str | None = None


@dataclass
class WakeRequested(Event):
    """Something wants a live session opened (only honored while STANDBY).
    `reason` = what asked: loud sound / wake word / wave / button / follow-up.
    `context` = model-facing text explaining the wake-up (e.g. the transcribed
    greeting) — the orchestrator hands it to the live session at open."""

    reason: str
    context: str = ""


@dataclass
class EndOfInteractionDetected(Event):
    """The end-rules monitor judged the conversation over; the orchestrator
    should close the session (ACTIVE → STANDBY)."""

    reason: str


# --- interaction / voice -----------------------------------------------------
@dataclass
class InteractionStarted(Event):
    """A live voice session opened. `reason` = what woke it
    (loud sound / wake word / wave / button / follow-up); `context` = the
    model-facing wake-up explanation that was passed to the session."""

    reason: str
    context: str = ""


@dataclass
class InteractionEnded(Event):
    """The live voice session closed. `reason` = why
    (left frame + silence / error / shutdown)."""

    reason: str


@dataclass
class UserSaid(Event):
    """What HARP heard the user say (final or a streaming piece)."""

    text: str
    final: bool = False


@dataclass
class AgentSaid(Event):
    """What HARP said back (final or a streaming piece)."""

    text: str
    final: bool = False


@dataclass
class TalkKeyChanged(Event):
    """The push-to-talk key's *effective* hold flipped. Debounce-bridged: a
    hardware button whose firmware re-taps while pressed (see
    interaction/push_to_talk.py) reads as ONE hold, so this never flickers
    mid-train. The end-user page renders it (green "Listening" while held)."""

    held: bool


# --- memory (harp/memory: summarizer + context writer) ------------------------
@dataclass
class MemoryWritten(Event):
    """The summarizer turned a finished interaction into long-term memory.
    `person_ids` are the enrolled people it was attached to — empty means the
    visitors were unknown and the memory went to the guestbook instead."""

    person_ids: list[str]
    summary: str
    follow_up: str = ""


@dataclass
class ContextPrepared(Event):
    """The context writer pre-computed a wake briefing (camera frame + stored
    memories) for whoever is currently in frame, ready to hand to the live
    session at the next open. `people` are the person_ids it covers."""

    people: list[str]
    text: str


# --- tools (RAG / web-search bridge) -----------------------------------------
@dataclass
class ToolRequested(Event):
    """The model asked for a tool; mirrored on the bus for the dashboard."""

    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class ToolCompleted(Event):
    """A tool finished; its output is on its way back to the model."""

    id: str
    output: object = None


# --- audio control (dashboard mic-mute button) --------------------------------
@dataclass
class MicMuteChanged(Event):
    """The OS-level mic mute flipped (dashboard button, or anything else that
    calls audio_control.set_mic_muted). `muted=True` silences the physical
    input device itself (via the system mixer) — every subsystem reading the
    mic, now or later, is affected with no code changes on its side."""

    muted: bool


@dataclass
class VoiceTuningChanged(Event):
    """The live noise/VAD tuning changed (a dashboard slider, or the seeded
    startup value) for whichever agent currently owns the microphone — the
    single-agent VoiceBridge, or the two-agent filter. Carries the full current
    snapshot so every open tab stays in sync. See harp/config.VoiceTuning."""

    near_field_level: float
    vad_threshold: float
    vad_silence_ms: int
    noise_reduction: str


# --- health & errors ---------------------------------------------------------
@dataclass
class Heartbeat(Event):
    """Emitted periodically so the watchdog knows the agent is still alive."""

    ts: float


@dataclass
class ErrorRaised(Event):
    """A subsystem failed. `fatal` decides whether the orchestrator narrates +
    retries (non-fatal) or heads for STOPPING (fatal)."""

    where: str
    message: str
    fatal: bool = False
