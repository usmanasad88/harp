"""The provider abstraction: one interface, many backends.

Every backend (Gemini Live now, OpenAI Realtime next, Moshi later) speaks the
same small vocabulary so the rest of HARP is written once:

  - it accepts a `SessionConfig`,
  - it takes microphone audio (and, later, camera frames / text) *in*,
  - it emits a stream of normalized `VoiceEvent`s *out*.

Keeping this vocabulary provider-agnostic is the whole point of the class:
swapping Gemini for OpenAI must not ripple into audio I/O, RAG, or the
orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


# --- What a session is configured with --------------------------------------
@dataclass
class SessionConfig:
    """Provider-agnostic knobs for one live voice session."""

    system_instruction: str
    voice: str
    model: str
    # BCP-47 language hint, e.g. "ur-IN". None = let the model auto-detect,
    # which is the reliable path for native-audio models (see PLAN.md).
    language: str | None = None
    input_rate: int = 16000  # sample rate of the mic audio we send (Hz)
    output_rate: int = 24000  # sample rate of the audio the model returns (Hz)
    # Seam for later: tool/function declarations exposed to the model (RAG,
    # web search). Empty in the vanilla providers; wired when tools land.
    tools: list[Any] = field(default_factory=list)
    # Voice-activity-detection tuning. Mainly for the two-agent filter, which
    # must NOT commit a turn on room noise; each backend maps these onto its own
    # VAD config. None = leave the provider's default.
    vad_threshold: float | None = None       # 0..1; higher = needs louder/clearer speech
    vad_silence_ms: int | None = None        # trailing silence (ms) that ends a turn
    noise_reduction: str | None = None       # "near_field" | "far_field" | None


# --- Normalized events a provider emits -------------------------------------
@dataclass
class AudioOut:
    """A chunk of spoken audio: raw 16-bit PCM, mono, at output_rate."""

    pcm: bytes


@dataclass
class UserTranscript:
    """What the model heard the user say (may arrive in streaming pieces)."""

    text: str
    final: bool = False


@dataclass
class AgentTranscript:
    """What the model is saying, as text (streaming pieces)."""

    text: str
    final: bool = False


@dataclass
class ToolCall:
    """The model is asking us to run a tool. Seam for RAG/web-search later."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TurnComplete:
    """The model finished its turn."""


@dataclass
class Interrupted:
    """Barge-in: the user spoke over the model, so stop playing its audio."""


@dataclass
class ProviderError:
    """The backend reported (or raised) an error; the loop can decide what to do."""

    message: str


VoiceEvent = (
    AudioOut
    | UserTranscript
    | AgentTranscript
    | ToolCall
    | TurnComplete
    | Interrupted
    | ProviderError
)


# --- The interface itself ----------------------------------------------------
@runtime_checkable
class VoiceConnection(Protocol):
    """A live, open session with one backend."""

    async def send_audio(self, pcm: bytes) -> None:
        """Send a chunk of mic audio (16-bit PCM mono at input_rate)."""

    async def send_image(self, jpeg: bytes, mime_type: str = "image/jpeg") -> None:
        """Send a camera frame (vision arrives in a later phase)."""

    async def send_text(self, text: str) -> None:
        """Inject a text message into the conversation."""

    async def respond_tool(self, call: "ToolCall", output: Any) -> None:
        """Return a tool's result to the model. Seam; unused in vanilla."""

    async def interrupt(self) -> None:
        """Tell the model to stop talking (barge-in is usually automatic via VAD)."""

    def events(self) -> AsyncIterator[VoiceEvent]:
        """Async-iterate the normalized events from this session."""


class VoiceProvider(Protocol):
    """A backend factory. `connect` is an async context manager yielding a
    `VoiceConnection` and tearing it down on exit."""

    name: str

    def connect(self, config: SessionConfig):  # -> async context manager
        ...
