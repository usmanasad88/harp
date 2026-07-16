"""Bridge one live voice session onto the bus — what ACTIVE actually runs.

This turns the standalone voice core (harp/voice) into a supervised
subsystem: the orchestrator starts `VoiceBridge.run()` when a session opens
and cancels it when the session closes. While running, the bridge:

  - opens the provider connection + mic + speaker (the same runner shape as
    session.py, which remains the bare `python -m harp` path);
  - delivers the wake-up context, plus who face-ID says is standing there,
    to the model at session start — so it knows why it woke and to whom it's
    talking;
  - translates the session's VoiceEvents into bus events: UserTranscript →
    UserSaid, AgentTranscript → AgentSaid, ToolCall → ToolRequested +
    ToolCompleted (running the tool dispatcher in between and returning the
    result to the model), ProviderError → ErrorRaised — which is how the
    dashboard sees the conversation and how the orchestrator's error/backoff
    handling gets triggered;
  - plays the model's audio and clears the speaker on barge-in;
  - optionally gates the mic through a `LoudnessGate` (see loudness_gate.py) —
    the same proximity/noise gate the two-agent filter uses — so room noise
    below `near_field_level` never reaches the model as real audio. Live-tuned
    from the dashboard alongside the filter's copy (harp/config.VoiceTuning).

Composition stays out of here: which provider, which tools, and whose
identity callable are all injected by app.py (and by tests, which is why the
mic/speaker classes are constructor parameters too).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable

from ..core.bus import Bus
from ..core.events import (
    AgentSaid,
    ErrorRaised,
    ToolCompleted,
    ToolRequested,
    UserSaid,
)
from . import get_provider
from .audio_io import Microphone, Speaker
from .loudness_gate import LoudnessGate
from .provider import (
    AgentTranscript,
    AudioOut,
    Interrupted,
    ProviderError,
    SessionConfig,
    ToolCall,
    UserTranscript,
)

logger = logging.getLogger(__name__)

ToolDispatch = Callable[[str, dict], Awaitable[Any]]


def gated_mic_payload(pcm: bytes, mic_gate: Callable[[], bool] | None) -> bytes:
    """The audio to actually send for one mic chunk: the real audio while the
    gate is open (or absent), else same-length digital silence.

    Sending silence rather than nothing keeps a continuous stream, so the
    provider's own voice-activity detection still sees the trailing quiet and
    ends the turn — without any of the room's background noise ever reaching
    it. Shared by the bridge's push-to-talk gate, the filter agent's
    half-duplex gate, and the offline recorder (spike_ptt_gate.py), so what the
    recorder saves is byte-identical to what a live session would send."""
    if mic_gate is None or mic_gate():
        return pcm
    return bytes(len(pcm))


class VoiceBridge:
    def __init__(
        self,
        bus: Bus,
        provider_name: str,
        make_config: Callable[[], SessionConfig],
        tool_dispatch: ToolDispatch | None = None,
        identity_context: Callable[[], str] | None = None,
        mic_gate: Callable[[], bool] | None = None,
        near_field_level: Callable[[], float] | None = None,
        text_inbox: "asyncio.Queue[str] | None" = None,
        provider=None,
        mic_factory: Callable[[int], Microphone] = Microphone,
        speaker_factory: Callable[[int], Speaker] = Speaker,
    ) -> None:
        self._bus = bus
        self._provider_name = provider_name
        self._make_config = make_config
        self._tool_dispatch = tool_dispatch
        self._identity_context = identity_context
        # Push-to-talk gate: when it returns False, the mic is "released" and we
        # feed the model digital silence instead of what the room sounds like.
        # None = ungated (stream the mic straight through, the default).
        self._mic_gate = mic_gate
        # Live loudness-gate threshold (0..1 RMS; 0/None = off). Applied AFTER
        # the push-to-talk gate, same as the filter agent's mic pump — a
        # callable so a dashboard change takes effect on the next chunk.
        self._near_field_level = near_field_level
        # Text-driven mode (the two-agent responder): when a queue is given, this
        # session opens NO microphone and instead speaks in response to text turns
        # pushed onto the queue (the filter agent's relayed messages). Everything
        # else — tools, transcripts, speaker, barge-in — is unchanged. None = the
        # normal mic-driven session.
        self._text_inbox = text_inbox
        self._provider = provider  # tests inject a fake; None = the real backend
        self._mic_factory = mic_factory
        self._speaker_factory = speaker_factory

    async def run(self, context: str = "") -> None:
        """Run one live session until cancelled (or the provider ends it).

        `context` is the wake-up explanation from WakeRequested; it is sent
        into the session at open, together with the identity line, so the
        model starts the conversation knowing why and with whom.
        """
        provider = self._provider or get_provider(self._provider_name)
        config = self._make_config()
        text_driven = self._text_inbox is not None
        logger.info(
            "voice session opening (%s, %s%s)",
            self._provider_name, config.model, ", text-driven" if text_driven else "",
        )
        async with contextlib.AsyncExitStack() as stack:
            conn = await stack.enter_async_context(provider.connect(config))
            speaker = await stack.enter_async_context(self._speaker_factory(config.output_rate))
            # In text-driven mode the responder has no mic of its own — the filter
            # agent owns the only microphone — so we don't open one here.
            mic = (
                None if text_driven
                else await stack.enter_async_context(self._mic_factory(config.input_rate))
            )
            opening = self._opening_text(context)
            if opening:
                await conn.send_text(opening)
            input_task = asyncio.create_task(
                self._pump_text(conn) if mic is None else self._pump_mic(mic, conn)
            )
            try:
                await self._pump_events(conn, speaker)
            finally:
                input_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await input_task
                logger.info("voice session closed")

    async def _pump_text(self, conn) -> None:
        """Text-driven input: forward each relayed message to the model as a user
        turn. Used only in two-agent mode; the filter agent fills the queue."""
        assert self._text_inbox is not None
        while True:
            text = await self._text_inbox.get()
            await conn.send_text(text)

    def _opening_text(self, context: str) -> str:
        parts = [context]
        if self._identity_context is not None:
            parts.append(self._identity_context())
        return "\n".join(part for part in parts if part).strip()

    async def _pump_mic(self, mic: Microphone, conn) -> None:
        # Two gates in series, same as the filter agent's mic pump: the hard
        # push-to-talk gate first (silence outright when not held/ungated),
        # then the loudness gate on whatever it lets through.
        gate = LoudnessGate(self._near_field_level or (lambda: 0.0))
        async for pcm in mic.chunks():
            await conn.send_audio(gate.process(self._mic_payload(pcm)))

    def _mic_payload(self, pcm: bytes) -> bytes:
        """Real mic audio while push-to-talk is held (or ungated), else
        same-length digital silence — see gated_mic_payload."""
        return gated_mic_payload(pcm, self._mic_gate)

    async def _pump_events(self, conn, speaker: Speaker) -> None:
        """Consume the session's events until its stream ends."""
        async for ev in conn.events():
            if isinstance(ev, AudioOut):
                speaker.play(ev.pcm)
            elif isinstance(ev, AgentTranscript):
                await self._bus.publish(AgentSaid(text=ev.text, final=ev.final))
            elif isinstance(ev, UserTranscript):
                await self._bus.publish(UserSaid(text=ev.text, final=ev.final))
            elif isinstance(ev, Interrupted):
                # Barge-in: the model already stopped; drop its unplayed audio.
                speaker.clear()
            elif isinstance(ev, ToolCall):
                await self._handle_tool(conn, ev)
            elif isinstance(ev, ProviderError):
                # The orchestrator reacts to this: close session, back off.
                await self._bus.publish(
                    ErrorRaised(where="voice.provider", message=ev.message)
                )
            # TurnComplete needs no bus mirror: the final transcript pieces
            # already close the turns for transcript consumers.

    async def _handle_tool(self, conn, call: ToolCall) -> None:
        await self._bus.publish(
            ToolRequested(id=call.id, name=call.name, arguments=call.arguments)
        )
        if self._tool_dispatch is None:
            output: Any = {"error": f"no tool handler wired for {call.name}"}
        else:
            try:
                output = await self._tool_dispatch(call.name, call.arguments)
            except Exception as exc:
                # A failing tool degrades into the model apologizing, not a
                # dead session.
                logger.exception("tool %s failed", call.name)
                output = {"error": str(exc)}
        await self._bus.publish(ToolCompleted(id=call.id, output=output))
        await conn.respond_tool(call, output)
