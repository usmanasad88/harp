"""Agent 1 of the two-agent pipeline: the listening filter.

This is the front half of harp/voice/two_agent. It opens its own live session
whose ONLY job is to decide what the visitor actually wants to say to HARP and
relay that — as clean text — to the responder. It never speaks to the visitor.

How it works, and why it's shaped this way:

  - It hears the room through the microphone (native audio in), the same as the
    normal bridge, but it has NO speaker: its spoken output is thrown away. We
    read the model's *transcript* (AgentTranscript) as the relayed message, so
    no provider changes are needed — every backend already emits the transcript
    of what it "says". (Switching the filter to a text-only output modality is a
    later cost/latency optimization; it would feed this same relay path.)
  - The filter persona (prompts/filter_instructions.md) tells it to output the
    intended message, or the sentinel `[[ignore]]` for noise / background / the
    assistant's own voice / a CONTEXT note. We strip the sentinel and relay only
    what's left; an all-sentinel (or empty) turn relays nothing.
  - `add_context(text)` feeds the responder's last reply back in as a
    `CONTEXT:` note so short follow-ups ("yes", "how much") make sense. The
    persona absorbs CONTEXT lines and answers `[[ignore]]`, so this uses the
    ordinary text channel — no special provider method.
  - `mic_gate` is the half-duplex gate the coordinator owns: while the responder
    is speaking it returns False and we feed the model digital silence, so the
    filter never relays the robot talking to itself (same silence-substitution
    trick push-to-talk uses in the bridge).

Everything network-/hardware-shaped is injected so this is testable with a fake
provider and fake mic, exactly like VoiceBridge.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections import deque
from typing import Awaitable, Callable

import numpy as np

from ..config import IGNORE_SENTINEL
from ..core.bus import Bus
from ..core.events import ErrorRaised
from . import get_provider
from .audio_io import Microphone
from .provider import (
    AgentTranscript,
    ProviderError,
    SessionConfig,
    UserTranscript,
)

logger = logging.getLogger(__name__)

RelayCallback = Callable[[str], Awaitable[None]]

# Match the sentinel however the model punctuates/spaces it ("[[ ignore ]]",
# "[[IGNORE]].") so a stray period or capital never leaks a fake user turn.
_SENTINEL_RE = re.compile(r"\[\[\s*ignore\s*\]\]", re.IGNORECASE)


def _rms(pcm: bytes) -> float:
    """Loudness of one PCM chunk, 0.0–1.0 of int16 full scale. Identical formula
    to listener.detector.rms_level, so the dashboard's near_field_level calibrates
    against the same numbers `python -m harp.listener` shows."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((samples / 32768.0) ** 2)))


class LoudnessGate:
    """A loudness/proximity gate for the filter's mic: only audio at or above a
    live threshold passes; the quiet room becomes digital silence, so the filter
    never commits a turn on ambient noise. A short pre-roll (so a word's onset
    isn't clipped when speech crosses the threshold) and a hangover (so a brief
    dip mid-word doesn't chop the turn) keep it usable rather than choppy.

    The threshold is read fresh every chunk via `level()`, so dashboard changes
    take effect instantly. `level() <= 0` disables the gate (everything passes)."""

    def __init__(
        self,
        level: Callable[[], float],
        preroll_chunks: int = 3,
        hangover_chunks: int = 8,
    ) -> None:
        self._level = level
        self._preroll: deque[bytes] = deque(maxlen=max(1, preroll_chunks))
        self._hangover_max = max(1, hangover_chunks)
        self._hangover = 0

    def process(self, pcm: bytes) -> bytes:
        threshold = self._level()
        if threshold <= 0.0:
            return pcm  # gate disabled: pass the room through untouched
        if _rms(pcm) >= threshold:
            self._hangover = self._hangover_max
            if self._preroll:  # speech onset: prepend the buffered lead-in once
                lead_in = b"".join(self._preroll)
                self._preroll.clear()
                return lead_in + pcm
            return pcm
        if self._hangover > 0:  # trailing quiet within a word — keep passing
            self._hangover -= 1
            return pcm
        self._preroll.append(pcm)   # below threshold: hold as possible lead-in
        return bytes(len(pcm))       # ...and send silence in its place


def clean_relay(text: str) -> str:
    """The message to actually relay from one filter turn: the transcript minus
    any ignore-sentinels, trimmed. Empty string means "relay nothing".

    A turn with no alphanumeric content left (the sentinel plus a stray period,
    a lone comma, whitespace) relays nothing too — otherwise "[[ignore]]." would
    leak "." as a fake user turn."""
    stripped = _SENTINEL_RE.sub("", text).strip()
    if not any(ch.isalnum() for ch in stripped):
        return ""
    return stripped


class FilterAgent:
    def __init__(
        self,
        bus: Bus,
        provider_name: str,
        make_config: Callable[[], SessionConfig],
        on_relay: RelayCallback,
        mic_gate: Callable[[], bool] | None = None,
        near_field_level: Callable[[], float] | None = None,
        provider=None,
        mic_factory: Callable[[int], Microphone] = Microphone,
    ) -> None:
        self._bus = bus
        self._provider_name = provider_name
        self._make_config = make_config
        self._on_relay = on_relay
        self._mic_gate = mic_gate
        # Live loudness-gate threshold (0..1 RMS; 0 = off). A callable so the
        # dashboard can move it mid-session and the next chunk honors it.
        self._near_field_level = near_field_level
        self._provider = provider
        self._mic_factory = mic_factory
        # Context notes waiting to be sent to the live session (the responder's
        # replies, fed back so follow-ups make sense). Filled by add_context,
        # drained by _pump_context once the session is open.
        self._context_inbox: asyncio.Queue[str] = asyncio.Queue()

    def add_context(self, text: str) -> None:
        """Queue a note about what the responder just said. Sent to the filter as
        a `CONTEXT:` line, which the persona absorbs (it replies `[[ignore]]`)."""
        if text.strip():
            self._context_inbox.put_nowait(text.strip())

    async def run(self, context: str = "") -> None:
        """Run the filter session until cancelled (or the provider ends it)."""
        provider = self._provider or get_provider(self._provider_name)
        config = self._make_config()
        logger.info("filter session opening (%s, %s)", self._provider_name, config.model)
        async with (
            provider.connect(config) as conn,
            self._mic_factory(config.input_rate) as mic,
        ):
            if context.strip():
                # Seed the filter with why it woke / who's here, as context it
                # must not relay.
                await conn.send_text(self._context_note(context.strip()))
            mic_task = asyncio.create_task(self._pump_mic(mic, conn))
            ctx_task = asyncio.create_task(self._pump_context(conn))
            try:
                await self._pump_events(conn)
            finally:
                for task in (mic_task, ctx_task):
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                logger.info("filter session closed")

    @staticmethod
    def _context_note(text: str) -> str:
        return f"CONTEXT: {text}"

    async def _pump_mic(self, mic: Microphone, conn) -> None:
        # Two gates in series: the half-duplex gate (hard-mute while the responder
        # speaks) then the loudness gate (drop quiet ambient). The loudness gate
        # also sees the half-duplex silence, so its pre-roll/hangover never carry
        # stale audio across a reply.
        loudness = LoudnessGate(self._near_field_level or (lambda: 0.0))
        async for pcm in mic.chunks():
            await conn.send_audio(loudness.process(self._mic_payload(pcm)))

    def _mic_payload(self, pcm: bytes) -> bytes:
        """Real mic audio when the gate is open (or ungated), else same-length
        digital silence — so the room's noise, and the robot's own voice while
        it's replying, never reach the filter, but the provider VAD still sees a
        continuous stream. Same substitution the bridge uses for push-to-talk."""
        if self._mic_gate is None or self._mic_gate():
            return pcm
        return bytes(len(pcm))

    async def _pump_context(self, conn) -> None:
        while True:
            note = await self._context_inbox.get()
            await conn.send_text(self._context_note(note))

    async def _pump_events(self, conn) -> None:
        """Accumulate each spoken turn's transcript and relay the cleaned message
        when the turn finalizes. Audio, tool calls, interruptions are ignored —
        the filter has no speaker and no tools.

        Two things are logged at INFO to make false relays debuggable: what the
        filter's ASR *heard* (input transcription) and its *raw output* (relay
        vs. the ignore sentinel). A native-audio model hallucinates greetings on
        silence/faint noise, so seeing both localizes where a bogus turn came
        from — the audio it committed, or the model inventing one."""
        said = ""   # the model's output for this turn = the relay decision
        heard = ""  # the input-audio transcription = what its ASR thinks it heard
        async for ev in conn.events():
            if isinstance(ev, AgentTranscript):
                said += ev.text
                if ev.final:
                    await self._finish_turn(said)
                    said = ""
            elif isinstance(ev, UserTranscript):
                heard += ev.text
                if ev.final:
                    if heard.strip():
                        logger.info("filter heard (asr): %s", heard.strip())
                    heard = ""
            elif isinstance(ev, ProviderError):
                await self._bus.publish(
                    ErrorRaised(where="voice.filter", message=ev.message)
                )

    async def _finish_turn(self, transcript: str) -> None:
        raw = transcript.strip()
        relay = clean_relay(transcript)
        if relay:
            logger.info("filter relaying: %s", relay)
            await self._on_relay(relay)
        elif raw:
            # Heard something and chose to drop it (noise/background/the ignore
            # sentinel). Logged at INFO so you can watch suppression working and
            # catch cases where it SHOULD have ignored but relayed instead.
            logger.info("filter ignored (output: %r)", raw)
