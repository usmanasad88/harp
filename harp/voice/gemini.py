"""Gemini Live backend for the provider abstraction.

Ports the working spike into the shared interface: same v1beta channel and
native-audio model, but now it emits normalized `VoiceEvent`s instead of
printing, and it reports interruptions and transcripts so the rest of HARP
(and, later, the dashboard) can consume them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from ..config import require_key
from .provider import (
    AgentTranscript,
    AudioOut,
    Interrupted,
    ProviderError,
    SessionConfig,
    ToolCall,
    TurnComplete,
    UserTranscript,
    VoiceEvent,
)

logger = logging.getLogger(__name__)

# Sentinel pushed onto the event queue when the receive loop ends.
_DONE = object()


def _activity_detection(config: SessionConfig):
    """Map SessionConfig's VAD knobs onto Gemini automatic activity detection.

    Best-effort and guarded: the exact types move around across google-genai
    versions, so if they're absent we return None and Gemini keeps its defaults.
    Gemini has no direct noise-reduction knob, so `noise_reduction` isn't applied
    here (it's an OpenAI lever) — the loudness gate is the provider-agnostic one."""
    if config.vad_threshold is None and config.vad_silence_ms is None:
        return None
    try:
        kwargs: dict[str, Any] = {}
        if config.vad_silence_ms is not None:
            kwargs["silence_duration_ms"] = int(config.vad_silence_ms)
        if config.vad_threshold is not None:
            # Higher threshold ⇒ commit fewer turns ⇒ LOW sensitivity.
            low = config.vad_threshold >= 0.6
            kwargs["start_of_speech_sensitivity"] = (
                types.StartSensitivity.START_SENSITIVITY_LOW
                if low
                else types.StartSensitivity.START_SENSITIVITY_HIGH
            )
            kwargs["end_of_speech_sensitivity"] = (
                types.EndSensitivity.END_SENSITIVITY_LOW
                if low
                else types.EndSensitivity.END_SENSITIVITY_HIGH
            )
        aad = types.AutomaticActivityDetection(**kwargs)
        return types.RealtimeInputConfig(automatic_activity_detection=aad)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("gemini VAD tuning unavailable in this google-genai: %s", exc)
        return None


def _build_live_config(config: SessionConfig) -> types.LiveConnectConfig:
    speech = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.voice)
        ),
        # Only set a language_code if explicitly requested; otherwise the
        # native-audio model auto-detects (the reliable path — see PLAN.md).
        language_code=config.language or None,
    )
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=config.system_instruction,
        speech_config=speech,
        # Ask for text of both sides so we can print/log/show what was said.
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=_activity_detection(config),
        tools=config.tools or None,
    )


class GeminiConnection:
    """Wraps a live Gemini session and normalizes its events."""

    def __init__(self, session, config: SessionConfig):
        self._session = session
        self._config = config
        self._events: asyncio.Queue = asyncio.Queue()
        self._recv_task = asyncio.create_task(self._receive())

    # --- sending -------------------------------------------------------------
    async def send_audio(self, pcm: bytes) -> None:
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={self._config.input_rate}")
        )

    async def send_image(self, jpeg: bytes, mime_type: str = "image/jpeg") -> None:
        await self._session.send_realtime_input(media=types.Blob(data=jpeg, mime_type=mime_type))

    async def send_text(self, text: str) -> None:
        await self._session.send_realtime_input(text=text)

    async def respond_tool(self, call: ToolCall, output: Any) -> None:
        await self._session.send_tool_response(
            function_responses=[
                types.FunctionResponse(id=call.id, name=call.name, response={"result": output})
            ]
        )

    async def interrupt(self) -> None:
        # Barge-in is automatic under Gemini's activity detection; nothing to do.
        return None

    # --- receiving -----------------------------------------------------------
    def events(self) -> AsyncIterator[VoiceEvent]:
        return self._drain()

    async def _drain(self) -> AsyncIterator[VoiceEvent]:
        while True:
            ev = await self._events.get()
            if ev is _DONE:
                return
            yield ev

    async def _receive(self) -> None:
        try:
            async for msg in self._session.receive():
                sc = msg.server_content
                if sc is not None:
                    if sc.interrupted:
                        self._events.put_nowait(Interrupted())
                    if sc.input_transcription and sc.input_transcription.text:
                        self._events.put_nowait(
                            UserTranscript(
                                sc.input_transcription.text,
                                final=bool(sc.input_transcription.finished),
                            )
                        )
                    if sc.output_transcription and sc.output_transcription.text:
                        self._events.put_nowait(
                            AgentTranscript(
                                sc.output_transcription.text,
                                final=bool(sc.output_transcription.finished),
                            )
                        )
                if msg.data:
                    self._events.put_nowait(AudioOut(msg.data))
                if msg.tool_call is not None:
                    for fc in msg.tool_call.function_calls:
                        self._events.put_nowait(
                            ToolCall(id=fc.id, name=fc.name, arguments=dict(fc.args or {}))
                        )
                if sc is not None and sc.turn_complete:
                    self._events.put_nowait(TurnComplete())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surface, don't crash the whole app
            self._events.put_nowait(ProviderError(str(exc)))
        finally:
            self._events.put_nowait(_DONE)

    async def aclose(self) -> None:
        self._recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._recv_task


class GeminiProvider:
    name = "gemini"

    @asynccontextmanager
    async def connect(self, config: SessionConfig):
        api_key = require_key("GEMINI_API_KEY")
        # v1beta is the channel that exposes the Live / native-audio models.
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        live_config = _build_live_config(config)
        async with client.aio.live.connect(model=config.model, config=live_config) as session:
            conn = GeminiConnection(session, config)
            try:
                yield conn
            finally:
                await conn.aclose()
