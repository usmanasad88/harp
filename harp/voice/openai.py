"""OpenAI Realtime backend for the provider abstraction.

Mirrors the session config proven in the web-realtime sandbox (GA "realtime"
session, 24 kHz PCM both ways, server VAD, input transcription, function
tools with tool_choice auto). Whatever declarations are in
`SessionConfig.tools` are advertised to the model; harp/knowledge provides
the search_knowledge declaration + dispatcher, and app.py wires them in.

Transport note: the browser sandbox uses WebRTC (audio on a media track). A
headless process uses the WebSocket path instead, via the official SDK, so
audio arrives as `response.output_audio.delta` events and is sent with
`input_audio_buffer.append`. Everything else (event names, VAD, transcription)
matches the sandbox.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from ..config import load_openai_transcribe_prompt, require_key
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

_DONE = object()

# HARP is English/Urdu only, and we want the transcript in the Latin alphabet:
# English as-is, and spoken Urdu ROMANIZED (e.g. "aap kaise hain") rather than in
# Perso-Arabic — readable on the dashboard and consistent with the romanized wake
# words in harp.yaml. We steer script with a prompt rather than pinning
# `language`, which would mis-transcribe the other language whenever the user
# switches. See _still_prompt_prefix below for the price of priming a
# Whisper-family model this way. Wording lives in
# prompts/transcription_openai.md (see prompts/README.md).
_DEFAULT_TRANSCRIBE_PROMPT = load_openai_transcribe_prompt()


def _transcribe_prompt() -> str:
    return os.getenv("OPENAI_TRANSCRIBE_PROMPT", _DEFAULT_TRANSCRIBE_PROMPT)


def _norm(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for prefix comparison."""
    return " ".join(re.sub(r"[^\w\s]", " ", text).lower().split())


def _still_prompt_prefix(text: str, prompt: str) -> bool:
    """True while `text` could still be the opening of the transcription prompt.

    Whisper-family transcribers (gpt-4o-mini-transcribe here) regurgitate their
    priming prompt verbatim when handed silence / a breath / background noise
    that the server VAD still commits as a turn — it would otherwise reach the
    dashboard as though the visitor said it. Such an echo reproduces the prompt
    from its start, so while a user transcript is still a prefix of the prompt we
    withhold it; the moment it diverges (real speech does so within a word or
    two) we know it's genuine and stream it. Compared on normalized text so a
    delta that splits a word mid-token ("...only Eng") still reads as a prefix.
    """
    t = _norm(text)
    return not t or _norm(prompt).startswith(t)


def _build_session(config: SessionConfig) -> dict:
    """The GA 'realtime' session config, mirroring web-realtime/server.js."""
    audio_input = {
        "format": {"type": "audio/pcm", "rate": config.input_rate},
        "transcription": {
            "model": os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
            "prompt": _transcribe_prompt(),
        },
        "turn_detection": {
            "type": "server_vad",
            # SessionConfig can raise these (the two-agent filter does, so room
            # noise doesn't commit a turn); None keeps the sandbox-proven defaults.
            "threshold": config.vad_threshold if config.vad_threshold is not None else 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": (
                config.vad_silence_ms if config.vad_silence_ms is not None else 500
            ),
        },
    }
    if config.noise_reduction:
        # OpenAI's own input noise reduction; near_field for a close mic, far_field
        # for a distant one. Omitted (no key) = off, the default.
        audio_input["noise_reduction"] = {"type": config.noise_reduction}
    session = {
        "type": "realtime",
        "instructions": config.system_instruction,
        "output_modalities": ["audio"],
        "audio": {
            "input": audio_input,
            "output": {
                "format": {"type": "audio/pcm", "rate": config.output_rate},
                "voice": config.voice,
            },
        },
        "tools": config.tools or [],
    }
    if config.tools:
        # Same setting the sandbox proved out: the model decides when to call.
        session["tool_choice"] = "auto"
    return session


class OpenAIConnection:
    """Wraps a live OpenAI Realtime WebSocket connection and normalizes events."""

    def __init__(self, conn, config: SessionConfig):
        self._conn = conn
        self._config = config
        self._events: asyncio.Queue = asyncio.Queue()
        self._prompt = _transcribe_prompt()  # withhold echoes of it (see _still_prompt_prefix)
        self._user_held = ""  # user-transcript text held back while it still looks like the prompt
        self._user_streaming = False  # once True this turn is genuine speech — stream it directly
        self._recv_task = asyncio.create_task(self._receive())

    # --- sending -------------------------------------------------------------
    async def send_audio(self, pcm: bytes) -> None:
        await self._conn.input_audio_buffer.append(audio=base64.b64encode(pcm).decode("ascii"))

    async def send_image(self, jpeg: bytes, mime_type: str = "image/jpeg") -> None:
        # Vision is a later phase; the OpenAI image path is added with it.
        return None

    async def send_text(self, text: str) -> None:
        await self._conn.conversation.item.create(
            item={"type": "message", "role": "user",
                  "content": [{"type": "input_text", "text": text}]}
        )
        await self._conn.response.create()

    async def respond_tool(self, call: ToolCall, output: Any) -> None:
        payload = output if isinstance(output, str) else json.dumps(output)
        await self._conn.conversation.item.create(
            item={"type": "function_call_output", "call_id": call.id, "output": payload}
        )
        await self._conn.response.create()

    async def interrupt(self) -> None:
        # Server VAD interrupts the model automatically; the local speaker is
        # cleared on the speech_started event by the runner.
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

    def _on_user_delta(self, delta: str) -> None:
        """Stream a user-transcript delta, but hold it back while the turn so far
        still looks like the transcription prompt echoed on silence/noise."""
        if self._user_streaming:
            if delta:
                self._events.put_nowait(UserTranscript(delta))
            return
        self._user_held += delta
        if not _still_prompt_prefix(self._user_held, self._prompt):
            # Diverged from the prompt -> genuine speech. Release what we held
            # and stream the rest of this turn directly.
            if self._user_held:
                self._events.put_nowait(UserTranscript(self._user_held))
            self._user_held = ""
            self._user_streaming = True

    def _finish_user_turn(self) -> None:
        streamed = self._user_streaming
        self._user_held = ""
        self._user_streaming = False
        if streamed:
            # Text already streamed via deltas; this just closes the turn.
            self._events.put_nowait(UserTranscript("", final=True))
        # else: nothing was streamed — an empty turn, or the transcriber echoing
        # its own priming prompt on silence/noise. Drop it; it never happened.

    async def _receive(self) -> None:
        try:
            async for event in self._conn:
                t = event.type
                if t == "response.output_audio.delta":
                    self._events.put_nowait(AudioOut(base64.b64decode(event.delta)))
                elif t == "response.output_audio_transcript.delta":
                    self._events.put_nowait(AgentTranscript(event.delta))
                elif t == "response.output_audio_transcript.done":
                    # Empty final marker: the text already streamed via deltas;
                    # this just tells consumers the agent's turn is closed.
                    self._events.put_nowait(AgentTranscript("", final=True))
                elif t == "conversation.item.input_audio_transcription.delta":
                    self._on_user_delta(event.delta or "")
                elif t == "conversation.item.input_audio_transcription.completed":
                    self._finish_user_turn()
                elif t == "input_audio_buffer.speech_started":
                    self._events.put_nowait(Interrupted())
                elif t == "response.function_call_arguments.done":
                    self._events.put_nowait(
                        ToolCall(
                            id=event.call_id,
                            name=event.name,
                            arguments=json.loads(event.arguments or "{}"),
                        )
                    )
                elif t == "response.done":
                    self._events.put_nowait(TurnComplete())
                elif t == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", None) or getattr(err, "code", None) or str(event)
                    self._events.put_nowait(ProviderError(str(msg)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._events.put_nowait(ProviderError(str(exc)))
        finally:
            self._events.put_nowait(_DONE)

    async def aclose(self) -> None:
        self._recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._recv_task


class OpenAIProvider:
    name = "openai"

    @asynccontextmanager
    async def connect(self, config: SessionConfig):
        api_key = require_key("OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key)
        async with client.realtime.connect(model=config.model) as conn:
            await conn.session.update(session=_build_session(config))
            wrapper = OpenAIConnection(conn, config)
            try:
                yield wrapper
            finally:
                await wrapper.aclose()
