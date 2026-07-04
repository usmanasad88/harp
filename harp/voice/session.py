"""The runner: wire a provider to the mic and speaker and pump events.

This is provider-agnostic — it never mentions Gemini or OpenAI. It is what a
future orchestrator will call to open/close an ACTIVE voice session.
"""

from __future__ import annotations

import asyncio
import sys

from . import get_provider
from .audio_io import Microphone, Speaker
from .provider import (
    AgentTranscript,
    AudioOut,
    Interrupted,
    ProviderError,
    SessionConfig,
    ToolCall,
    TurnComplete,
    UserTranscript,
)


class _Printer:
    """Prints streaming transcripts with a speaker prefix, readably."""

    def __init__(self) -> None:
        self._who: str | None = None

    def show(self, who: str, text: str) -> None:
        if not text:
            return
        if who != self._who:
            sys.stdout.write(f"\n{who}: ")
            self._who = who
        sys.stdout.write(text)
        sys.stdout.flush()

    def turn_end(self) -> None:
        if self._who is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._who = None


async def run(provider_name: str, config: SessionConfig) -> None:
    """Open a live session and talk until Ctrl+C."""
    provider = get_provider(provider_name)
    printer = _Printer()

    async with (
        provider.connect(config) as conn,
        Microphone(config.input_rate) as mic,
        Speaker(config.output_rate) as speaker,
    ):

        async def pump_mic() -> None:
            async for pcm in mic.chunks():
                await conn.send_audio(pcm)

        async def handle_events() -> None:
            async for ev in conn.events():
                if isinstance(ev, AudioOut):
                    speaker.play(ev.pcm)
                elif isinstance(ev, AgentTranscript):
                    printer.show("HARP", ev.text)
                elif isinstance(ev, UserTranscript):
                    printer.show("You", ev.text)
                elif isinstance(ev, Interrupted):
                    speaker.clear()
                    printer.turn_end()
                elif isinstance(ev, TurnComplete):
                    printer.turn_end()
                elif isinstance(ev, ToolCall):
                    print(f"\n[tool call: {ev.name}({ev.arguments})]", file=sys.stderr)
                elif isinstance(ev, ProviderError):
                    print(f"\n[provider error] {ev.message}", file=sys.stderr)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(pump_mic())
            tg.create_task(handle_events())
