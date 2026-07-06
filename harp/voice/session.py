"""The runner: wire a provider to the mic and speaker and pump events.

This is provider-agnostic — it never mentions Gemini or OpenAI. It is what a
future orchestrator will call to open/close an ACTIVE voice session.

Data retrieval rides in through the same `tool_dispatch` seam the supervised
bridge uses: when the model emits a ToolCall (e.g. search_knowledge), we run
the injected dispatcher and hand the result back with `respond_tool`, so
`python -m harp` grounds its answers in `data/` exactly like harp.app does —
just without the bus/dashboard around it. No dispatcher wired = the old
behavior (tool calls are only reported), so this stays provider- and
corpus-agnostic; __main__.py is the composition root that injects knowledge.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import Any, Awaitable, Callable

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

# (tool name, arguments) -> result for respond_tool. Same shape the bridge
# uses; knowledge.tools.dispatch is the one wired in today.
ToolDispatch = Callable[[str, dict], Awaitable[Any]]


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


async def run(
    provider_name: str,
    config: SessionConfig,
    tool_dispatch: ToolDispatch | None = None,
    *,
    provider=None,
    mic_factory: Callable[[int], Microphone] = Microphone,
    speaker_factory: Callable[[int], Speaker] = Speaker,
) -> None:
    """Open a live session and talk until Ctrl+C (or the provider ends the stream).

    `tool_dispatch(name, arguments)` runs a tool the model requests and returns
    its result; None keeps the old report-only behavior. The provider, mic, and
    speaker are injectable so this is testable without hardware — the same shape
    the bridge uses.
    """
    provider = provider or get_provider(provider_name)
    printer = _Printer()

    async with (
        provider.connect(config) as conn,
        mic_factory(config.input_rate) as mic,
        speaker_factory(config.output_rate) as speaker,
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
                    output = await _run_tool(conn, ev, tool_dispatch)
                    print(
                        f"\n[tool {ev.name}({ev.arguments}) -> {_summarize(output)}]",
                        file=sys.stderr,
                    )
                elif isinstance(ev, ProviderError):
                    print(f"\n[provider error] {ev.message}", file=sys.stderr)

        # Mic runs as a side task so that when the provider closes its event
        # stream, handle_events returns and we tear the mic down (rather than
        # both tasks blocking forever in a TaskGroup).
        mic_task = asyncio.create_task(pump_mic())
        try:
            await handle_events()
        finally:
            mic_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mic_task


async def _run_tool(conn, call: ToolCall, tool_dispatch: ToolDispatch | None) -> Any:
    """Run a requested tool and return its result to the model. A missing
    handler or a failing tool degrades into an {"error": ...} the model can
    apologize for, never a crashed session — same contract as the bridge."""
    if tool_dispatch is None:
        output: Any = {"error": f"no tool handler wired for {call.name}"}
    else:
        try:
            output = await tool_dispatch(call.name, call.arguments)
        except Exception as exc:  # noqa: BLE001 — a bad tool must not kill the session
            output = {"error": str(exc)}
    await conn.respond_tool(call, output)
    return output


def _summarize(output: Any) -> str:
    """Compact one-line view of a tool result for the terminal log."""
    if isinstance(output, list):
        return f"{len(output)} result(s)"
    return str(output)
