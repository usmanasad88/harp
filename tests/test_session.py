"""session.run: the bare `python -m harp` runner. Focus here is the data-
retrieval seam — a ToolCall gets dispatched and its result returned to the
model — plus that the runner ends when the provider closes its event stream.
Everything hardware- or network-shaped is injected (fake provider/mic/speaker),
so no mic, speaker, model, or API key is touched."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from harp.voice import session
from harp.voice.provider import (
    AudioOut,
    SessionConfig,
    ToolCall,
)


def _config() -> SessionConfig:
    return SessionConfig(system_instruction="test", voice="v", model="m")


class FakeConn:
    def __init__(self, scripted):
        self._scripted = scripted
        self.sent_audio: list[bytes] = []
        self.tool_responses: list[tuple[ToolCall, object]] = []

    async def send_audio(self, pcm):
        self.sent_audio.append(pcm)

    async def respond_tool(self, call, output):
        self.tool_responses.append((call, output))

    async def events(self):
        for ev in self._scripted:
            yield ev


class FakeProvider:
    def __init__(self, scripted):
        self.scripted = scripted
        self.conn: FakeConn | None = None

    @asynccontextmanager
    async def connect(self, config):
        self.conn = FakeConn(self.scripted)
        yield self.conn


class FakeMic:
    def __init__(self, rate):
        self.rate = rate

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def chunks(self):
        while True:  # a mic that never hears anything; run() cancels us
            await asyncio.sleep(3600)
            yield b""


class FakeSpeaker:
    def __init__(self, rate):
        self.rate = rate
        self.played: list[bytes] = []
        self.cleared = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def play(self, pcm):
        self.played.append(pcm)

    def clear(self):
        self.cleared += 1


async def _run(scripted, tool_dispatch=None):
    provider = FakeProvider(scripted)
    await asyncio.wait_for(
        session.run(
            "openai",
            _config(),
            tool_dispatch=tool_dispatch,
            provider=provider,
            mic_factory=FakeMic,
            speaker_factory=FakeSpeaker,
        ),
        1.0,
    )
    return provider


async def test_run_returns_when_provider_stream_ends():
    # The scripted events are exhausted immediately; run() must not hang on the
    # never-ending mic (regression guard for the TaskGroup-hang shape).
    provider = await _run([AudioOut(b"pcm")])
    assert provider.conn.tool_responses == []


async def test_tool_call_is_dispatched_and_answered():
    call = ToolCall(id="c1", name="search_knowledge", arguments={"query": "tickets"})

    seen = []

    async def dispatch(name, arguments):
        seen.append((name, arguments))
        return [{"heading": "Tickets", "text": "500 rupees"}]

    provider = await _run([call], tool_dispatch=dispatch)

    assert seen == [("search_knowledge", {"query": "tickets"})]
    assert provider.conn.tool_responses == [
        (call, [{"heading": "Tickets", "text": "500 rupees"}])
    ]


async def test_tool_failure_returns_error_payload_to_model():
    call = ToolCall(id="c1", name="search_knowledge", arguments={})

    async def dispatch(name, arguments):
        raise RuntimeError("index on fire")

    provider = await _run([call], tool_dispatch=dispatch)

    ((_, output),) = provider.conn.tool_responses
    assert output == {"error": "index on fire"}


async def test_tool_call_without_dispatcher_still_answers_the_model():
    # No handler wired = the old report-only path must still close the tool
    # call, or the model would hang waiting for a function result.
    call = ToolCall(id="c1", name="search_knowledge", arguments={"query": "x"})

    provider = await _run([call], tool_dispatch=None)

    ((_, output),) = provider.conn.tool_responses
    assert "error" in output
