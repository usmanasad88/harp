"""TwoAgentBridge (the coordinator): the two wires between filter and responder
(relay → UserSaid + responder inbox; responder reply → filter context), the
half-duplex mic gate, and that run() composes both sessions and ends cleanly
when one ends. Fakes for providers/mic/speaker, like test_bridge."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from harp.core.bus import Bus
from harp.core.events import AgentSaid, UserSaid
from harp.voice.provider import SessionConfig
from harp.voice.two_agent import TwoAgentBridge


def _config() -> SessionConfig:
    return SessionConfig(system_instruction="x", voice="v", model="m")


class FakeConn:
    def __init__(self, scripted, hold_open=False):
        self._scripted = scripted
        self._hold_open = hold_open
        self.sent_texts: list[str] = []
        self.sent_audio: list[bytes] = []

    async def send_audio(self, pcm):
        self.sent_audio.append(pcm)

    async def send_text(self, text):
        self.sent_texts.append(text)

    async def respond_tool(self, call, output):
        pass

    async def events(self):
        for ev in self._scripted:
            yield ev
        if self._hold_open:
            await asyncio.Event().wait()


class FakeProvider:
    def __init__(self, scripted, hold_open=False):
        self.scripted = scripted
        self.hold_open = hold_open
        self.conn: FakeConn | None = None

    @asynccontextmanager
    async def connect(self, config):
        self.conn = FakeConn(self.scripted, self.hold_open)
        yield self.conn


class FakeMic:
    def __init__(self, rate):
        self.rate = rate

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def chunks(self):
        while True:
            await asyncio.sleep(3600)
            yield b""


class FakeSpeaker:
    def __init__(self, rate):
        self.rate = rate

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def play(self, pcm):
        pass

    def clear(self):
        pass


class FakeFilter:
    """Stand-in for the filter agent when testing _watch_responder in isolation."""

    def __init__(self):
        self.contexts: list[str] = []

    def add_context(self, text):
        self.contexts.append(text)


def _bridge(bus, **kwargs):
    return TwoAgentBridge(
        bus,
        "openai",
        make_config=_config,
        make_filter_config=_config,
        mic_factory=FakeMic,
        speaker_factory=FakeSpeaker,
        **kwargs,
    )


async def _wait_for(pred, timeout=1.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


async def test_relay_publishes_user_turn_feeds_responder_and_mutes_mic():
    bus = Bus()
    stream = bus.subscribe(UserSaid)
    bridge = _bridge(bus)

    await bridge._relay("where is the robotics stage")

    assert await anext(stream) == UserSaid(text="where is the robotics stage", final=True)
    assert bridge._to_responder.get_nowait() == "where is the robotics stage"
    # Right after a relay the filter's mic is muted (half-duplex), even before the
    # responder starts speaking.
    assert bridge._filter_mic_gate() is False


async def test_responder_reply_feeds_back_as_context_and_gates_mic():
    bus = Bus()
    bridge = _bridge(bus, response_tail_seconds=0.05)
    fake_filter = FakeFilter()
    bridge._filter = fake_filter
    watcher = asyncio.create_task(bridge._watch_responder())
    await asyncio.sleep(0.05)  # let the watcher subscribe before we publish

    await bus.publish(AgentSaid(text="Tickets ", final=False))
    await _wait_for(lambda: bridge._speaking)
    assert bridge._filter_mic_gate() is False  # muted while HARP speaks

    await bus.publish(AgentSaid(text="are 500 rupees.", final=True))
    await _wait_for(lambda: fake_filter.contexts)
    assert fake_filter.contexts == [
        "You (the assistant) just said: Tickets are 500 rupees."
    ]
    # After the tail elapses the mic reopens on its own.
    await _wait_for(lambda: bridge._filter_mic_gate() is True, timeout=1.0)

    watcher.cancel()


async def test_external_gate_can_still_mute_the_filter_mic():
    bus = Bus()
    allow = {"v": True}
    bridge = _bridge(bus, external_mic_gate=lambda: allow["v"])

    assert bridge._filter_mic_gate() is True   # idle + external allows → live
    allow["v"] = False
    assert bridge._filter_mic_gate() is False  # push-to-talk key up → muted


async def test_run_composes_both_sessions_and_ends_when_one_ends():
    """The filter session ending (provider closed the stream) ends the whole
    interaction and cancels the responder — the shape the orchestrator relies on
    to return to STANDBY."""
    bus = Bus()
    filter_provider = FakeProvider([])                 # ends immediately
    responder_provider = FakeProvider([], hold_open=True)  # would block forever
    bridge = _bridge(
        bus,
        filter_provider=filter_provider,
        responder_provider=responder_provider,
    )

    await asyncio.wait_for(bridge.run(context="hi"), 1.0)  # returns, doesn't hang
