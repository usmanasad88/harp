"""VoiceBridge: provider VoiceEvents in → bus events out, tool round-trips,
and the opening context. Everything hardware- or network-shaped is injected:
a fake provider (scripted events), fake mic (never yields), fake speaker
(records play/clear)."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

import numpy as np
import pytest

from harp.core.bus import Bus
from harp.core.events import (
    AgentSaid,
    ErrorRaised,
    ToolCompleted,
    ToolRequested,
    UserSaid,
)
from harp.voice.bridge import VoiceBridge
from harp.voice.provider import (
    AgentTranscript,
    AudioOut,
    Interrupted,
    ProviderError,
    SessionConfig,
    ToolCall,
    UserTranscript,
)


def _config() -> SessionConfig:
    return SessionConfig(system_instruction="test", voice="v", model="m")


class FakeConn:
    def __init__(self, scripted, hold_open=False):
        self._scripted = scripted
        self._hold_open = hold_open
        self.sent_texts: list[str] = []
        self.sent_audio: list[bytes] = []
        self.tool_responses: list[tuple[ToolCall, object]] = []

    async def send_audio(self, pcm):
        self.sent_audio.append(pcm)

    async def send_text(self, text):
        self.sent_texts.append(text)

    async def respond_tool(self, call, output):
        self.tool_responses.append((call, output))

    async def events(self):
        for ev in self._scripted:
            yield ev
        if self._hold_open:  # keep the session open (tests that drive input live)
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


def make_bridge(bus, scripted, **kwargs):
    provider = FakeProvider(scripted)
    speakers: list[FakeSpeaker] = []

    def speaker_factory(rate):
        speaker = FakeSpeaker(rate)
        speakers.append(speaker)
        return speaker

    bridge = VoiceBridge(
        bus,
        "openai",
        make_config=_config,
        provider=provider,
        mic_factory=FakeMic,
        speaker_factory=speaker_factory,
        **kwargs,
    )
    return bridge, provider, speakers


async def test_transcripts_become_bus_events():
    bus = Bus()
    stream = bus.subscribe(UserSaid, AgentSaid)
    bridge, _, _ = make_bridge(
        bus,
        [
            UserTranscript("hello", final=False),
            UserTranscript("", final=True),
            AgentTranscript("hi there", final=False),
            AgentTranscript("", final=True),
        ],
    )

    await asyncio.wait_for(bridge.run(), 1.0)

    assert await anext(stream) == UserSaid(text="hello", final=False)
    assert await anext(stream) == UserSaid(text="", final=True)
    assert await anext(stream) == AgentSaid(text="hi there", final=False)
    assert await anext(stream) == AgentSaid(text="", final=True)


async def test_audio_plays_and_interruption_clears_speaker():
    bus = Bus()
    bridge, _, speakers = make_bridge(bus, [AudioOut(b"pcm1"), Interrupted(), AudioOut(b"pcm2")])

    await asyncio.wait_for(bridge.run(), 1.0)

    (speaker,) = speakers
    assert speaker.played == [b"pcm1", b"pcm2"]
    assert speaker.cleared == 1


async def test_tool_call_round_trip():
    bus = Bus()
    stream = bus.subscribe(ToolRequested, ToolCompleted)
    call = ToolCall(id="c1", name="search_knowledge", arguments={"query": "tickets"})

    async def dispatch(name, arguments):
        assert (name, arguments) == ("search_knowledge", {"query": "tickets"})
        return [{"text": "tickets cost 500"}]

    bridge, provider, _ = make_bridge(bus, [call], tool_dispatch=dispatch)

    await asyncio.wait_for(bridge.run(), 1.0)

    requested = await anext(stream)
    assert (requested.id, requested.name) == ("c1", "search_knowledge")
    completed = await anext(stream)
    assert completed.output == [{"text": "tickets cost 500"}]
    assert provider.conn.tool_responses == [(call, [{"text": "tickets cost 500"}])]


async def test_tool_failure_returns_error_payload_to_model():
    bus = Bus()
    call = ToolCall(id="c1", name="search_knowledge", arguments={})

    async def dispatch(name, arguments):
        raise RuntimeError("index on fire")

    bridge, provider, _ = make_bridge(bus, [call], tool_dispatch=dispatch)

    await asyncio.wait_for(bridge.run(), 1.0)

    ((_, output),) = provider.conn.tool_responses
    assert output == {"error": "index on fire"}


async def test_provider_error_becomes_error_raised():
    bus = Bus()
    stream = bus.subscribe(ErrorRaised)
    bridge, _, _ = make_bridge(bus, [ProviderError("quota exceeded")])

    await asyncio.wait_for(bridge.run(), 1.0)

    ev = await anext(stream)
    assert ev.where == "voice.provider"
    assert "quota" in ev.message


async def test_opening_text_combines_wake_context_and_identity():
    bus = Bus()
    bridge, provider, _ = make_bridge(
        bus, [], identity_context=lambda: "(you are talking to Ada.)"
    )

    await asyncio.wait_for(bridge.run(context="(someone said hello.)"), 1.0)

    assert provider.conn.sent_texts == ["(someone said hello.)\n(you are talking to Ada.)"]


async def test_no_opening_text_sends_nothing():
    bus = Bus()
    bridge, provider, _ = make_bridge(bus, [], identity_context=lambda: "")

    await asyncio.wait_for(bridge.run(context=""), 1.0)

    assert provider.conn.sent_texts == []


def test_mic_gate_sends_silence_while_released():
    """Push-to-talk: real mic audio only while held; same-length silence when
    released, so no room noise reaches the model but its VAD still sees quiet."""
    bus = Bus()
    held = {"down": True}
    bridge, _, _ = make_bridge(bus, [], mic_gate=lambda: held["down"])

    assert bridge._mic_payload(b"abcd") == b"abcd"  # key held → real audio
    held["down"] = False
    assert bridge._mic_payload(b"abcd") == b"\x00\x00\x00\x00"  # released → silence


def test_ungated_mic_passes_audio_through():
    bus = Bus()
    bridge, _, _ = make_bridge(bus, [])  # no gate = the default, non-PTT behavior
    assert bridge._mic_payload(b"abcd") == b"abcd"


async def _wait_for(pred, timeout=1.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


async def test_text_driven_responder_forwards_text_and_opens_no_mic():
    """The two-agent responder: fed by an injected text queue, it forwards each
    message to the model as a user turn and never opens a microphone."""
    bus = Bus()
    inbox: asyncio.Queue[str] = asyncio.Queue()
    provider = FakeProvider([], hold_open=True)  # stay open so _pump_text runs

    def no_mic(rate):
        raise AssertionError("a text-driven responder must not open a mic")

    bridge = VoiceBridge(
        bus,
        "openai",
        make_config=_config,
        text_inbox=inbox,
        provider=provider,
        mic_factory=no_mic,
        speaker_factory=FakeSpeaker,
    )
    task = asyncio.create_task(bridge.run())
    inbox.put_nowait("hello from the filter")
    await _wait_for(
        lambda: provider.conn is not None
        and "hello from the filter" in provider.conn.sent_texts
    )
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _pcm(level: float, frames: int = 1024) -> bytes:
    """Constant-amplitude PCM whose RMS is exactly `level` (0..1 full scale)."""
    return np.full(frames, int(level * 32768), dtype=np.int16).tobytes()


SILENCE = _pcm(0.0)


async def test_mic_pump_applies_the_loudness_gate():
    """The loudness gate isn't just wired into the two-agent filter — the plain
    single-agent bridge's own mic pump applies it too (the point of extending
    the dashboard's noise/VAD panel to single-agent mode: it has to actually
    affect this session's mic, not just move sliders)."""
    bus = Bus()

    class ScriptMic:
        def __init__(self, rate):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def chunks(self):
            yield _pcm(0.01)   # quiet room -> should be muted
            yield _pcm(0.30)   # loud speech -> should pass
            while True:
                await asyncio.sleep(3600)
                yield b""

    provider = FakeProvider([], hold_open=True)
    bridge = VoiceBridge(
        bus,
        "openai",
        make_config=_config,
        near_field_level=lambda: 0.1,
        provider=provider,
        mic_factory=ScriptMic,
        speaker_factory=FakeSpeaker,
    )
    task = asyncio.create_task(bridge.run())
    await _wait_for(lambda: provider.conn is not None and len(provider.conn.sent_audio) >= 2)

    assert provider.conn.sent_audio[0] == SILENCE                    # quiet muted
    assert provider.conn.sent_audio[1] == _pcm(0.01) + _pcm(0.30)     # pre-roll + loud
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_loudness_gate_defaults_off_no_behavior_change():
    """near_field_level=None (the default) must not alter mic audio at all —
    existing push-to-talk/ungated behavior stays exactly as before."""
    bus = Bus()
    bridge, _, _ = make_bridge(bus, [])
    from harp.voice.loudness_gate import LoudnessGate

    gate = LoudnessGate(bridge._near_field_level or (lambda: 0.0))
    assert gate.process(_pcm(0.001)) == _pcm(0.001)
