"""FilterAgent (Agent 1): scripted transcript turns → relayed messages, with the
ignore-sentinel dropped; context notes injected; the half-duplex mic gate; and
provider errors surfaced. Everything hardware-/network-shaped is a fake, like
test_bridge."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

import numpy as np

from harp.core.bus import Bus
from harp.core.events import ErrorRaised
from harp.voice.filter_agent import FilterAgent, LoudnessGate, clean_relay
from harp.voice.provider import AgentTranscript, ProviderError, SessionConfig


def _pcm(level: float, frames: int = 1024) -> bytes:
    """Constant-amplitude PCM whose RMS is exactly `level` (0..1 full scale)."""
    return np.full(frames, int(level * 32768), dtype=np.int16).tobytes()


SILENCE = _pcm(0.0)


def test_loudness_gate_disabled_passes_everything():
    gate = LoudnessGate(lambda: 0.0)
    quiet = _pcm(0.001)
    assert gate.process(quiet) == quiet  # threshold 0 = off


def test_loudness_gate_mutes_quiet_and_passes_loud_with_preroll():
    gate = LoudnessGate(lambda: 0.1, preroll_chunks=2, hangover_chunks=2)
    # Quiet chunks below threshold become silence (and are buffered as pre-roll).
    q1, q2 = _pcm(0.01), _pcm(0.02)
    assert gate.process(q1) == SILENCE
    assert gate.process(q2) == SILENCE
    # A loud chunk flushes the buffered pre-roll ahead of it, so the word onset
    # isn't clipped: output = last 2 quiet chunks + the loud chunk.
    loud = _pcm(0.3)
    out = gate.process(loud)
    assert out == q1 + q2 + loud


def test_loudness_gate_hangover_bridges_a_brief_dip():
    gate = LoudnessGate(lambda: 0.1, preroll_chunks=1, hangover_chunks=2)
    loud = _pcm(0.3)
    gate.process(loud)                      # arms hangover = 2
    dip = _pcm(0.01)
    assert gate.process(dip) == dip         # within hangover → still passes
    assert gate.process(dip) == dip         # hangover = 0 after this
    assert gate.process(_pcm(0.01)) == SILENCE  # now muted again


def test_loudness_gate_threshold_is_read_live():
    level = {"v": 0.5}
    gate = LoudnessGate(lambda: level["v"], preroll_chunks=1)
    mid = _pcm(0.2)
    assert gate.process(mid) == SILENCE  # 0.2 < 0.5 → muted (held as pre-roll)
    level["v"] = 0.1
    out = gate.process(mid)              # threshold lowered live → now passes
    assert out != SILENCE and out.endswith(mid)  # pre-roll flushed ahead of it


def _config() -> SessionConfig:
    return SessionConfig(system_instruction="filter", voice="v", model="m")


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
        while True:  # never actually hears anything; run() cancels us
            await asyncio.sleep(3600)
            yield b""


def _make_agent(bus, scripted, relayed, *, hold_open=False, mic_gate=None):
    provider = FakeProvider(scripted, hold_open)

    async def on_relay(text):
        relayed.append(text)

    agent = FilterAgent(
        bus,
        "openai",
        _config,
        on_relay,
        mic_gate=mic_gate,
        provider=provider,
        mic_factory=FakeMic,
    )
    return agent, provider


async def _wait_for(pred, timeout=1.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_clean_relay_strips_sentinel_and_punctuation_only_turns():
    assert clean_relay("aap kaise hain") == "aap kaise hain"
    assert clean_relay("[[ignore]]") == ""
    assert clean_relay("[[IGNORE]] .") == ""       # sentinel + stray punctuation
    assert clean_relay("  [[ ignore ]]  ") == ""    # spacing variants
    assert clean_relay("hello there.") == "hello there."


async def test_relays_cleaned_turns_and_drops_ignores():
    bus = Bus()
    relayed: list[str] = []
    agent, _ = _make_agent(
        bus,
        [
            AgentTranscript("aap kaise", final=False),
            AgentTranscript(" hain", final=True),   # streamed turn → one relay
            AgentTranscript("[[ignore]]", final=True),  # background/noise → dropped
            AgentTranscript("where is the stage?", final=True),
            AgentTranscript("[[IGNORE]].", final=True),  # dropped despite the period
        ],
        relayed,
    )

    await asyncio.wait_for(agent.run(), 1.0)

    assert relayed == ["aap kaise hain", "where is the stage?"]


async def test_opening_context_is_sent_as_a_context_note():
    bus = Bus()
    agent, provider = _make_agent(bus, [], [])  # empty script → run returns at once

    await asyncio.wait_for(agent.run(context="someone waved at you"), 1.0)

    assert provider.conn.sent_texts == ["CONTEXT: someone waved at you"]


async def test_no_opening_context_sends_nothing():
    bus = Bus()
    agent, provider = _make_agent(bus, [], [])

    await asyncio.wait_for(agent.run(context=""), 1.0)

    assert provider.conn.sent_texts == []


async def test_add_context_is_forwarded_as_a_context_note():
    bus = Bus()
    agent, provider = _make_agent(bus, [], [], hold_open=True)
    task = asyncio.create_task(agent.run())

    agent.add_context("You (the assistant) just said: tickets are 500 rupees")
    await _wait_for(lambda: provider.conn is not None and provider.conn.sent_texts)

    assert provider.conn.sent_texts == [
        "CONTEXT: You (the assistant) just said: tickets are 500 rupees"
    ]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_mic_gate_substitutes_silence_when_muted():
    """Half-duplex: real audio while the gate is open, same-length silence when
    closed (responder speaking), so the filter never relays HARP's own voice."""
    bus = Bus()
    open_ = {"v": True}
    agent, _ = _make_agent(bus, [], [], mic_gate=lambda: open_["v"])

    assert agent._mic_payload(b"abcd") == b"abcd"
    open_["v"] = False
    assert agent._mic_payload(b"abcd") == b"\x00\x00\x00\x00"


async def test_provider_error_becomes_error_raised():
    bus = Bus()
    stream = bus.subscribe(ErrorRaised)
    agent, _ = _make_agent(bus, [ProviderError("filter quota exceeded")], [])

    await asyncio.wait_for(agent.run(), 1.0)

    ev = await anext(stream)
    assert ev.where == "voice.filter"
    assert "quota" in ev.message


async def test_mic_pump_applies_the_loudness_gate():
    """The loudness gate is actually wired into the filter's mic path: quiet
    room audio is sent as silence, loud speech passes (with its pre-roll)."""
    bus = Bus()

    class ScriptMic:
        def __init__(self, rate):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def chunks(self):
            yield _pcm(0.01)   # quiet room → should be muted
            yield _pcm(0.30)   # loud speech → should pass
            while True:
                await asyncio.sleep(3600)
                yield b""

    provider = FakeProvider([], hold_open=True)

    async def on_relay(text):
        pass

    agent = FilterAgent(
        bus, "openai", _config, on_relay,
        near_field_level=lambda: 0.1,
        provider=provider, mic_factory=ScriptMic,
    )
    task = asyncio.create_task(agent.run())
    await _wait_for(lambda: provider.conn is not None and len(provider.conn.sent_audio) >= 2)

    assert provider.conn.sent_audio[0] == SILENCE            # quiet muted
    assert provider.conn.sent_audio[1] == _pcm(0.01) + _pcm(0.30)  # pre-roll + loud
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_build_filter_config_stamps_tuning_onto_the_session():
    from harp.config import FilterTuning, build_filter_config
    from harp.voice.openai import _build_session

    tuning = FilterTuning(vad_threshold=0.8, vad_silence_ms=900, noise_reduction="near_field")
    cfg = build_filter_config("openai", tuning)
    assert cfg.vad_threshold == 0.8
    assert cfg.vad_silence_ms == 900
    assert cfg.noise_reduction == "near_field"
    assert cfg.tools == []  # the filter never gets tools

    # ...and the OpenAI backend maps them into its server-VAD / noise config.
    session = _build_session(cfg)
    td = session["audio"]["input"]["turn_detection"]
    assert td["threshold"] == 0.8
    assert td["silence_duration_ms"] == 900
    assert session["audio"]["input"]["noise_reduction"] == {"type": "near_field"}


def test_build_filter_config_none_noise_reduction_is_omitted():
    from harp.config import FilterTuning, build_filter_config
    from harp.voice.openai import _build_session

    cfg = build_filter_config("openai", FilterTuning(noise_reduction="none"))
    assert cfg.noise_reduction is None  # "none" → unset, so the backend omits it
    assert "noise_reduction" not in _build_session(cfg)["audio"]["input"]
