"""Wake-detector tests with synthetic audio — no mic, no Whisper.

Chunks are constant-amplitude PCM, so their RMS level is exactly the amplitude
we ask for, and phrase timing is derived from sample counts (deterministic).
"""

from __future__ import annotations

import asyncio

import numpy as np

from harp.config import ListenerSettings
from harp.core.bus import Bus
from harp.core.events import PhraseHeard
from harp.listener.detector import (
    LoudSound,
    Phrase,
    WakeDetector,
    match_wake_word,
    rms_level,
)
from harp.listener.listener import AlwaysOnListener

RATE = 16000


def chunk(level: float, seconds: float = 0.064) -> bytes:
    """Constant-amplitude PCM whose RMS is exactly `level`."""
    value = int(level * 32768)
    return np.full(int(seconds * RATE), value, dtype=np.int16).tobytes()


def settings(**overrides) -> ListenerSettings:
    base = dict(
        wake_level=0.5,
        transcribe_level=0.1,
        silence_seconds=0.2,
        max_phrase_seconds=1.0,
    )
    base.update(overrides)
    return ListenerSettings(**base)


def test_rms_level_matches_amplitude():
    assert abs(rms_level(chunk(0.3)) - 0.3) < 0.01
    assert rms_level(chunk(0.0)) == 0.0
    assert rms_level(b"") == 0.0


def test_loud_sound_wakes_immediately():
    detector = WakeDetector(settings(), RATE)
    decision = detector.feed(chunk(0.6))
    assert isinstance(decision, LoudSound)
    assert abs(decision.level - 0.6) < 0.01


def test_quiet_audio_does_nothing():
    detector = WakeDetector(settings(), RATE)
    for _ in range(20):
        assert detector.feed(chunk(0.02)) is None


def test_speech_then_silence_yields_phrase_with_preroll():
    detector = WakeDetector(settings(), RATE)
    assert detector.feed(chunk(0.02)) is None  # quiet: goes into pre-roll
    speech = [chunk(0.3) for _ in range(3)]
    for c in speech:
        assert detector.feed(c) is None
    decision = None
    while decision is None:  # trailing silence closes the phrase
        decision = detector.feed(chunk(0.02))
    assert isinstance(decision, Phrase)
    # Captured audio includes the speech plus the quiet pre-roll chunk.
    assert len(decision.pcm) > sum(len(c) for c in speech)


def test_phrase_is_capped_at_max_seconds():
    detector = WakeDetector(settings(max_phrase_seconds=0.5), RATE)
    decision = None
    fed = 0.0
    while decision is None:  # continuous speech, never any silence
        decision = detector.feed(chunk(0.3))
        fed += 0.064
        assert fed < 2.0, "detector never capped the phrase"
    assert isinstance(decision, Phrase)


def test_loud_sound_mid_phrase_wins():
    detector = WakeDetector(settings(), RATE)
    assert detector.feed(chunk(0.3)) is None  # capturing
    assert isinstance(detector.feed(chunk(0.7)), LoudSound)
    # Capture was reset: quiet audio now yields nothing, not a leftover phrase.
    for _ in range(10):
        assert detector.feed(chunk(0.02)) is None


def test_match_wake_word_whole_words_only():
    words = ["hey", "hi", "hello"]
    assert match_wake_word("Hello there!", words) == "hello"
    assert match_wake_word("HI!", words) == "hi"
    assert match_wake_word("this is fine", words) is None  # "hi" inside "this"
    assert match_wake_word("", words) is None


def test_match_wake_word_multiword_and_nonlatin():
    assert match_wake_word("assalam o alaikum", ["assalam o alaikum"]) is not None
    assert match_wake_word("سلام، آپ کیسے ہیں؟", ["سلام"]) == "سلام"


async def test_transcribed_phrase_is_published_even_without_wake_word():
    """Every transcript goes on the bus as PhraseHeard (transcriber injected —
    still no mic, no Whisper), so the dashboard shows what the ears heard."""
    bus = Bus()
    listener = AlwaysOnListener(
        bus,
        settings(wake_words=["hello"]),
        transcribe=lambda pcm, rate: "no matching words here",
    )
    stream = bus.subscribe(PhraseHeard)
    await listener._handle(Phrase(pcm=chunk(0.3)))
    heard = await asyncio.wait_for(anext(stream), timeout=2.0)
    assert heard.text == "no matching words here"
    assert heard.wake_word is None
    await stream.aclose()
