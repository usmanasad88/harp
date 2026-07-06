"""The OpenAI input transcriber is primed with a prompt to steer Urdu into the
Latin alphabet. Whisper-family models regurgitate that prompt verbatim on
silence/noise, and it used to reach the dashboard as though the visitor said it.
These pin the guard: a pure prompt-echo is dropped, genuine speech still streams
(and streams promptly — held only while it could still be the prompt)."""

from __future__ import annotations

import asyncio

from harp.voice import openai as oai
from harp.voice.provider import UserTranscript

PROMPT = "The speaker uses only English or Urdu. Romanize Urdu into Latin letters."


def _conn(prompt: str = PROMPT) -> oai.OpenAIConnection:
    # Skip __init__ so no live socket / receive task is created; we only drive
    # the user-transcript bookkeeping directly.
    c = oai.OpenAIConnection.__new__(oai.OpenAIConnection)
    c._events = asyncio.Queue()
    c._prompt = prompt
    c._user_held = ""
    c._user_streaming = False
    return c


def _drained(c: oai.OpenAIConnection) -> list[UserTranscript]:
    out: list[UserTranscript] = []
    while not c._events.empty():
        out.append(c._events.get_nowait())
    return out


def test_prompt_echo_is_dropped():
    c = _conn()
    for tok in ["The speaker ", "uses only ", "English or Urdu. ", "Romanize Urdu."]:
        c._on_user_delta(tok)
    c._finish_user_turn()
    assert _drained(c) == []  # nothing ever reached the bus / dashboard


def test_split_word_echo_still_dropped():
    # Deltas can split a word ("...only Eng" | "lish"); the prefix guard is
    # normalized so a token boundary mid-word doesn't look like divergence.
    c = _conn()
    for tok in ["The speaker uses only Eng", "lish or Urdu. Romanize Urdu."]:
        c._on_user_delta(tok)
    c._finish_user_turn()
    assert _drained(c) == []


def test_real_speech_streams_and_closes():
    c = _conn()
    c._on_user_delta("Hello ")
    c._on_user_delta("there")
    c._finish_user_turn()
    assert [(e.text, e.final) for e in _drained(c)] == [
        ("Hello ", False),  # diverges immediately -> streamed at once, no lag
        ("there", False),
        ("", True),  # turn closed
    ]


def test_speech_that_shares_an_opening_word_is_kept():
    # "The weather..." starts like the prompt ("The speaker...") but diverges on
    # the second word; it must not be swallowed.
    c = _conn()
    c._on_user_delta("The ")
    c._on_user_delta("weather is nice")
    c._finish_user_turn()
    out = _drained(c)
    assert "".join(e.text for e in out) == "The weather is nice"
    assert out[-1].final
