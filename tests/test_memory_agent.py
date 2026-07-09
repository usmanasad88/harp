"""The shared Flash Lite helper (harp/memory/agent): the rate limiter must
actually slide (a stuck window either locks the quota out forever or blows
through it), and generate() must NEVER raise into a consumer — app.py treats
a finished runner task as a crash, so one API hiccup raising here would take
the whole agent down."""

from __future__ import annotations

import asyncio

from harp.memory.agent import GeminiAgent, RateLimiter, parse_json


def test_rate_limiter_window_slides():
    t = 0.0
    limiter = RateLimiter(3, period=60.0, clock=lambda: t)
    assert limiter.try_acquire()  # recorded at t=0
    t = 30.0
    assert limiter.try_acquire() and limiter.try_acquire()  # window full at t=30
    t = 59.9
    assert not limiter.try_acquire()
    t = 60.0  # only the t=0 call ages out
    assert limiter.try_acquire()
    assert not limiter.try_acquire()  # the two t=30 calls + the new one fill it


async def test_acquire_waits_for_a_freed_slot_and_respects_timeout():
    limiter = RateLimiter(1, period=0.2)
    assert limiter.try_acquire()
    # Too short to see the slot free: gives up rather than hanging.
    assert not await limiter.acquire(timeout=0.05)
    # Long enough: picks the slot up as soon as the period lapses.
    assert await limiter.acquire(timeout=2.0)


async def test_generate_returns_none_instead_of_raising():
    async def boom(prompt, image, json_response):
        raise RuntimeError("api down")

    agent = GeminiAgent("m", RateLimiter(10), caller=boom)
    assert await agent.generate("hi") is None

    async def hangs(prompt, image, json_response):
        await asyncio.sleep(5)
        return "too late"

    agent = GeminiAgent("m", RateLimiter(10), caller=hangs)
    assert await agent.generate("hi", timeout=0.05) is None


async def test_generate_skips_when_the_shared_budget_is_spent():
    calls = 0

    async def ok(prompt, image, json_response):
        nonlocal calls
        calls += 1
        return "text"

    agent = GeminiAgent("m", RateLimiter(1, period=60.0), caller=ok)
    assert await agent.generate("one") == "text"
    # wait=False (background callers): no slot -> no call, not a queued one.
    assert await agent.generate("two") is None
    assert calls == 1


def test_parse_json_tolerates_fences_and_rejects_non_objects():
    # Gemini routinely wraps JSON in a markdown fence — that must still parse.
    assert parse_json('```json\n{"summary": "x"}\n```') == {"summary": "x"}
    assert parse_json('{"summary": "x"}') == {"summary": "x"}
    assert parse_json("not json at all") is None
    assert parse_json("[1, 2]") is None
