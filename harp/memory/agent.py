"""The parallel helper agent: one Gemini Flash Lite client, three consumers.

Runs alongside the realtime voice session, never inside it. The summarizer
(memory/summarizer) has it turn a finished conversation into long-term memory;
the context writer (memory/context) has it pre-compute the wake briefing from
a camera frame + stored memories; the live model's describe_scene tool
(vision/describe) has it describe the current frame mid-session.

Two design rules, both enforced here so no consumer can violate them:

  - **Fail-safe**: `generate()` returns None on ANY failure — rate limit, SDK
    error, timeout, no API key — and logs it. It never raises into a consumer;
    every consumer has a defined degradation for None (leave the transcript
    pending, serve the static identity line, tell the model the camera helper
    is unavailable).
  - **One shared rate limit** (harp.yaml `memory.calls_per_minute`, default 14
    — the free-tier budget): a sliding 60 s window across all consumers, so a
    burst of interactions can't starve the quota. Callers that can retry later
    skip when the window is full (`wait=False`); callers with a model waiting
    on them block for a slot (`wait=True`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Caller signature tests inject instead of the real SDK:
# (prompt, image_jpeg, json_response) -> the model's text reply.
Caller = Callable[[str, "bytes | None", bool], Awaitable[str]]


class RateLimiter:
    """Sliding-window call budget: at most `max_calls` starts per `period`
    seconds, shared by everyone holding this instance."""

    def __init__(
        self,
        max_calls: int,
        period: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max(1, int(max_calls))
        self._period = period
        self._clock = clock
        self._calls: deque[float] = deque()

    def try_acquire(self) -> bool:
        """Claim a slot now if one is free; False (and no slot) otherwise."""
        now = self._clock()
        while self._calls and now - self._calls[0] >= self._period:
            self._calls.popleft()
        if len(self._calls) >= self._max:
            return False
        self._calls.append(now)
        return True

    async def acquire(self, timeout: float | None = None) -> bool:
        """Wait for a slot, up to `timeout` seconds (None = forever)."""
        deadline = None if timeout is None else self._clock() + timeout
        while not self.try_acquire():
            now = self._clock()
            if deadline is not None and now >= deadline:
                return False
            # The oldest recorded call ages out of the window first.
            wait = self._calls[0] + self._period - now if self._calls else self._period
            if deadline is not None:
                wait = min(wait, deadline - now)
            await asyncio.sleep(max(wait, 0.01))
        return True


class GeminiAgent:
    """The shared Flash Lite client. `caller` is injectable so consumers are
    tested without the SDK or a key; the default lazily builds a real
    google-genai client on first use."""

    def __init__(self, model: str, limiter: RateLimiter, caller: Caller | None = None) -> None:
        self._model = model
        self._limiter = limiter
        self._caller = caller or self._real_call
        self._client = None

    async def generate(
        self,
        prompt: str,
        *,
        image_jpeg: bytes | None = None,
        json_response: bool = False,
        wait: bool = False,
        timeout: float = 30.0,
    ) -> str | None:
        """One model call. `wait=False` skips immediately when the shared
        rate window is full (for callers that can retry later); `wait=True`
        blocks for a slot up to `timeout` (for callers a model is waiting on).
        `json_response` asks the model for a JSON body (parse it with
        `parse_json`). Returns the reply text, or None on any failure."""
        if wait:
            got_slot = await self._limiter.acquire(timeout=timeout)
        else:
            got_slot = self._limiter.try_acquire()
        if not got_slot:
            logger.info("memory agent: call budget exhausted, skipping a %s call",
                        "waited" if wait else "background")
            return None
        try:
            text = await asyncio.wait_for(
                self._caller(prompt, image_jpeg, json_response), timeout
            )
        except Exception:
            logger.warning("memory agent: generate failed", exc_info=True)
            return None
        text = (text or "").strip()
        return text or None

    async def _real_call(self, prompt: str, image_jpeg: bytes | None, json_response: bool) -> str:
        # Imported here so tests (which always inject a caller) never touch the
        # SDK, and a missing key fails per-call (a logged None) not at wiring.
        from google import genai
        from google.genai import types

        from ..config import require_key

        if self._client is None:
            self._client = genai.Client(api_key=require_key("GEMINI_API_KEY"))
        contents: list = []
        if image_jpeg is not None:
            contents.append(types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"))
        contents.append(prompt)
        config = (
            types.GenerateContentConfig(response_mime_type="application/json")
            if json_response
            else None
        )
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        return response.text or ""


def parse_json(text: str) -> dict | None:
    """Parse a model reply that should be a JSON object, tolerating a markdown
    code fence around it; None when it isn't a parseable object (callers
    degrade — e.g. the summarizer stores the raw text as the summary)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
