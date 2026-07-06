"""TwoAgentBridge: a noise/intent filter in front of the responder.

This is the opt-in two-agent mode (harp.yaml `filter_agent.enabled`). It exposes
the SAME `run(context)` interface as VoiceBridge, so the orchestrator drives it
interchangeably — app.py just builds this instead of a plain VoiceBridge when the
flag is on, and nothing else in the supervisor changes.

Inside, it composes two live sessions:

    mic ─audio─▶  FilterAgent  ─clean text─▶  VoiceBridge (responder) ─▶ speaker
   (Agent 1: hears the room,   (Agent 2: today's bridge, but fed by injected
    relays only what's meant    text turns instead of a mic — it never hears the
    for HARP)                   raw room, only "silence, then a clean message")

Two wires connect them:

  - relay: each message the filter approves is (a) published as `UserSaid` so the
    dashboard shows the filtered user turn in the transcript, and (b) pushed to
    the responder's text inbox, which makes it reply.
  - feedback: the responder's finished reply is fed back into the filter as a
    CONTEXT note, so short follow-ups ("yes", "how much") are interpretable.

Half-duplex (the chosen design): while the responder is speaking, the filter's
mic is muted (the gate returns False → the filter agent sends silence), so HARP
never relays its own voice back to itself. A visitor therefore can't interrupt a
reply mid-sentence — the robust trade-off for a loud room. The mute is lifted a
short tail after the reply's transcript finalizes, to cover audio still draining
from the speaker.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable

from ..core.bus import Bus
from ..core.events import AgentSaid, UserSaid
from .audio_io import Microphone, Speaker
from .bridge import VoiceBridge
from .filter_agent import FilterAgent
from .provider import SessionConfig

logger = logging.getLogger(__name__)

ToolDispatch = Callable[[str, dict], Awaitable[Any]]

# Safety cap: if the responder never answers a relayed message (e.g. an error
# swallowed the turn), don't leave the filter muted forever — reopen its mic
# this long after a relay even if no reply was seen.
_RELAY_GRACE_SECONDS = 12.0


class TwoAgentBridge:
    def __init__(
        self,
        bus: Bus,
        provider_name: str,
        make_config: Callable[[], SessionConfig],
        make_filter_config: Callable[[], SessionConfig],
        tool_dispatch: ToolDispatch | None = None,
        identity_context: Callable[[], str] | None = None,
        filter_provider_name: str | None = None,
        response_tail_seconds: float = 0.8,
        external_mic_gate: Callable[[], bool] | None = None,
        near_field_level: Callable[[], float] | None = None,
        # Injection seams (tests pass fakes; None = the real backends/devices).
        responder_provider=None,
        filter_provider=None,
        mic_factory: Callable[[int], Microphone] = Microphone,
        speaker_factory: Callable[[int], Speaker] = Speaker,
    ) -> None:
        self._bus = bus
        self._provider_name = provider_name
        self._make_config = make_config
        self._make_filter_config = make_filter_config
        self._tool_dispatch = tool_dispatch
        self._identity_context = identity_context
        self._filter_provider_name = filter_provider_name or provider_name
        self._response_tail = response_tail_seconds
        self._external_mic_gate = external_mic_gate
        self._near_field_level = near_field_level
        self._responder_provider = responder_provider
        self._filter_provider = filter_provider
        self._mic_factory = mic_factory
        self._speaker_factory = speaker_factory

        # Per-session state (re-initialized in run(); set here so the wiring
        # methods are directly unit-testable without opening a session).
        self._to_responder: asyncio.Queue[str] = asyncio.Queue()
        self._filter: FilterAgent | None = None
        self._speaking = False       # responder is mid-reply (deltas seen, no final yet)
        self._busy_until = 0.0        # monotonic deadline the mic stays muted until
        self._agent_buf = ""          # accumulates the responder's reply for feedback

    # --- the half-duplex mic gate the filter consults ------------------------
    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def _is_busy(self) -> bool:
        """True while the responder is (or was just) speaking — the window in
        which the filter's mic must stay muted."""
        return self._speaking or self._now() < self._busy_until

    def _filter_mic_gate(self) -> bool:
        """The filter agent's mic is live only when the responder is idle AND any
        external gate (push-to-talk) also allows it."""
        if self._external_mic_gate is not None and not self._external_mic_gate():
            return False
        return not self._is_busy()

    # --- relay: filter approved a message → drive the responder --------------
    async def _relay(self, text: str) -> None:
        # Mute the filter's mic immediately (bridge the gap until the responder's
        # first audio; _watch_responder takes over once it starts speaking).
        self._busy_until = self._now() + _RELAY_GRACE_SECONDS
        # Show the filtered user turn on the dashboard (the responder gets text,
        # not audio, so it emits no user transcript of its own).
        await self._bus.publish(UserSaid(text=text, final=True))
        await self._to_responder.put(text)

    # --- feedback + speaking-state tracking from the responder ---------------
    async def _watch_responder(self) -> None:
        """Follow the responder's AgentSaid stream to (1) know when it's speaking,
        for the half-duplex gate, and (2) feed each finished reply back to the
        filter as context."""
        async for ev in self._bus.subscribe(AgentSaid):
            if ev.final:
                self._speaking = False
                self._busy_until = self._now() + self._response_tail
                reply = (self._agent_buf + ev.text).strip()
                self._agent_buf = ""
                if reply and self._filter is not None:
                    self._filter.add_context(f"You (the assistant) just said: {reply}")
            else:
                self._speaking = True
                self._agent_buf += ev.text

    # --- the VoiceBridge-compatible entry point ------------------------------
    async def run(self, context: str = "") -> None:
        # Fresh state for this session.
        self._to_responder = asyncio.Queue()
        self._speaking = False
        self._busy_until = 0.0
        self._agent_buf = ""

        self._filter = FilterAgent(
            self._bus,
            self._filter_provider_name,
            self._make_filter_config,
            on_relay=self._relay,
            mic_gate=self._filter_mic_gate,
            near_field_level=self._near_field_level,
            provider=self._filter_provider,
            mic_factory=self._mic_factory,
        )
        responder = VoiceBridge(
            self._bus,
            self._provider_name,
            make_config=self._make_config,
            tool_dispatch=self._tool_dispatch,
            identity_context=self._identity_context,
            text_inbox=self._to_responder,
            provider=self._responder_provider,
            mic_factory=self._mic_factory,       # unused: text-driven responder opens no mic
            speaker_factory=self._speaker_factory,
        )

        logger.info(
            "two-agent session opening (filter=%s, responder=%s)",
            self._filter_provider_name, self._provider_name,
        )
        watcher = asyncio.create_task(self._watch_responder(), name="two_agent.watch")
        sessions = [
            # The filter gets NO wake context: that context is written for the
            # responder ("someone said hello — greet them and offer help") and
            # would prime the filter to hallucinate greetings on silence. The
            # filter's only input is live mic audio; the responder still gets the
            # full context so it can open the conversation properly.
            asyncio.create_task(self._filter.run(""), name="two_agent.filter"),
            asyncio.create_task(responder.run(context), name="two_agent.responder"),
        ]
        try:
            # First session to end (provider closed the stream, or a crash) ends
            # the whole interaction — same first-exit-wins shape app.py uses.
            done, _ = await asyncio.wait(sessions, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()  # re-raise a crash so the orchestrator's error path runs
        finally:
            for task in (watcher, *sessions):
                task.cancel()
            await asyncio.gather(watcher, *sessions, return_exceptions=True)
            logger.info("two-agent session closed")
