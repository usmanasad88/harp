"""Keyboard push-to-talk — an on-demand hold-to-talk session for noisy places.

Real-time voice is fragile in loud environments (an expo hall): background
noise causes false wakes and bleeds into the model's turn-taking. Push-to-talk
lets you start a conversation the room can't interfere with — the mic only
reaches the model while a key is held.

It is a **per-session mode**, not a global switch:

  - At start-up push-to-talk is *inactive*. The always-on wake listener and the
    wave trigger run as usual, so HARP still wakes hands-free.
  - Pressing the key **while idle (STANDBY)** starts a session
    (`WakeRequested(reason="button")`) AND marks *this* session push-to-talk:
    for its whole duration the mic is gated — real audio only while the key is
    held, silence otherwise.
  - When that session ends (face-absence timeout, the agent's own end_session
    tool, an error — anything that returns HARP to STANDBY) push-to-talk goes
    inactive again and the listener / wave resume.

A session that was woken hands-free (wave, wake word, loud sound) is *not*
gated — `mic_open` returns True throughout it, so it streams normally. Only a
press-started session holds the gate.

The keyboard backend (pynput, a global key listener on its own thread) is
injected via `listener_factory`, so this class is pynput-agnostic and unit-
testable by driving `press()` / `release()` directly. The default factory
imports pynput lazily, so importing this module never requires it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Protocol

from ..core.bus import Bus
from ..core.events import StateChanged, WakeRequested
from ..core.state import AppState

logger = logging.getLogger(__name__)

# Delivered to the model at session open. Push-to-talk means the person presses,
# THEN speaks — so tell the model to listen first rather than launch into a long
# welcome the user would just talk over.
_PTT_CONTEXT = (
    "(A person started a push-to-talk conversation: they hold a button while "
    "they speak, and are about to say something. Listen for it and answer. A "
    "very short greeting is fine, but don't give a long welcome before they "
    "have spoken.)"
)


class _KeyListener(Protocol):
    """The slice of pynput.keyboard.Listener this subsystem uses."""

    def start(self) -> None: ...
    def stop(self) -> None: ...


# (on_press, on_release) -> a started-elsewhere key listener.
ListenerFactory = Callable[[Callable[[], None], Callable[[], None]], _KeyListener]


class PushToTalk:
    """Hold a key to talk. A press while idle starts a gated session; the gate
    (real mic audio vs. silence) follows the key until that session ends."""

    def __init__(
        self,
        bus: Bus,
        key: str = "space",
        context: str = _PTT_CONTEXT,
        listener_factory: ListenerFactory | None = None,
    ) -> None:
        self._bus = bus
        self._context = context
        self._listener_factory = listener_factory or _pynput_factory(key)
        # All set from the key-listener thread and/or the state watcher, read
        # from the asyncio side (the mic gate). Bare bools are fine: single-word
        # reads/writes are atomic under the GIL, and a one-chunk (~64 ms) lag is
        # inaudible.
        self._held = False           # is the talk key down right now
        self._ptt_session = False    # was the current session started by a press
        self._standby = False        # is HARP idle (a press may start a session)
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def held(self) -> bool:
        """True while the talk key is down."""
        return self._held

    @property
    def mic_open(self) -> bool:
        """Whether the voice bridge should stream real mic audio right now: always,
        unless *this* is a push-to-talk session, in which case only while held."""
        return (not self._ptt_session) or self._held

    def press(self) -> None:
        """Key-down. Safe to call from the listener thread."""
        if self._held:  # key auto-repeat while held → act once per real press
            return
        self._held = True
        # Only a press while idle opens a session (and marks it push-to-talk); a
        # press during an already-running session just re-opens the mic gate.
        if self._standby:
            self._ptt_session = True
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._wake(), self._loop)

    def release(self) -> None:
        """Key-up. Safe to call from the listener thread."""
        self._held = False

    async def _wake(self) -> None:
        await self._bus.publish(WakeRequested(reason="button", context=self._context))

    async def run(self) -> None:
        """Arm the key listener and track state until cancelled. Watching
        StateChanged is what makes push-to-talk per-session: returning to
        STANDBY (session over) clears the gate so the listener/wave take over."""
        self._loop = asyncio.get_running_loop()
        try:
            listener = self._listener_factory(self.press, self.release)
            listener.start()
        except Exception:
            # A missing display / permissions problem shouldn't crash HARP; it
            # just means no push-to-talk this run.
            logger.exception("push-to-talk key listener failed to start — disabled")
            return
        logger.info("push-to-talk armed — hold the talk key to start a session")
        try:
            async for ev in self._bus.subscribe(StateChanged):
                self._standby = ev.new == AppState.STANDBY.value
                if self._standby:
                    # Session ended (or boot finished): drop back to hands-free.
                    self._ptt_session = False
        finally:
            self._held = False
            self._ptt_session = False
            listener.stop()


_KEY_ALIASES = {"space": "space", "spacebar": "space", "enter": "enter", "return": "enter"}


def _pynput_factory(key: str) -> ListenerFactory:
    """Build the real (pynput) listener factory, matching one configured key.

    pynput is imported here, not at module top, so `import push_to_talk` and the
    unit tests never need it or a display."""

    def factory(on_press: Callable[[], None], on_release: Callable[[], None]) -> _KeyListener:
        from pynput import keyboard  # lazy: only when actually arming the mic

        target = _resolve_key(keyboard, key)

        def matches(pressed) -> bool:
            return pressed == target or getattr(pressed, "char", None) == getattr(
                target, "char", object()
            )

        return keyboard.Listener(
            on_press=lambda k: on_press() if matches(k) else None,
            on_release=lambda k: on_release() if matches(k) else None,
        )

    return factory


def _resolve_key(keyboard, key: str):
    """Map a harp.yaml key name ('space', 'enter', or a single character) to a
    pynput key object. Unknown names fall back to space with a warning."""
    name = _KEY_ALIASES.get(key.strip().lower(), key.strip().lower())
    special = getattr(keyboard.Key, name, None)
    if special is not None:
        return special
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    logger.warning("push_to_talk.key=%r not recognized; using space", key)
    return keyboard.Key.space
