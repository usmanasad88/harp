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

**Exclusive mode** (`push_to_talk.exclusive` in harp.yaml) makes the button the
whole interface. Two things change, both wired in app.py: sessions START only
from the button — the always-on wake listener isn't run at all, and the
orchestrator's wake policy vetoes every non-button WakeRequested (waves
included) — and the model hears the mic ONLY while the key is held, in every
session. This class itself is unchanged by the flag beyond `mic_open`; the
button-only waking lives in the orchestrator's `wake_allowed` policy.

**Release debounce** (`push_to_talk.release_debounce_seconds`) exists for
hardware talk buttons that can't hold: HARP's arcade button is an ESP32 BLE
keyboard whose firmware re-TAPS the whole combo ~2.5x/s while pressed (all keys
down ~95 ms, up ~300 ms — verified with spike_ptt_gate.py --debug-keys) instead
of holding it down. With a debounce longer than the tap gap, a key-up followed
by a re-press inside the window reads as one continuous hold, so the tap train
becomes the hold the firmware can't send. The cost: the mic closes only after
the window expires, so ~a debounce-worth of trailing room audio passes after
each real release. 0 (the default) disables it — a plain keyboard needs none.

The keyboard backend (pynput, a global key listener on its own thread) is
injected via `listener_factory`, so this class is pynput-agnostic and unit-
testable by driving `press()` / `release()` directly. The default factory
imports pynput lazily, so importing this module never requires it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Protocol

from ..config import load_wake_context_push_to_talk
from ..core.bus import Bus
from ..core.events import StateChanged, TalkKeyChanged, WakeRequested
from ..core.state import AppState

logger = logging.getLogger(__name__)

# Delivered to the model at session open. Push-to-talk means the person presses,
# THEN speaks — so tell the model to listen first rather than launch into a long
# welcome the user would just talk over. Wording lives in
# prompts/wake_context_push_to_talk.md (see prompts/README.md).
_PTT_CONTEXT = load_wake_context_push_to_talk()


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
        exclusive: bool = False,
        release_debounce_seconds: float = 0.0,
        context: str = _PTT_CONTEXT,
        listener_factory: ListenerFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus = bus
        self._exclusive = exclusive
        self._release_debounce = release_debounce_seconds
        self._clock = clock
        self._context = context
        self._listener_factory = listener_factory or _pynput_factory(key)
        # All set from the key-listener thread and/or the state watcher, read
        # from the asyncio side (the mic gate). Bare bools/floats are fine:
        # single-word reads/writes are atomic under the GIL, and a one-chunk
        # (~64 ms) lag is inaudible.
        self._held = False           # is the talk key down right now
        self._released_at = float("-inf")  # clock time of the last key-up
        self._ptt_session = False    # was the current session started by a press
        self._standby = False        # is HARP idle (a press may start a session)
        self._held_published = False  # last TalkKeyChanged.held put on the bus
        self._loop: asyncio.AbstractEventLoop | None = None

    def _within_debounce(self) -> bool:
        """Did the last key-up land less than the debounce window ago? While
        true, a re-tapping button is still mid-"hold" (see module docstring)."""
        return self._clock() - self._released_at < self._release_debounce

    @property
    def held(self) -> bool:
        """True while the talk key is down — or, with a release debounce, until
        the window after the last key-up expires (a tap train reads as a hold)."""
        return self._held or self._within_debounce()

    @property
    def mic_open(self) -> bool:
        """Whether the voice bridge should stream real mic audio right now.
        Exclusive mode: only while held, in every session. Otherwise: always,
        unless *this* is a push-to-talk session, in which case only while held."""
        if self._exclusive:
            return self.held
        return (not self._ptt_session) or self.held

    def press(self) -> None:
        """Key-down. Safe to call from the listener thread."""
        if self._held:  # key auto-repeat while held → act once per real press
            return
        # A press inside the debounce window is a tap-train continuation of a
        # hold that already did its waking, not a new button press.
        new_hold = not self._within_debounce()
        self._held = True
        self._publish_held()  # before the wake, so the page greens on the press
        # Only a press while idle opens a session (and marks it push-to-talk); a
        # press during an already-running session just re-opens the mic gate.
        if self._standby and new_hold:
            self._ptt_session = True
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._wake(), self._loop)

    def release(self) -> None:
        """Key-up. Safe to call from the listener thread."""
        self._held = False
        self._released_at = self._clock()
        if not self._within_debounce():
            self._publish_held()  # no debounce: the hold really ended now
        elif self._loop is not None:
            # Debounced: the hold only ends if the window expires without a
            # re-press. Settle that on the loop, where call_later lives.
            self._loop.call_soon_threadsafe(self._settle_release)

    def _publish_held(self) -> None:
        """Mirror the effective hold onto the bus (TalkKeyChanged) when it
        changed. Dedupes, so a tap-train re-press inside the debounce window
        publishes nothing. Thread-safe: run_coroutine_threadsafe works from the
        listener thread and the loop thread alike, and preserves order."""
        held = self.held
        if self._loop is None or held == self._held_published:
            return
        self._held_published = held
        asyncio.run_coroutine_threadsafe(
            self._bus.publish(TalkKeyChanged(held=held)), self._loop
        )

    def _settle_release(self) -> None:
        """Loop-side: wait out the debounce window after a key-up, then publish
        the close if no re-press bridged it. Re-arms itself while the window is
        still open (each newer key-up pushes `_released_at` forward)."""
        if self._held or self._loop is None:
            return  # re-pressed: the hold continues; the next key-up re-arms us
        remaining = self._release_debounce - (self._clock() - self._released_at)
        if remaining > 0:
            self._loop.call_later(remaining, self._settle_release)
            return
        self._publish_held()

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
            self._released_at = float("-inf")
            self._ptt_session = False
            self._publish_held()  # best-effort: unlight any connected page
            listener.stop()


_KEY_ALIASES = {
    "space": "space",
    "spacebar": "space",
    "enter": "enter",
    "return": "enter",
    "control": "ctrl",
    "option": "alt",
    "win": "cmd",
    "windows": "cmd",
    "super": "cmd",
}


def _pynput_factory(key: str) -> ListenerFactory:
    """Build the real (pynput) listener factory, matching one configured key or
    '+'-joined combo (e.g. 'ctrl+shift+m').

    pynput is imported here, not at module top, so `import push_to_talk` and the
    unit tests never need it or a display."""

    def factory(on_press: Callable[[], None], on_release: Callable[[], None]) -> _KeyListener:
        from pynput import keyboard  # lazy: only when actually arming the mic

        combo = _resolve_combo(keyboard, key)
        # canonical() folds left/right modifier variants (ctrl_l → ctrl) and
        # strips modifier effects from characters (Ctrl+M's '\r' → 'm'), so
        # what the OS reports matches the combo members parsed from harp.yaml.
        handle_press, handle_release = _combo_handlers(
            combo, lambda k: listener.canonical(k), on_press, on_release
        )
        listener = keyboard.Listener(on_press=handle_press, on_release=handle_release)
        return listener

    return factory


def _combo_handlers(
    combo: frozenset,
    canonical: Callable,
    on_press: Callable[[], None],
    on_release: Callable[[], None],
) -> tuple[Callable, Callable]:
    """Turn per-key press/release events into hold-the-whole-combo semantics:
    on_press fires when the last combo key lands, on_release when any combo key
    lifts. Keys outside the combo are ignored; OS auto-repeat of a held key
    can't re-fire because activation is edge-triggered."""
    down: set = set()
    active = False

    def handle_press(key) -> None:
        nonlocal active
        key = canonical(key)
        if key in combo:
            down.add(key)
            if not active and len(down) == len(combo):
                active = True
                on_press()

    def handle_release(key) -> None:
        nonlocal active
        key = canonical(key)
        if key in combo:
            down.discard(key)
            if active:
                active = False
                on_release()

    return handle_press, handle_release


def _resolve_combo(keyboard, key: str) -> frozenset:
    """Map a harp.yaml key spec — one key ('space', 'm') or a '+'-joined combo
    ('ctrl+shift+m') — to the set of pynput keys that must be held together.
    A spec with any unrecognized part falls back to space with a warning."""
    keys = set()
    for part in key.split("+"):
        name = _KEY_ALIASES.get(part.strip().lower(), part.strip().lower())
        special = getattr(keyboard.Key, name, None)
        if special is not None:
            keys.add(special)
        elif len(name) == 1:
            keys.add(keyboard.KeyCode.from_char(name))
        else:
            logger.warning("push_to_talk.key=%r not recognized; using space", key)
            return frozenset({keyboard.Key.space})
    return frozenset(keys)
