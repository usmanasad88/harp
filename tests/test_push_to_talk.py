"""PushToTalk: a per-session hold-to-talk mode.

A press *while idle* starts a session and gates its mic (real audio only while
held); a session woken hands-free is never gated; the gate clears when HARP
returns to STANDBY. The keyboard backend (pynput) is injected as a fake, so
these tests drive `press()` / `release()` directly — no display, no keypresses.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from harp.core.bus import Bus
from harp.core.events import StateChanged, TalkKeyChanged, WakeRequested
from harp.core.state import AppState
from harp.interaction.push_to_talk import PushToTalk, _combo_handlers


class FakeListener:
    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def make_ptt(bus: Bus, **kwargs):
    """Build a PushToTalk with a fake listener; return it + the fake once armed."""
    holder: dict[str, FakeListener] = {}

    def factory(press, release):
        holder["listener"] = FakeListener(press, release)
        return holder["listener"]

    return PushToTalk(bus, listener_factory=factory, **kwargs), holder


async def _arm(ptt: PushToTalk) -> asyncio.Task:
    task = asyncio.create_task(ptt.run())
    await asyncio.sleep(0)  # let run() capture the loop, start listener, subscribe
    return task


async def _set_state(bus: Bus, new: AppState) -> None:
    await bus.publish(StateChanged(old="?", new=new.value))
    await asyncio.sleep(0.02)  # let the state watcher process it


async def _quiet(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_press_while_idle_wakes_and_gates_the_session():
    bus = Bus()
    wakes = bus.subscribe(WakeRequested)
    ptt, _ = make_ptt(bus)
    task = await _arm(ptt)
    await _set_state(bus, AppState.STANDBY)  # HARP is idle

    ptt.press()
    assert ptt.held is True and ptt.mic_open is True  # held → mic streams
    ev = await asyncio.wait_for(anext(wakes), 1.0)
    assert ev.reason == "button" and ev.context

    await _set_state(bus, AppState.ACTIVE)  # the session opened
    ptt.release()
    assert ptt.mic_open is False  # released mid push-to-talk session → silence
    ptt.press()
    assert ptt.mic_open is True   # re-pressing re-opens the gate

    # No second wake for a press during an already-open session.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(anext(wakes), 0.1)

    await _quiet(task)


async def test_session_end_returns_to_hands_free():
    bus = Bus()
    ptt, _ = make_ptt(bus)
    task = await _arm(ptt)
    await _set_state(bus, AppState.STANDBY)

    ptt.press()
    await _set_state(bus, AppState.ACTIVE)
    ptt.release()
    assert ptt.mic_open is False  # gated while the push-to-talk session runs

    await _set_state(bus, AppState.STANDBY)  # session ended
    assert ptt.mic_open is True  # gate cleared: back to hands-free (ungated)

    await _quiet(task)


async def test_hands_free_session_is_not_gated():
    """A session woken by wave/wake-word (not a press) streams normally, even if
    the key is idle or pressed mid-session."""
    bus = Bus()
    ptt, _ = make_ptt(bus)
    task = await _arm(ptt)
    await _set_state(bus, AppState.STANDBY)
    await _set_state(bus, AppState.ACTIVE)  # something else opened the session

    assert ptt.mic_open is True          # not a push-to-talk session
    ptt.press()                          # a press mid-hands-free-session
    assert ptt.mic_open is True          # still ungated
    ptt.release()
    assert ptt.mic_open is True

    await _quiet(task)


async def test_exclusive_mode_gates_every_session_to_the_key():
    """push_to_talk.exclusive: the model hears the mic ONLY while the key is
    held — even in a session woken hands-free (wave / wake word), which without
    the flag streams ungated (see test_hands_free_session_is_not_gated)."""
    bus = Bus()
    ptt, _ = make_ptt(bus, exclusive=True)
    task = await _arm(ptt)
    await _set_state(bus, AppState.STANDBY)
    assert ptt.mic_open is False            # idle: silence

    await _set_state(bus, AppState.ACTIVE)  # something else woke the session
    assert ptt.mic_open is False            # still gated, unlike default mode
    ptt.press()
    assert ptt.mic_open is True             # held → real audio flows
    ptt.release()
    assert ptt.mic_open is False            # released → back to silence

    await _set_state(bus, AppState.STANDBY)  # session over
    assert ptt.mic_open is False             # exclusive never reverts to open

    await _quiet(task)


async def test_press_before_standby_does_not_wake():
    bus = Bus()
    wakes = bus.subscribe(WakeRequested)
    ptt, _ = make_ptt(bus)
    task = await _arm(ptt)  # still STARTING — never went STANDBY

    ptt.press()
    assert ptt.held is True
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(anext(wakes), 0.1)  # no session started

    await _quiet(task)


async def test_key_autorepeat_wakes_only_once():
    bus = Bus()
    wakes = bus.subscribe(WakeRequested)
    ptt, _ = make_ptt(bus)
    task = await _arm(ptt)
    await _set_state(bus, AppState.STANDBY)

    ptt.press()
    ptt.press()  # held keys auto-repeat key-down; must not re-wake
    ptt.press()

    assert await asyncio.wait_for(anext(wakes), 1.0)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(anext(wakes), 0.1)

    await _quiet(task)


class FakeClock:
    """Deterministic stand-in for time.monotonic — advance .now by hand."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_release_debounce_bridges_a_tap_train():
    """The ESP32 BLE arcade button can't hold: its firmware re-taps the combo
    (~95 ms down, ~300 ms up, ~2.5x/s) while pressed. With a release debounce
    longer than the tap gap, the train must read as ONE continuous hold, and
    the gate must close only after the window expires past the last tap."""
    clock = FakeClock()
    ptt, _ = make_ptt(
        Bus(), exclusive=True, release_debounce_seconds=0.7, clock=clock
    )

    ptt.press()                      # tap 1 lands
    assert ptt.mic_open is True
    clock.now += 0.095
    ptt.release()                    # tap 1 lifts — the ~300 ms gap begins
    assert ptt.mic_open is True      # bridged: still inside the window
    clock.now += 0.3
    ptt.press()                      # tap 2 — the "hold" continues
    clock.now += 0.095
    ptt.release()
    assert ptt.mic_open is True

    clock.now += 0.71                # no tap 3: the button was really let go
    assert ptt.mic_open is False     # window expired → gate closed


async def test_tap_train_in_standby_wakes_only_once():
    """Every tap of the train is a press-while-STANDBY; without dedupe each one
    would publish its own WakeRequested. Only a press outside the debounce
    window is a NEW hold that may wake."""
    clock = FakeClock()
    bus = Bus()
    wakes: list[WakeRequested] = []

    async def collect():
        async for ev in bus.subscribe(WakeRequested):
            wakes.append(ev)

    collector = asyncio.create_task(collect())
    ptt, _ = make_ptt(bus, release_debounce_seconds=0.7, clock=clock)
    task = await _arm(ptt)
    await _set_state(bus, AppState.STANDBY)

    ptt.press()                      # tap 1: wakes
    clock.now += 0.095
    ptt.release()
    clock.now += 0.3
    ptt.press()                      # tap 2, same hold: must not re-wake
    clock.now += 0.095
    ptt.release()
    await asyncio.sleep(0.05)        # let the scheduled wake publish + collect
    assert [ev.reason for ev in wakes] == ["button"]

    clock.now += 1.0                 # well past the window: a genuine new press
    ptt.press()
    await asyncio.sleep(0.05)
    assert [ev.reason for ev in wakes] == ["button", "button"]

    await _quiet(collector)
    await _quiet(task)


async def test_talk_key_changes_are_mirrored_to_the_bus():
    """The end-user page's "Listening" screen renders TalkKeyChanged — the key
    must be mirrored on every edge, in any app state (a mid-session press that
    doesn't wake anything still lights the screen)."""
    bus = Bus()
    keys = bus.subscribe(TalkKeyChanged)
    ptt, _ = make_ptt(bus)
    task = await _arm(ptt)

    ptt.press()
    ev = await asyncio.wait_for(anext(keys), 1.0)
    assert ev.held is True
    ptt.release()
    ev = await asyncio.wait_for(anext(keys), 1.0)
    assert ev.held is False

    await _quiet(task)


async def test_talk_key_events_bridge_the_tap_train():
    """The arcade button's firmware re-taps while pressed (see the release-
    debounce tests above). The bus mirror must not flicker with it: one
    held=True at the first tap, NO held=False inside the window, and the
    held=False only once the window truly expires past the last key-up.
    Real clock — the close is settled by a loop timer, which is the thing
    under test."""
    bus = Bus()
    events: list[bool] = []

    async def collect():
        async for ev in bus.subscribe(TalkKeyChanged):
            events.append(ev.held)

    collector = asyncio.create_task(collect())
    ptt, _ = make_ptt(bus, release_debounce_seconds=0.2)
    task = await _arm(ptt)

    ptt.press()                      # tap 1
    await asyncio.sleep(0.05)
    ptt.release()                    # gap begins — inside the 0.2 s window
    await asyncio.sleep(0.05)
    ptt.press()                      # tap 2: the "hold" continues
    await asyncio.sleep(0.05)
    ptt.release()                    # last key-up; window starts over
    await asyncio.sleep(0.05)        # still inside it
    assert events == [True]          # no flicker mid-train

    await asyncio.sleep(0.4)         # well past the window: really let go
    assert events == [True, False]

    await _quiet(collector)
    await _quiet(task)


async def test_cancel_stops_the_listener():
    bus = Bus()
    ptt, holder = make_ptt(bus)
    task = await _arm(ptt)

    await _quiet(task)
    assert holder["listener"].stopped is True
    assert ptt.held is False


def test_combo_holds_only_while_every_key_is_down():
    """The ctrl+shift+m state machine: fires press exactly when the last combo
    key lands, release when ANY combo key lifts — with OS auto-repeat of a held
    key and non-combo keys unable to re-fire or leak through. Strings stand in
    for pynput keys; upper→lower plays the role of Listener.canonical()."""
    events: list[str] = []
    press, release = _combo_handlers(
        frozenset({"ctrl", "shift", "m"}),
        str.lower,
        lambda: events.append("press"),
        lambda: events.append("release"),
    )

    press("ctrl")
    press("shift")
    assert events == []                    # partial combo isn't a hold
    press("M")                             # canonical() folds case
    assert events == ["press"]
    press("m")                             # OS auto-repeat while held
    press("x")                             # unrelated key mid-hold
    assert events == ["press"]

    release("shift")                       # any combo key up ends the hold
    assert events == ["press", "release"]
    release("m")
    release("ctrl")
    assert events == ["press", "release"]  # no double release

    press("m")                             # modifiers no longer down
    assert events == ["press", "release"]
    press("shift")
    press("ctrl")
    assert events == ["press", "release", "press"]  # full combo again re-arms


async def test_listener_that_fails_to_start_disables_ptt_without_crashing():
    bus = Bus()

    def factory(press, release):
        raise RuntimeError("no display / no input permission")

    ptt = PushToTalk(bus, listener_factory=factory)
    await asyncio.wait_for(ptt.run(), 1.0)  # returns cleanly, does not propagate
    assert ptt.held is False
