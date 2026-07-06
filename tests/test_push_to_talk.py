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
from harp.core.events import StateChanged, WakeRequested
from harp.core.state import AppState
from harp.interaction.push_to_talk import PushToTalk


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


def make_ptt(bus: Bus):
    """Build a PushToTalk with a fake listener; return it + the fake once armed."""
    holder: dict[str, FakeListener] = {}

    def factory(press, release):
        holder["listener"] = FakeListener(press, release)
        return holder["listener"]

    return PushToTalk(bus, listener_factory=factory), holder


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


async def test_cancel_stops_the_listener():
    bus = Bus()
    ptt, holder = make_ptt(bus)
    task = await _arm(ptt)

    await _quiet(task)
    assert holder["listener"].stopped is True
    assert ptt.held is False


async def test_listener_that_fails_to_start_disables_ptt_without_crashing():
    bus = Bus()

    def factory(press, release):
        raise RuntimeError("no display / no input permission")

    ptt = PushToTalk(bus, listener_factory=factory)
    await asyncio.wait_for(ptt.run(), 1.0)  # returns cleanly, does not propagate
    assert ptt.held is False
