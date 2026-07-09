"""Dashboard server: test it against a real bus and a real WebSocket client,
on an ephemeral port — no browser needed."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import pytest
import websockets

from harp.core.bus import Bus
from harp.core.events import UserSaid
from harp.dashboard.server import _build_server


@pytest.fixture
async def running_server():
    bus = Bus()
    async with _build_server(bus, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield bus, port


async def test_publishes_bus_event_to_connected_client(running_server):
    bus, port = running_server
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as client:
        await bus.publish(UserSaid(text="hello", final=True))

        raw = await asyncio.wait_for(client.recv(), timeout=1)
        msg = json.loads(raw)

        assert msg["type"] == "UserSaid"
        assert msg["fields"] == {"text": "hello", "final": True}
        assert "server_ts" in msg


async def test_each_connected_client_gets_its_own_stream(running_server):
    bus, port = running_server
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as a:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as b:
            await bus.publish(UserSaid(text="hi", final=True))

            msg_a = json.loads(await asyncio.wait_for(a.recv(), timeout=1))
            msg_b = json.loads(await asyncio.wait_for(b.recv(), timeout=1))
            assert msg_a["fields"]["text"] == "hi"
            assert msg_b["fields"]["text"] == "hi"


async def test_disconnecting_client_is_cleaned_up_from_the_bus(running_server):
    bus, port = running_server
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as client:
        await bus.publish(UserSaid(text="first", final=True))
        await asyncio.wait_for(client.recv(), timeout=1)

    async def _wait_no_subscribers():
        while len(bus._subscribers) != 0:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_wait_no_subscribers(), timeout=1)


def _get(url: str):
    """`urllib` is blocking, so run it off-thread — otherwise it would freeze
    the same event loop the (asyncio) server needs to answer it, and hang."""
    return urllib.request.urlopen(url, timeout=2)


async def test_serves_the_static_index_page_over_http(running_server):
    _bus, port = running_server
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, _get, f"http://127.0.0.1:{port}/")
    with resp:
        assert resp.status == 200
        assert "text/html" in resp.headers["Content-Type"]
        body = resp.read().decode()
        assert "HARP" in body


async def test_unknown_path_returns_404(running_server):
    _bus, port = running_server
    loop = asyncio.get_running_loop()
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        await loop.run_in_executor(None, _get, f"http://127.0.0.1:{port}/nope")
    assert exc_info.value.code == 404


async def test_camera_route_serves_snapshot_and_ignores_cache_buster():
    bus = Bus()
    fake_jpeg = b"\xff\xd8not-really-a-jpeg"
    async with _build_server(bus, "127.0.0.1", 0, snapshot=lambda: fake_jpeg) as server:
        port = server.sockets[0].getsockname()[1]
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, _get, f"http://127.0.0.1:{port}/camera.jpg?t=123"
        )
        with resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/jpeg"
            assert resp.headers["Cache-Control"] == "no-store"
            assert resp.read() == fake_jpeg


async def test_camera_route_404_when_no_snapshot_attached(running_server):
    _bus, port = running_server
    loop = asyncio.get_running_loop()
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        await loop.run_in_executor(None, _get, f"http://127.0.0.1:{port}/camera.jpg")
    assert exc_info.value.code == 404


async def _recv_type(client, wanted, timeout=1.0):
    """Read messages until one of `wanted` type arrives (skip seeding/others)."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        msg = json.loads(await asyncio.wait_for(client.recv(), timeout=timeout))
        if msg["type"] == wanted:
            return msg
    raise AssertionError(f"no {wanted} received")


async def test_set_voice_tuning_applies_and_broadcasts():
    """This wiring is what makes the dashboard's tuning panel work in BOTH
    single-agent and two-agent mode — app.py now injects it unconditionally."""
    from harp.config import VoiceTuning

    bus = Bus()
    tuning = VoiceTuning()
    async with _build_server(
        bus, "127.0.0.1", 0,
        set_voice_tuning=tuning.apply,
        get_voice_tuning=tuning.snapshot,
    ) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as client:
            # A fresh connection is seeded with the current tuning once.
            seed = await _recv_type(client, "VoiceTuningChanged")
            assert seed["fields"]["near_field_level"] == 0.0

            await client.send(json.dumps(
                {"type": "SetVoiceTuning", "field": "near_field_level", "value": 0.07}
            ))
            echo = await _recv_type(client, "VoiceTuningChanged")
            assert echo["fields"]["near_field_level"] == 0.07
            assert tuning.near_field_level == 0.07  # applied server-side too


async def test_bad_voice_tuning_raises_error_not_crash():
    from harp.config import VoiceTuning

    bus = Bus()
    tuning = VoiceTuning()
    async with _build_server(
        bus, "127.0.0.1", 0,
        set_voice_tuning=tuning.apply,
        get_voice_tuning=tuning.snapshot,
    ) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as client:
            await _recv_type(client, "VoiceTuningChanged")  # initial seed
            await client.send(json.dumps(
                {"type": "SetVoiceTuning", "field": "bogus", "value": 1}
            ))
            err = await _recv_type(client, "ErrorRaised")
            assert err["fields"]["where"] == "dashboard.voice_tuning"
