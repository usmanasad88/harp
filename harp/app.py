"""Composition root: build the bus, wire every subsystem, run HARP.

This is the ONE place that knows all subsystems exist. It constructs the shared
Bus, instantiates each subsystem with that bus alongside the Orchestrator, and
runs them concurrently. Because everything talks only through the bus, any
subsystem can be commented out here and the rest still runs — that is the
modularity guarantee, and the recommended way to bring subsystems online one at
a time.

Contrast with `python -m harp` (__main__.py), which runs ONLY the bare voice
session for quick manual testing. `python -m harp.app` is the full supervised
agent — currently the subsystems that exist: orchestrator + the real voice
session (VoiceBridge with the search_knowledge tool), wake listener, camera +
gestures + face-ID, and the dashboard (host/port from harp.yaml's `dashboard:`
section — localhost-only by default, or LAN-visible for phone/other-PC access)
all sharing one bus. A wake now opens a real conversation; what still can't
happen automatically is *ending* one (interaction/end_rules is unbuilt), so a
session stays open until the provider closes it, an error closes it, or
Ctrl+C.

Not wired yet (joins here as each is built): presence, web-search fallback,
memory's logger/summarizer, triggers, watchdog, status_voice.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import socket

from . import audio_control
from .config import (
    PEOPLE_STORE,
    REPO_ROOT,
    build_session_config,
    dashboard_bind_host,
    load_settings,
)
from .core.bus import Bus
from .dashboard.server import serve as serve_dashboard
from .knowledge import tools as knowledge_tools
from .listener.listener import AlwaysOnListener
from .memory.store import MemoryStore
from .orchestrator.orchestrator import Orchestrator
from .vision.camera import Camera
from .vision.face_id import FaceID
from .vision.frames import jpeg_snapshot
from .vision.gestures import GestureRecognizer
from .voice.bridge import VoiceBridge

logger = logging.getLogger(__name__)


def _lan_ip() -> str | None:
    """Best-effort LAN IP for the interface that would reach the internet (a
    UDP "connect" only does a routing-table lookup, no packet is sent) —
    picks the real Wi-Fi/Ethernet address over Docker/VPN bridge interfaces."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


async def run_app(provider_name: str = "gemini") -> None:
    """Wire the bus + all enabled subsystems + orchestrator; run until shutdown."""
    settings = load_settings()
    bus = Bus()

    # A missing/busy webcam shouldn't keep the voice side from running; the
    # camera-fed subsystems just stay offline for this run.
    camera: Camera | None = Camera()
    try:
        await camera.start()
    except RuntimeError as exc:
        logger.warning(
            "camera unavailable (%s) — gestures + face-ID disabled this run", exc
        )
        camera = None

    # Built once so the same instances both run (publish GestureDetected /
    # PersonIdentified) and contribute what they see to the dashboard's
    # camera-view overlay.
    gestures = GestureRecognizer(bus, camera) if camera is not None else None
    store = MemoryStore(PEOPLE_STORE)
    face_id = FaceID(bus, camera, store) if camera is not None else None

    def identity_context() -> str:
        """Model-facing "you are talking to <name>" line, delivered into the
        voice session at open when face-ID currently sees someone enrolled."""
        current = face_id.current if face_id is not None else None
        if current is None or not current.is_known:
            return ""
        try:
            person = store.get(current.person_id)
        except KeyError:  # store edited/cleared since the sighting
            return ""
        line = f"(Face recognition: you are talking to {person.name or current.person_id}"
        if person.notes:
            line += f". Notes about them: {person.notes}"
        return line + ".)"

    def make_session_config():
        config = build_session_config(provider_name)
        config.tools = knowledge_tools.declarations(provider_name)
        return config

    voice_bridge = VoiceBridge(
        bus,
        provider_name,
        make_config=make_session_config,
        tool_dispatch=knowledge_tools.dispatch,
        identity_context=identity_context,
    )

    orchestrator = Orchestrator(
        bus,
        provider_name,
        heartbeat_interval=settings.heartbeat.interval_seconds,
        heartbeat_file=REPO_ROOT / settings.heartbeat.file,
        voice_bridge=voice_bridge,
    )

    dashboard_host = dashboard_bind_host(settings.dashboard.bind)
    if dashboard_host == "0.0.0.0":
        lan_ip = _lan_ip()
        print(f"HARP dashboard: http://127.0.0.1:{settings.dashboard.port} (this PC)")
        if lan_ip:
            print(
                f"                http://{lan_ip}:{settings.dashboard.port} "
                "(phone/other PCs on the same network)"
            )
        else:
            print("                (could not detect a LAN IP for other devices)")
    else:
        print(f"HARP dashboard: http://127.0.0.1:{settings.dashboard.port} (localhost only)")

    snapshot = None
    if camera is not None:
        assert gestures is not None and face_id is not None
        snapshot = functools.partial(
            jpeg_snapshot, camera, overlays=(gestures.overlay, face_id.overlay)
        )

    runners = {
        # The dashboard watches the bus, and (camera permitting) serves the
        # shared camera's latest frame read-only at /camera.jpg, with what the
        # gesture recognizer currently sees drawn over it.
        "dashboard": serve_dashboard(
            bus,
            host=dashboard_host,
            port=settings.dashboard.port,
            snapshot=snapshot,
            set_mic_muted=audio_control.set_mic_muted,
            get_mic_muted=audio_control.get_mic_muted,
        ),
        "orchestrator": orchestrator.run(),
    }
    if settings.listener.enabled:
        runners["listener"] = AlwaysOnListener(bus, settings.listener).run()
    if gestures is not None:
        runners["gestures"] = gestures.run()
    if face_id is not None:
        runners["face_id"] = face_id.run()

    tasks = [asyncio.create_task(coro, name=name) for name, coro in runners.items()]
    try:
        # First exit wins: the orchestrator reaching STOPPING ends the app
        # normally; any other task finishing first means a subsystem crashed,
        # and task.result() re-raises that crash here.
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if camera is not None:
            await camera.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m harp.app",
        description="HARP — the full supervised agent (all built subsystems + dashboard)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("HARP_PROVIDER", "gemini"),
        choices=["gemini", "openai"],
        help="which real-time backend to use (default: gemini)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Every plain HTTP GET (the dashboard page, app.js, /camera.jpg polled ~4x/s)
    # logs as "connection rejected (200 OK)" at INFO — it's the websockets
    # library's log line for any non-upgrade request on the same port, not an
    # actual problem. Quiet it to warnings/errors only.
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    try:
        asyncio.run(run_app(args.provider))
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
