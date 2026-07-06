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
gestures + face-ID, the wave→wake trigger engine, the interaction end-rules
(face-ID presence: no face for a while closes the session), and the dashboard
(host/port from harp.yaml's `dashboard:` section — localhost-only by default,
or LAN-visible for phone/other-PC access) all sharing one bus. A wake (wake
word, loud sound, or wave) opens a real conversation, and the person walking
off ends it.

Status narration (orchestrator/status_voice) is wired here too: canned boot /
connectivity / standby / error / shutdown lines, played from local clips.

Not wired yet (joins here as each is built): web-search fallback, memory's
logger/summarizer, watchdog.
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
    STATUS_VOICE_DIR,
    FilterTuning,
    build_filter_config,
    build_session_config,
    dashboard_bind_host,
    load_settings,
)
from .core.bus import Bus
from .dashboard.server import serve as serve_dashboard
from .interaction import session_tools
from .interaction.end_rules import EndOfInteractionMonitor
from .interaction.push_to_talk import PushToTalk
from .knowledge import tools as knowledge_tools
from .listener.listener import AlwaysOnListener
from .memory.store import MemoryStore
from .orchestrator.orchestrator import Orchestrator
from .orchestrator.status_voice import StatusVoice
from .triggers.engine import TriggerEngine
from .vision.camera import Camera
from .vision.face_id import FaceID
from .vision.frames import jpeg_snapshot
from .vision.gestures import GestureRecognizer
from .voice.bridge import VoiceBridge
from .voice.two_agent import TwoAgentBridge

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


def _internet_reachable(timeout: float = 3.0) -> bool:
    """True if we can open a TCP connection to a public host — a real
    reachability check (DNS + routing + a completed handshake), not just "is a
    cable plugged in". The orchestrator runs this at boot to announce
    "connection established" vs "no internet"; it runs off-thread so the ~timeout
    on a dead network never stalls the event loop."""
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=timeout):
            return True
    except OSError:
        return False


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
        # Tools the model can call: knowledge retrieval + hanging up on itself.
        config.tools = knowledge_tools.declarations(provider_name) + session_tools.declarations(
            provider_name
        )
        return config

    async def dispatch(name: str, arguments: dict):
        # end_session has a side effect on the orchestrator (it publishes an end
        # event), so it needs the bus; everything else is knowledge retrieval.
        if name == session_tools.TOOL_NAME:
            return await session_tools.end_session(bus, arguments)
        return await knowledge_tools.dispatch(name, arguments)

    # Push-to-talk (harp.yaml push_to_talk.enabled): arms a talk key. It runs
    # ALONGSIDE the always-on listener — pressing the key while idle starts a
    # session whose mic is gated (real audio only while held), so a loud room
    # can't interfere; hands-free wakes (wave / wake word) are ungated as before.
    # `mic_open` is the gate the voice bridge consults.
    push_to_talk = (
        PushToTalk(bus, key=settings.push_to_talk.key)
        if settings.push_to_talk.enabled
        else None
    )

    # The push-to-talk gate applies to whichever agent owns the mic: the plain
    # bridge in single-agent mode, or the filter agent in two-agent mode.
    ptt_gate = (lambda: push_to_talk.mic_open) if push_to_talk is not None else None

    # Live-tunable filter knobs (loudness gate + filter-session VAD), seeded from
    # harp.yaml and adjustable on the dashboard while tuning against the real
    # room. Only meaningful in two-agent mode; wired to the dashboard only then.
    filter_tuning = FilterTuning(
        near_field_level=settings.filter_agent.near_field_level,
        vad_threshold=settings.filter_agent.vad_threshold,
        vad_silence_ms=settings.filter_agent.vad_silence_ms,
        noise_reduction=settings.filter_agent.noise_reduction,
    )

    voice_bridge: VoiceBridge | TwoAgentBridge
    if settings.filter_agent.enabled:
        # Two-agent mode (harp.yaml filter_agent.enabled): a filter agent hears the
        # room and relays only the intended message (as text) to the normal
        # responder. Same run(context) interface, so the orchestrator is unchanged.
        filter_provider = settings.filter_agent.provider or provider_name
        voice_bridge = TwoAgentBridge(
            bus,
            provider_name,
            make_config=make_session_config,
            # Rebuilt at each session open, so the current dashboard VAD/noise
            # knobs are stamped onto the next conversation.
            make_filter_config=lambda: build_filter_config(filter_provider, filter_tuning),
            tool_dispatch=dispatch,
            identity_context=identity_context,
            filter_provider_name=filter_provider,
            response_tail_seconds=settings.filter_agent.response_tail_seconds,
            external_mic_gate=ptt_gate,
            # Read live per mic chunk, so moving the dashboard slider is instant.
            near_field_level=lambda: filter_tuning.near_field_level,
        )
        print(
            f"Two-agent filter mode ON: filter={filter_provider}, "
            f"responder={provider_name} (experimental; expect ~1-2s extra latency)."
        )
    else:
        voice_bridge = VoiceBridge(
            bus,
            provider_name,
            make_config=make_session_config,
            tool_dispatch=dispatch,
            identity_context=identity_context,
            mic_gate=ptt_gate,
        )

    # Canned status narration (boot / connectivity / standby / error / shutdown),
    # played from local clips when the cloud voice can't be. Disabled → the
    # orchestrator runs silently; the boot connectivity probe is only wired when
    # narration is on (its only consumer is the "connection established"/"no
    # internet" line).
    status_voice = (
        StatusVoice(STATUS_VOICE_DIR, lang=settings.status_voice.lang)
        if settings.status_voice.enabled
        else None
    )
    orchestrator = Orchestrator(
        bus,
        provider_name,
        heartbeat_interval=settings.heartbeat.interval_seconds,
        heartbeat_file=REPO_ROOT / settings.heartbeat.file,
        voice_bridge=voice_bridge,
        status_voice=status_voice,
        connectivity_check=_internet_reachable if status_voice is not None else None,
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
            jpeg_snapshot, camera, overlays=(gestures.overlay, face_id.overlays)
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
            # The filter-tuning sliders are only wired in two-agent mode; in
            # single-agent mode the dashboard shows no tuning panel.
            set_filter_tuning=filter_tuning.apply if settings.filter_agent.enabled else None,
            get_filter_tuning=filter_tuning.snapshot if settings.filter_agent.enabled else None,
        ),
        "orchestrator": orchestrator.run(),
        # Closes a live session when the person walks off: face-ID's presence
        # goes absent and stays absent past the timeout (harp.yaml interaction:).
        # `is_present` seeds presence at each open (the bus won't replay it), so
        # a session that wakes with nobody in frame still closes on its own.
        "end_rules": EndOfInteractionMonitor(
            bus,
            absence_timeout=settings.interaction.absence_timeout_seconds,
            is_present=(lambda: face_id.current is not None) if face_id is not None else None,
        ).run(),
    }
    if settings.listener.enabled:
        runners["listener"] = AlwaysOnListener(bus, settings.listener).run()
    if push_to_talk is not None:
        # Runs alongside the listener: a press starts a gated session on demand,
        # and the session ending returns HARP to the hands-free listener/wave.
        print(
            f"Push-to-talk armed: hold {settings.push_to_talk.key.upper()} to "
            "start a hold-to-talk session."
        )
        runners["push_to_talk"] = push_to_talk.run()
    if gestures is not None:
        runners["gestures"] = gestures.run()
        # A wave is a wake condition; the engine turns GestureDetected into a
        # WakeRequested the orchestrator honors while STANDBY.
        runners["triggers"] = TriggerEngine(bus).run()
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
