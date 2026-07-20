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
connectivity / standby / error / shutdown lines, played from local clips. So is
the per-run session log (core/session_log): one JSONL timeline per run — a
settings/model header, every bus event, every module's log lines — written to
.harp/logs/ for post-hoc debugging (harp.yaml `session_log:`).

Long-term memory (harp/memory, harp.yaml `memory:`) is wired here too: the
per-interaction transcript logger, the summarizer that turns each finished
conversation into per-person memory (guestbook for unknown visitors), the
context writer that pre-computes a wake briefing (camera frame + memories)
whenever face-ID sees someone while idle, and the live model's search_memory /
describe_scene tools — all sharing one Gemini Flash Lite helper agent behind
one rate limiter.

An optional autonomous-patrol check (harp/motion/autonomous_patrol.py +
harp/motion/patrol_state.py) is wired here too: while patrol drives the wheels
in a separate process, voice wakes are silently ignored — checked via a tiny
local HTTP flag, imports nothing from harp.motion.

Not wired yet (joins here as each is built): web-search fallback, watchdog.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import json
import logging
import os
import platform
import socket
import sys
import urllib.request

from . import audio_control
from .config import (
    FALLBACK_IDENTITY_CONTEXT,
    FALLBACK_IDENTITY_CONTEXT_WITH_NOTES,
    GUESTBOOK_FILE,
    INTERACTIONS_DIR,
    PEOPLE_STORE,
    REPO_ROOT,
    STATUS_VOICE_DIR,
    VoiceTuning,
    build_filter_config,
    build_session_config,
    dashboard_bind_host,
    format_prompt,
    load_identity_context,
    load_identity_context_with_notes,
    load_settings,
)
from .core.bus import Bus
from .core.session_log import SessionLog
from .dashboard.server import serve as serve_dashboard
from .interaction import session_tools
from .interaction.end_rules import EndOfInteractionMonitor
from .interaction.push_to_talk import PushToTalk
from .knowledge import tools as knowledge_tools
from .listener.listener import AlwaysOnListener
from .memory import tools as memory_tools
from .memory.agent import GeminiAgent, RateLimiter
from .memory.context import ContextWriter
from .memory.logger import InteractionLogger
from .memory.store import MemoryStore
from .memory.summarizer import MemorySummarizer
from .orchestrator.orchestrator import Orchestrator
from .orchestrator.status_voice import StatusVoice
from .triggers.engine import TriggerEngine
from .vision import describe as describe_tool
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


def _patrol_active(timeout: float = 0.5) -> bool:
    """True while autonomous_patrol.py (harp/motion) is driving the wheels.
    Polls the tiny local flag server it starts (harp/motion/patrol_state.py);
    any failure (not running, unreachable) counts as "not active" so voice
    behaves normally whenever patrol isn't in the picture."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8790/patrol", timeout=timeout
        ) as resp:
            data = json.loads(resp.read().decode())
            return bool(data.get("active", False))
    except Exception:
        return False


def _run_header(provider_name: str, settings) -> dict:
    """The session log's first record: what a later debugging session needs to
    know about this run — the knobs as they actually were (merged harp.yaml +
    defaults), the resolved model/voice, and the machine. No secrets: API keys
    live in .env and are never part of Settings or the session config fields
    recorded here."""
    session = build_session_config(provider_name)
    return {
        "provider": provider_name,
        "model": session.model,
        "voice": session.voice,
        "settings": dataclasses.asdict(settings),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


async def run_app(provider_name: str = "gemini") -> None:
    """Wire the bus + all enabled subsystems + orchestrator; run until shutdown."""
    settings = load_settings()
    bus = Bus()

    # The per-run developer log (harp.yaml session_log:) — opened before
    # anything else so even the earliest warnings (a missing camera, below)
    # land in the timeline. Its handler mirrors every module's log lines into
    # the same file the bus events go to.
    session_log: SessionLog | None = None
    log_handler: logging.Handler | None = None
    if settings.session_log.enabled:
        session_log = SessionLog(
            bus,
            REPO_ROOT / settings.session_log.dir,
            keep_runs=settings.session_log.keep_runs,
        )
        log_path = session_log.open(_run_header(provider_name, settings))
        log_handler = session_log.handler()
        logging.getLogger().addHandler(log_handler)
        logger.info("session log: %s", log_path)

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

    # A clean (overlay-free) view of the shared camera for the memory helper —
    # the wake briefing and describe_scene should see the room, not our boxes.
    clean_snapshot = (
        functools.partial(jpeg_snapshot, camera) if camera is not None else lambda: None
    )

    # Long-term memory (harp/memory, harp.yaml `memory:`): the transcript
    # logger always runs when enabled (raw records need no API); the parallel
    # Flash Lite helper — summarizer, wake-briefing writer, describe_scene —
    # additionally needs GEMINI_API_KEY. Without it, transcripts pile up
    # pending and are summarized by a later run that has the key.
    memory_agent: GeminiAgent | None = None
    interaction_logger: InteractionLogger | None = None
    summarizer: MemorySummarizer | None = None
    context_writer: ContextWriter | None = None
    if settings.memory.enabled:
        interaction_logger = InteractionLogger(
            bus,
            INTERACTIONS_DIR,
            people_now=face_id.people_now if face_id is not None else None,
        )
        if os.getenv("GEMINI_API_KEY"):
            memory_agent = GeminiAgent(
                settings.memory.model, RateLimiter(settings.memory.calls_per_minute)
            )
            summarizer = MemorySummarizer(
                bus, store, memory_agent, INTERACTIONS_DIR, GUESTBOOK_FILE
            )
            if face_id is not None:
                context_writer = ContextWriter(
                    bus,
                    memory_agent,
                    store,
                    people_now=face_id.people_now,
                    frame_jpeg=clean_snapshot,
                    ttl_seconds=settings.memory.context_ttl_seconds,
                )
        else:
            logger.warning(
                "memory: GEMINI_API_KEY not set — transcripts are recorded but "
                "summaries/briefings/describe_scene are off this run"
            )

    def identity_context() -> str:
        """Model-facing "who you're talking to" context, delivered into the
        voice session at open. Preferred source: the memory helper's
        pre-computed briefing (camera frame + stored memories, see
        memory/context.py); fallback: the static face-ID identity line.
        Wording lives in prompts/ (see prompts/README.md)."""
        if context_writer is not None:
            briefing = context_writer.context()
            if briefing:
                return briefing
        current = face_id.current if face_id is not None else None
        if current is None or not current.is_known:
            return ""
        try:
            person = store.get(current.person_id)
        except KeyError:  # store edited/cleared since the sighting
            return ""
        name = person.name or current.person_id
        if person.notes:
            template = load_identity_context_with_notes()
            return format_prompt(
                template, FALLBACK_IDENTITY_CONTEXT_WITH_NOTES, name=name, notes=person.notes
            )
        template = load_identity_context()
        return format_prompt(template, FALLBACK_IDENTITY_CONTEXT, name=name)

    def make_session_config():
        # Noise/VAD tuning (voice_tuning, below) is stamped onto this session
        # too — the single-agent bridge owns a real mic just like the filter
        # does, and faces the same room-noise problem.
        config = build_session_config(provider_name, tuning=voice_tuning)
        # Tools the model can call: knowledge retrieval, hanging up on itself,
        # and — when the memory subsystem is on — searching its own past
        # (needs no API key) and looking through the camera (needs the helper).
        config.tools = knowledge_tools.declarations(provider_name) + session_tools.declarations(
            provider_name
        )
        if settings.memory.enabled:
            config.tools += memory_tools.declarations(provider_name)
        if memory_agent is not None and camera is not None:
            config.tools += describe_tool.declarations(provider_name)
        return config

    async def dispatch(name: str, arguments: dict):
        # end_session has a side effect on the orchestrator (it publishes an end
        # event), so it needs the bus; search_memory reads the store/guestbook;
        # describe_scene asks the memory helper to look through the camera;
        # everything else is knowledge retrieval.
        if name == session_tools.TOOL_NAME:
            return await session_tools.end_session(bus, arguments)
        if name == memory_tools.TOOL_NAME:
            return await memory_tools.search_memory(store, GUESTBOOK_FILE, arguments)
        if name == describe_tool.TOOL_NAME:
            if memory_agent is None:
                return {"error": "the vision helper is not available this run"}
            return await describe_tool.describe_scene(memory_agent, clean_snapshot, arguments)
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

    # Live-tunable noise/VAD knobs (loudness gate + server-VAD + noise
    # reduction), seeded from harp.yaml and adjustable on the dashboard while
    # tuning against the real room. Applies to whichever agent currently owns
    # the mic — the plain bridge below, or (in two-agent mode) the filter.
    voice_tuning = VoiceTuning(
        near_field_level=settings.voice_tuning.near_field_level,
        vad_threshold=settings.voice_tuning.vad_threshold,
        vad_silence_ms=settings.voice_tuning.vad_silence_ms,
        noise_reduction=settings.voice_tuning.noise_reduction,
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
            make_filter_config=lambda: build_filter_config(filter_provider, voice_tuning),
            tool_dispatch=dispatch,
            identity_context=identity_context,
            filter_provider_name=filter_provider,
            response_tail_seconds=settings.filter_agent.response_tail_seconds,
            external_mic_gate=ptt_gate,
            # Read live per mic chunk, so moving the dashboard slider is instant.
            near_field_level=lambda: voice_tuning.near_field_level,
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
            # Same loudness gate the two-agent filter uses, applied to this
            # session's own mic since it's the one that owns it here.
            near_field_level=lambda: voice_tuning.near_field_level,
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
        patrol_active_check=_patrol_active,
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

    runners = {}
    if session_log is not None:
        # First in the dict so its catch-all bus subscription is registered
        # before the orchestrator publishes its first boot events.
        runners["session_log"] = session_log.run()
    runners.update({
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
            # The voice-tuning sliders are wired every run — they apply to
            # whichever agent currently owns the mic (plain bridge or filter).
            set_voice_tuning=voice_tuning.apply,
            get_voice_tuning=voice_tuning.snapshot,
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
    })
    if interaction_logger is not None:
        # Records each conversation; the summarizer (below, only with a key)
        # turns finished transcripts into per-person memory.
        runners["interaction_log"] = interaction_logger.run()
    if summarizer is not None:
        runners["memory_summarizer"] = summarizer.run()
    if context_writer is not None:
        # Pre-computes the wake briefing whenever face-ID sees someone while
        # idle, so identity_context() has it ready the instant a wake happens.
        runners["context_writer"] = context_writer.run()
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
        if session_log is not None:
            # Last, so teardown warnings from the lines above are still
            # captured; anything logged after close is silently dropped.
            logging.getLogger().removeHandler(log_handler)
            session_log.close()


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