"""CLI entry point: `python -m harp` runs the full HARP agent.

By default this launches the complete supervised agent — the orchestrator, the
real cloud voice session (VoiceBridge with `search_knowledge` over `data/`), the
always-on wake listener (threshold-based session start, tuned in `harp.yaml`),
the camera + gesture + face-ID vision stack, and the developer dashboard — all
sharing one event bus. Behavior knobs live in `harp.yaml`; secrets in `.env`.

    python -m harp                     # full agent + dashboard (default)
    python -m harp --provider openai   # pick the realtime backend
    python -m harp --voice-only        # bare mic+speaker session only — no
                                       # orchestrator/bus/dashboard/vision
                                       # (a quick way to smoke-test the voice core)

The full wiring lives in `harp.app.run_app` (the composition root that knows
every subsystem); this module is just the CLI in front of it, plus the
`--voice-only` shortcut to the bare `harp.voice.session.run` runner.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from . import config
from .knowledge import tools as knowledge_tools
from .voice.session import run as run_voice_session


def _run_voice_only(provider: str) -> None:
    """The bare voice core: mic + speaker + `search_knowledge`, nothing else.

    No bus, orchestrator, dashboard, wake listener, or vision — just talk into
    the mic and HARP answers, grounded in `data/`. Handy for a fast check of the
    provider/audio path without spinning up the whole agent.
    """
    cfg = config.build_session_config(provider)
    # Composition root for this path: advertise search_knowledge and wire its
    # dispatcher, keeping session.py provider- and corpus-agnostic.
    cfg.tools = knowledge_tools.declarations(provider)

    print(f"HARP (voice-only) — provider={provider}  model={cfg.model}  voice={cfg.voice}")
    print(f"Knowledge — {knowledge_tools.index_size()} chunks indexed from data/ (search_knowledge on)")
    print("Listening… speak (English or Urdu); Ctrl+C to quit.")

    try:
        asyncio.run(run_voice_session(provider, cfg, tool_dispatch=knowledge_tools.dispatch))
    except KeyboardInterrupt:
        print("\nBye.")
    except RuntimeError as exc:  # clean, expected errors (e.g. missing API key)
        sys.exit(str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="harp",
        description="HARP — bilingual (EN/Urdu) voice agent: orchestrator + wake "
        "listener + vision + dashboard.",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("HARP_PROVIDER", "openai"),
        choices=["gemini", "openai"],
        help="which real-time backend to use (default: openai)",
    )
    parser.add_argument(
        "--voice-only",
        action="store_true",
        help="run only the bare mic+speaker voice session — no orchestrator, "
        "wake listener, vision, or dashboard (quick smoke test)",
    )
    args = parser.parse_args()

    if args.voice_only:
        _run_voice_only(args.provider)
        return

    # Imported here, not at module top, so `--voice-only` stays a lightweight
    # fallback that doesn't pull in (or depend on) the heavy vision stack
    # (cv2 / insightface / mediapipe) that app.py wires together.
    from .app import run_app

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Every plain HTTP GET (the dashboard page, app.js, /camera.jpg polled ~4x/s)
    # logs as "connection rejected (200 OK)" at INFO — it's the websockets
    # library's log line for any non-upgrade request on the same port, not an
    # actual problem. Quiet it to warnings/errors only.
    logging.getLogger("websockets.server").setLevel(logging.WARNING)

    print(f"HARP — full agent  provider={args.provider}")
    print(f"Knowledge — {knowledge_tools.index_size()} chunks indexed from data/ (search_knowledge on)")
    print("Wake it with a wake word, a loud sound, or a wave; Ctrl+C to quit.")

    try:
        asyncio.run(run_app(args.provider))
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
