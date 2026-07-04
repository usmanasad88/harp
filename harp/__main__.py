"""CLI entry point: `python -m harp [--provider gemini|openai]`.

Talk into the mic; HARP answers through the speakers. Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from . import config
from .voice.session import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="harp", description="HARP voice core")
    parser.add_argument(
        "--provider",
        default=os.getenv("HARP_PROVIDER", "gemini"),
        choices=["gemini", "openai"],
        help="which real-time backend to use (default: gemini)",
    )
    args = parser.parse_args()

    cfg = config.build_session_config(args.provider)
    print(f"HARP — provider={args.provider}  model={cfg.model}  voice={cfg.voice}")
    print("Listening… speak (English or Urdu); Ctrl+C to quit.")

    try:
        asyncio.run(run(args.provider, cfg))
    except KeyboardInterrupt:
        print("\nBye.")
    except RuntimeError as exc:  # clean, expected errors (e.g. missing API key)
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
