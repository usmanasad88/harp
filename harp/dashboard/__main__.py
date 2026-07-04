"""Standalone dashboard runner: `python -m harp.dashboard`.

Binds the dashboard to a fresh, empty `Bus`. With no subsystem wired to that
bus yet, every panel shows its honest "nothing yet" state — this is the
intended way to check the dashboard itself works before anything real feeds
it. As subsystems are wired to a shared bus (see app.py), the same dashboard
will start showing their output with no changes here.
"""

from __future__ import annotations

import asyncio

from ..core.bus import Bus
from .server import serve


def main() -> None:
    bus = Bus()
    try:
        asyncio.run(serve(bus))
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
