#!/usr/bin/env python3
"""Manual sanity check for harp/vision/gestures.py against the real webcam +
real MediaPipe GestureRecognizer (no bus needed here — just prints what would
have been published).

Usage:
    uv run python scripts/preview_gestures.py
    # then hold up an open palm during the run window

Downloads gesture_recognizer.task (~8MB) to ~/.cache/mediapipe on first run.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe import Image, ImageFormat  # noqa: E402

from harp.core.bus import Bus  # noqa: E402
from harp.core.events import GestureDetected  # noqa: E402
from harp.vision.camera import Camera  # noqa: E402
from harp.vision.gestures import GestureRecognizer  # noqa: E402


def _top_candidates(rec: GestureRecognizer, frame, n: int = 3) -> str:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = rec._recognizer.recognize(mp_image)
    if not result.gestures or not result.gestures[0]:
        return "(no hand)"
    cats = result.gestures[0][:n]
    return ", ".join(f"{c.category_name}={c.score:.2f}" for c in cats)


async def main() -> None:
    duration = 15.0
    bus = Bus()
    stream = bus.subscribe(GestureDetected)
    cam = Camera(device=0)
    print("Opening camera...")
    await cam.start()
    await asyncio.sleep(1.5)

    print("Loading MediaPipe GestureRecognizer...")
    rec = GestureRecognizer(bus, cam)

    for n in (3, 2, 1):
        print(f"Starting in {n}...")
        await asyncio.sleep(1)
    print(f"\nGO — hold up an open palm for the next {duration:.0f}s...\n")
    start = time.monotonic()
    detections = 0

    async def watch():
        nonlocal detections
        async for _ in stream:
            detections += 1
            print(f"  >>> GESTURE DETECTED (#{detections}) <<<")

    watcher = asyncio.create_task(watch())
    try:
        while time.monotonic() - start < duration:
            frame = cam.latest()
            if frame is not None:
                candidates = _top_candidates(rec, frame)
                await rec.process_frame(frame, time.monotonic())
                elapsed = time.monotonic() - start
                print(f"t={elapsed:4.1f}s  top={candidates}")
            await asyncio.sleep(0.1)
    finally:
        watcher.cancel()
        await cam.stop()

    print(f"\nTotal gesture cues detected: {detections}")


if __name__ == "__main__":
    asyncio.run(main())
