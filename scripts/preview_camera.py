#!/usr/bin/env python3
"""Manual sanity check for harp/vision/camera.py — grabs a frame from the real
webcam and saves it to disk so you can look at it and confirm capture works.

Usage:
    uv run python scripts/preview_camera.py
    uv run python scripts/preview_camera.py --device 1 --out /tmp/frame.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harp.vision.camera import Camera  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=0, help="camera index or path (default: 0)")
    parser.add_argument("--out", default="/tmp/harp_camera_preview.jpg")
    parser.add_argument(
        "--warmup", type=float, default=2.5,
        help="seconds to wait before grabbing a frame (this webcam's cap.set() "
             "resolution/FPS renegotiation measured ~1.8-2s)",
    )
    args = parser.parse_args()

    device = int(args.device) if str(args.device).isdigit() else args.device
    cam = Camera(device=device)
    print(f"Opening camera device {device!r} ...")
    await cam.start()
    try:
        await asyncio.sleep(args.warmup)
        frame = cam.latest()
        if frame is None:
            print("No frame captured — device may not be producing frames.")
            return
        cv2.imwrite(args.out, frame)
        print(f"Saved a frame ({frame.shape[1]}x{frame.shape[0]}) to {args.out}")
        print("Open that file to confirm it looks right.")
    finally:
        await cam.stop()


if __name__ == "__main__":
    asyncio.run(main())
