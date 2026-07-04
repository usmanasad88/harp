#!/usr/bin/env python3
"""Manual sanity check for face-ID against the real webcam: detection
(InsightFace buffalo_l) plus recognition against the enrolled-people store if
one exists (built by scripts/enroll_people.py). Grabs a frame, draws a box per
face labeled with the matched name + similarity (or "unknown"), and saves the
result so you can look at it.

With no store (nobody enrolled yet) it still runs — detection boxes only,
which is what this script did before the matcher existed.

Downloads the buffalo_l model bundle (~350MB) to ~/.insightface on first run.

Usage:
    uv run python scripts/preview_face_id.py
    uv run python scripts/preview_face_id.py --device 1 --out /tmp/faces.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harp.config import PEOPLE_STORE  # noqa: E402
from harp.memory import matcher  # noqa: E402
from harp.memory.store import MemoryStore  # noqa: E402
from harp.vision.camera import Camera  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=0, help="camera index or path (default: 0)")
    parser.add_argument("--out", default="/tmp/harp_face_id_preview.jpg")
    parser.add_argument("--warmup", type=float, default=2.5,
                        help="seconds to wait before grabbing a frame (this webcam's "
                             "cap.set() resolution/FPS renegotiation measured ~1.8-2s)")
    parser.add_argument("--store", type=Path, default=PEOPLE_STORE,
                        help=f"enrolled-people store (default: {PEOPLE_STORE})")
    args = parser.parse_args()

    store = MemoryStore(args.store) if args.store.is_dir() else None
    if store is not None and store.people():
        names = ", ".join(p.name or p.person_id for p in store.people())
        print(f"Matching against {len(store.people())} enrolled: {names}")
    else:
        store = None
        print("No one enrolled yet (run scripts/enroll_people.py) — detection only.")

    print("Loading InsightFace (buffalo_l) — first run downloads the model...")
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_thresh=0.5)

    device = int(args.device) if str(args.device).isdigit() else args.device
    cam = Camera(device=device)
    print(f"Opening camera device {device!r} ...")
    await cam.start()
    try:
        await asyncio.sleep(args.warmup)
        frame = cam.latest()
        if frame is None:
            print("No frame captured.")
            return

        faces = app.get(frame)
        print(f"Detected {len(faces)} face(s).")
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            label = f"{face.det_score:.2f}"
            if store is not None:
                person_id, is_known, sim = matcher.match(face.normed_embedding, store)
                if is_known:
                    record = store.get(person_id)
                    label = f"{record.name or person_id} {sim:.2f}"
                else:
                    label = f"unknown {sim:.2f}"
            print(f"  face {i}: bbox=({x1},{y1},{x2},{y2}) "
                  f"det_score={face.det_score:.2f} -> {label}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        cv2.imwrite(args.out, frame)
        print(f"Saved annotated frame to {args.out}")
        print("Open that file to confirm the boxes and names are right.")
    finally:
        await cam.stop()


if __name__ == "__main__":
    asyncio.run(main())
