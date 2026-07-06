#!/usr/bin/env python3
"""Enroll known people (organizers, speakers, guests) so HARP can recognize them.

Feed it one folder per person under people/ (see people/README.md):

    people/
      usman-asad/            # folder name = the stable person_id
        info.yaml            # name (required); role, notes (optional)
        front.jpg            # 3-5 photos, different angles/lighting;
        smiling.jpg          # exactly ONE face per photo (group shots skipped)

Usage:
    uv run python scripts/enroll_people.py
    uv run python scripts/enroll_people.py --only usman-asad

Each photo becomes one face embedding in the store (.harp/memory/people/, one
JSON file per person — gitignored, like the photos themselves). Re-running
re-enrolls people from their current photos but keeps any accumulated
interaction summaries. Downloads the buffalo_l model bundle (~350MB) to
~/.insightface on first run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harp.config import PEOPLE_DIR, PEOPLE_STORE  # noqa: E402
from harp.memory.store import MemoryStore  # noqa: E402

_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _load_info(folder: Path) -> dict:
    info_file = folder / "info.yaml"
    if not info_file.exists():
        print(f"  ! no info.yaml — using folder name as the display name")
        return {"name": folder.name.replace("-", " ").title()}
    info = yaml.safe_load(info_file.read_text()) or {}
    if not info.get("name"):
        print(f"  ! info.yaml has no `name:` — using folder name as the display name")
        info["name"] = folder.name.replace("-", " ").title()
    return info


def _embed_photos(app, folder: Path) -> list:
    """One embedding per usable photo. A photo is skipped when it has no
    detectable face or more than one (no way to know which is the person)."""
    embeddings = []
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in _PHOTO_SUFFIXES)
    for photo in photos:
        img = cv2.imread(str(photo))
        if img is None:
            print(f"  ! {photo.name}: unreadable image — skipped")
            continue
        faces = app.get(img)
        if len(faces) == 0:
            print(f"  ! {photo.name}: no face detected — skipped")
        elif len(faces) > 1:
            print(f"  ! {photo.name}: {len(faces)} faces (group shot?) — skipped, "
                  "crop it to just this person")
        else:
            embeddings.append(faces[0].normed_embedding)
            print(f"  + {photo.name}: enrolled (det {faces[0].det_score:.2f})")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-dir", type=Path, default=PEOPLE_DIR,
                        help=f"folder of person folders (default: {PEOPLE_DIR})")
    parser.add_argument("--store", type=Path, default=PEOPLE_STORE,
                        help=f"where the store lives (default: {PEOPLE_STORE})")
    parser.add_argument("--only", help="enroll just this one person (folder name)")
    args = parser.parse_args()

    if not args.people_dir.is_dir():
        print(f"No {args.people_dir} folder — create people/<person>/ with photos "
              "and an info.yaml first (see people/README.md).")
        return

    folders = sorted(
        f for f in args.people_dir.iterdir()
        if f.is_dir() and (args.only is None or f.name == args.only)
    )
    if not folders:
        print(f"Nothing to enroll in {args.people_dir}"
              + (f" matching {args.only!r}" if args.only else ""))
        return

    print("Loading InsightFace (buffalo_l) — first run downloads the model...")
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_thresh=0.5)

    store = MemoryStore(args.store)
    enrolled = 0
    for folder in folders:
        print(f"\n{folder.name}/")
        info = _load_info(folder)
        embeddings = _embed_photos(app, folder)
        if not embeddings:
            print(f"  ! no usable photos — {folder.name} NOT enrolled")
            continue
        store.upsert_person(
            {
                "person_id": folder.name,
                "name": info["name"],
                "role": info.get("role"),
                "notes": info.get("notes", ""),
                "embeddings": embeddings,
            }
        )
        enrolled += 1
        print(f"  = {info['name']} ({len(embeddings)} photo(s))")

    print(f"\nEnrolled {enrolled} of {len(folders)} people -> {args.store}")
    print("Check recognition against the live webcam with: "
          "uv run python scripts/preview_face_id.py")


if __name__ == "__main__":
    main()
