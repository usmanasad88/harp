# people/ — who HARP should recognize

One folder per person. The **folder name is their stable id** (lowercase,
hyphens), so keep it once chosen:

```
people/
  usman-asad/
    info.yaml       # who this is (see below)
    front.jpg       # 3-5 photos: different angles, lighting, with/without
    smiling.jpg     # glasses. Exactly ONE face per photo — group shots are
    side.jpg        # skipped (crop them to just this person).
```

`info.yaml`:

```yaml
name: Usman Asad          # required — how HARP addresses/announces them
role: organizer           # optional — organizer / speaker / guest / ...
notes: >                  # optional — model-facing context, free text.
  Built HARP. Greet warmly; he likes short answers.
```

Then build the recognition store (re-run whenever photos/info change):

```bash
uv run python scripts/enroll_people.py
```

And check it against the live webcam:

```bash
uv run python scripts/preview_face_id.py
```

**Privacy:** these are photos and face fingerprints of real people. This
folder (except this README) and the built store (`.harp/memory/people/`) are
gitignored — keep it that way, and get people's OK before enrolling them.
