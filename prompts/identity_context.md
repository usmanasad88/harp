# Face-ID identity context

> Authoring notes stripped before use — see `harp/config.py`
> (`load_identity_context`, `load_identity_context_with_notes`). Delivered
> into the live session at open, right after the wake-reason context, when
> face-ID currently recognizes the person standing there (see
> `harp/app.py::identity_context`, `harp/vision/face_id.py`). NOT sent for
> unknown/unenrolled faces. `{name}` is the enrolled person's name;
> `{notes}` is the free-text notes field from their `people/<id>/info.yaml`
> (only used when notes are present — see the second file below).

(Face recognition: you are talking to {name}.)
