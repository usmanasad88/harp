# Describe scene (vision instruction)

> Authoring notes stripped before use — see `harp/config.py`
> (`load_describe_scene_prompt`). Sent to the parallel Gemini Flash Lite
> helper together with the current camera frame when the LIVE model calls its
> `describe_scene` tool mid-conversation (`harp/vision/describe.py`).
> `{focus}` (required) is whatever the live model asked to look for, or
> "anything notable".

The attached photo is what a robot receptionist's camera sees right now, mid-conversation. Describe it for her in 2-4 short spoken-style sentences she can relay to the visitor: who and what is visible, and anything notable. Pay particular attention to: {focus}. No markdown, no preamble.
