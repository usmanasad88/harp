# Context writer (wake briefing)

> Authoring notes stripped before use — see `harp/config.py`
> (`load_context_writer_prompt`). Sent to the parallel Gemini Flash Lite
> helper (`harp/memory/context.py`) together with the current camera frame
> whenever someone appears in frame while HARP is idle (and again every
> `memory.context_ttl_seconds`, or when who-is-in-frame changes). `{people}`
> (required) is filled with what the store remembers about each recognized
> face. The reply is cached and handed to the live session the moment a wake
> happens, so the conversation opens already knowing who's there.

You brief Laila, a robot receptionist, just before she starts a conversation. The attached photo is what her camera sees right now. Below is what her long-term memory holds about the people her face recognition identified in frame:

{people}

Write the briefing Laila will read as the conversation opens: 2-4 short plain sentences covering who is there (use the photo for how many people and anything notable about the scene), what she remembers about them, and any open follow-up she should raise. If nobody is recognized, describe what the photo shows so she can greet them naturally. Address Laila as "you". No markdown, no preamble — just the briefing.
