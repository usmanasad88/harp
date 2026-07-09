# HARP prompts

Every piece of text HARP's cloud agents are given — personas and tool
descriptions — lives here as plain markdown, not buried in Python. Edit a
file and restart HARP (`uv run python -m harp`) to pick up the change; no
code changes needed. Lines starting with `>` and the top `#` title are
authoring notes for humans — they're stripped before the model ever sees the
file (see `load_prompt` in [`harp/config.py`](../harp/config.py)).

If a file is deleted or emptied, HARP falls back to a built-in default (also
in `harp/config.py`) rather than failing to start.

| File | Used by | What it does |
|---|---|---|
| [`system_instructions.md`](system_instructions.md) | the main responder (`harp/voice/bridge.py`, both providers) | **Laila's persona.** Who she is, how she speaks, language mirroring (English/Urdu), honesty/grounding rules, and boundaries. This is the system instruction for every live conversation. |
| [`filter_instructions.md`](filter_instructions.md) | the optional two-agent noise filter (`harp/voice/filter_agent.py`, off by default — `filter_agent.enabled` in harp.yaml) | The persona for the *first* agent in noisy-room mode: hears the room, decides what (if anything) was said to HARP, and relays just that — never answers or chats. |
| [`search_knowledge_tool.md`](search_knowledge_tool.md) | the `search_knowledge` tool (`harp/knowledge/tools.py`) | The tool's `description` field — what teaches the model *when* to search `data/` before answering and to admit uncertainty on a miss. |
| [`end_session_tool.md`](end_session_tool.md) | the `end_session` tool (`harp/interaction/session_tools.py`) | The tool's `description` field — what teaches the model it can hang up on itself when a visitor says goodbye. |
| [`transcription_openai.md`](transcription_openai.md) | OpenAI's input transcriber (`harp/voice/openai.py`), OpenAI provider only | Not spoken to the visitor-facing model — steers the side-channel transcriber that produces the dashboard's text so spoken Urdu comes out romanized (Latin script) instead of Perso-Arabic. Overridable per-run with the `OPENAI_TRANSCRIBE_PROMPT` env var. |
| [`transcription_whisper.md`](transcription_whisper.md) | the local wake-word transcriber (`harp/listener/transcriber.py`) | `initial_prompt` for the offline faster-whisper model that checks captured phrases for a wake word while HARP is idle — same romanization goal, so it can match the romanized `wake_words` in harp.yaml. Overridable per-run with the `HARP_WHISPER_PROMPT` env var. |
| [`identity_context.md`](identity_context.md) | face-ID → the responder (`harp/app.py::identity_context`) | Sent at session open when face-ID currently recognizes the person standing there and they have no `notes` on file: "you are talking to `{name}`." Not sent for unknown/unenrolled faces. |
| [`identity_context_with_notes.md`](identity_context_with_notes.md) | face-ID → the responder (`harp/app.py::identity_context`) | Same as above, used instead when the recognized person's `people/<id>/info.yaml` has a non-empty `notes` field — appends "Notes about them: `{notes}`." |
| [`wake_context_wave.md`](wake_context_wave.md) | wave-wake (`harp/triggers/engine.py`) | Sent at session open when a raised palm woke HARP. |
| [`wake_context_push_to_talk.md`](wake_context_push_to_talk.md) | push-to-talk (`harp/interaction/push_to_talk.py`) | Sent at session open when a press of the push-to-talk key woke HARP — tells the model to listen first since the person presses *then* speaks. |
| [`wake_context_loud_sound.md`](wake_context_loud_sound.md) | the always-on listener (`harp/listener/listener.py`) | Sent at session open when a sound at/above `listener.wake_level` woke HARP; `{level}` is the measured RMS. |
| [`wake_context_wake_word.md`](wake_context_wake_word.md) | the always-on listener (`harp/listener/listener.py`) | Sent at session open when a transcribed phrase matched one of `listener.wake_words`; `{text}` is the transcript. |
| [`memory_summarizer.md`](memory_summarizer.md) | the memory helper (`harp/memory/summarizer.py`) | Instruction for the parallel Flash Lite agent that turns each finished conversation into long-term memory; `{facts}` and `{transcript}` are filled in. Must ask for the JSON keys `summary`/`follow_up`/`person_facts` — that's what the summarizer parses out. |
| [`context_writer.md`](context_writer.md) | the memory helper (`harp/memory/context.py`) | Instruction for the pre-computed **wake briefing**: sent with the live camera frame when someone appears while HARP is idle; `{people}` is what the store remembers about the recognized faces. The reply is what the live session reads at open. |
| [`describe_scene.md`](describe_scene.md) | the memory helper (`harp/vision/describe.py`) | Vision instruction sent with the current frame when the live model calls `describe_scene` mid-conversation; `{focus}` is what it asked to look for. |
| [`describe_scene_tool.md`](describe_scene_tool.md) | the `describe_scene` tool (`harp/vision/describe.py`) | The tool's `description` field — what teaches the live model it can look through the camera mid-conversation. |
| [`search_memory_tool.md`](search_memory_tool.md) | the `search_memory` tool (`harp/memory/tools.py`) | The tool's `description` field — what teaches the live model it can search its own past conversations and the people it has met. |

The last six are **context delivered to the responder model at the start of a
conversation** — not personas or tool descriptions, but the "why did I just
wake up, and who's in front of me" stage directions. They're templates
(`{name}`, `{level}`, etc. get filled in by code); a template that's edited to
remove a placeholder the code expects falls back to the built-in default for
that one message rather than crashing the subsystem.

Everything else in [`harp.yaml`](../harp.yaml) is a numeric/behavioral knob
(thresholds, timeouts, toggles); this folder is specifically the *wording*
HARP's agents act on.
