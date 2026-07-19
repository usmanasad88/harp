# Agent tools — everything the live model can call

[← Back to index](index.md)

During a conversation the realtime model can call function tools. This page collects them all in
one place; each also appears in its home package's page.

## Common shape

Every tool module follows the same template:

- `declarations(provider)` returns the tool's declaration in the provider's format — OpenAI:
  `[{"type": "function", "name", "description", "parameters"}]`; Gemini:
  `[{"function_declarations": [ ... ]}]`.
- The **description** — the text that teaches the model *when and how* to call the tool — is
  loaded from a markdown file in `prompts/`, with a hardcoded fallback in `config.py`.
- An async **handler** returns a small JSON-able result. Errors come back as
  `{"error": "..."}` payloads, never exceptions — a bad call degrades into the model
  apologizing instead of the session crashing.
- `app.py` assembles the declaration list per run (only capabilities that are actually enabled
  and working are advertised) and routes calls in its `dispatch()` function.
- The voice bridge mirrors every call onto the bus (`ToolRequested` → `ToolCompleted`) so the
  dashboard shows it live.

## The tools

### `search_knowledge` — ground answers in `data/`
*`harp/knowledge/tools.py` · description: `prompts/search_knowledge_tool.md` · always available*

Args: `query` (a few **English** keywords — the corpus is English even when the visitor speaks
Urdu). Returns up to 3 BM25-ranked chunks `{source, heading, text, score}` from `data/*.md`, or
`{"note": "no matches found"}`. The description instructs: call BEFORE answering any factual
question, base the spoken reply on the results, admit uncertainty on a miss.

### `web_search` — internet fallback
*`harp/knowledge/tools.py` + `web_search.py` · description: `prompts/web_search_tool.md` · always available*

Args: `query`. DuckDuckGo HTML search (no API key), up to 3 `{title, url, snippet}` results with
300-char snippets. Taught to be used ONLY when `search_knowledge` came up empty and the question
is outside the local documents (news, weather, general facts).

### `end_session` — hang up
*`harp/interaction/session_tools.py` · description: `prompts/end_session_tool.md` · always available*

Args: optional `reason`. Publishes `EndOfInteractionDetected(cause="agent")`, which the
orchestrator handles like any end rule (ACTIVE → STANDBY). Taught to say a short spoken goodbye
FIRST, because the hang-up is immediate. The status rule book plays "Going on standby." for this
cause (not a second "Goodbye." — the model already said its own).

### `search_memory` — its own history
*`harp/memory/tools.py` · description: `prompts/search_memory_tool.md` · when `memory.enabled` (no API key needed)*

Args: `query` (English keywords — a name, a topic). Token-overlap search (name matches boosted
×2) over every enrolled person's notes + interaction summaries + the guestbook. Returns up to 5
`{person, when, text}` entries or `{"note": "no matches found"}`. For "do you remember me?" and
references to earlier visits; taught to say honestly that it doesn't remember on a miss.

### `describe_scene` — look through the camera
*`harp/vision/describe.py` · descriptions: `prompts/describe_scene_tool.md` + `describe_scene.md` · when the memory helper AND camera are up*

Args: optional `focus` ("how many people", "what the visitor is holding"). Sends the current
**clean** (overlay-free) camera frame plus a vision prompt to the Gemini Flash Lite helper;
waits for a rate-limiter slot (a visitor is audibly waiting) with a 15 s cap. Returns
`{"description": ...}` or `{"error": ...}` (no frame / helper unavailable). Taught to say
something brief first because the call takes a moment.

### `move_around` — the stall patrol
*`harp/motion/tools.py` → `MoveAroundController` · description: `prompts/move_around_tool.md` · when `motion.enabled`*

Args: `action` = `start` (default) | `stop`. Start launches the bounded dead-reckoning patrol lap
(drive → look-around → corner turns, `laps` laps, then stops by itself) and returns a note with
the estimated duration; refused with a reason while follow mode holds the motors. Stop halts at
the next 20 Hz check and waits for the wheels to zero. Taught: say something brief first (it
keeps talking while driving), stop the MOMENT anyone asks, don't start with someone right in
front.

### `follow_person` — follow a known person
*`harp/motion/tools.py` → `FollowController` · description: `prompts/follow_tool.md` · when `motion.enabled` AND camera + face-ID are running*

Args: `action` = `start` (default) | `stop`. Start refuses unless face-ID currently recognizes an
**enrolled** person ("no known person in frame..."), else drives toward them — steering to keep
the face centered, using face size as the distance proxy, stopping when close enough — and
announces itself with a canned safety clip. Auto-stops when the person is unseen/unrecognized
past `follow_lost_seconds`. Refused while a patrol is running. Taught: warn about safe distance,
stop instantly on request, and never pretend to follow when the tool returned an error.

## Context injected at session open (not tools, but adjacent)

Alongside tools, the model receives one opening text message assembled by the bridge:

1. **Wake context** — why it woke, from the matching `prompts/wake_context_*.md` template:
   wave / push-to-talk ("listen first, don't launch into a long welcome") / loud sound (with the
   measured `{level}`) / wake word (with the transcribed `{text}`).
2. **Identity context** — who it's talking to: preferably the memory helper's pre-computed
   briefing (`(Briefing from your vision and memory helper: ...)`), else the static face-ID line
   (`identity_context.md`, or `identity_context_with_notes.md` when the person's record has
   notes), else nothing for strangers.
