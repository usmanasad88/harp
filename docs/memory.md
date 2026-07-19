# Memory (`harp/memory/`)

[← Back to index](index.md)

Long-term memory of people and conversations. One parallel **Gemini Flash Lite helper agent**
(separate from the live voice session, hard-capped at 14 calls/min to stay in the free tier)
serves three consumers: summarizing finished conversations, pre-computing wake briefings, and
answering the live model's `describe_scene` calls. Independent of the helper, raw transcripts are
always recorded, and the `search_memory` tool needs no model at all.

| File | Role |
|---|---|
| `store.py` | `MemoryStore` — the per-person JSON store (embeddings, notes, summaries) |
| `matcher.py` | Cosine-similarity face matching against the store |
| `agent.py` | `GeminiAgent` + `RateLimiter` — the shared, fail-safe helper client |
| `logger.py` | `InteractionLogger` — per-conversation turn-by-turn transcripts |
| `parse.py` | The transcript record vocabulary + mechanical digest |
| `summarizer.py` | `MemorySummarizer` — transcript → per-person memory / guestbook |
| `context.py` | `ContextWriter` — the pre-computed wake briefing |
| `tools.py` | The `search_memory` tool |

## On-disk layout

```
.harp/memory/
  people/                      one JSON file per enrolled person (the store)
  interactions/                per-conversation transcripts:
    interaction-<ts>.part        being written (live conversation)
    interaction-<ts>.jsonl       finalized, awaiting the summarizer
    interaction-<ts>.jsonl.done  summarized
    interaction-<ts>.jsonl.skipped  judged not worth a memory (nobody spoke)
  guestbook.jsonl              memories of unrecognized visitors (no faces stored)
```

## `store.py` — the person store

One JSON file per person under `.harp/memory/people/` — human-readable and hand-editable on
purpose, and at HARP's scale (dozens of people) a database would be pure overhead. Writes are
atomic (temp file + rename) so a crash mid-write can't corrupt a record.

`PersonRecord`: `person_id` (stable), `name`, `role`, `notes` (model-facing free text),
`embeddings` (face embeddings as float lists), `summaries` (accumulated `{ts, text, ...}`
entries). Two lifecycles live in one record: the **enrollment fields**
(name/role/notes/embeddings) are owned by `scripts/enroll_people.py` and replaced wholesale on
re-enrollment; **summaries** are appended by the summarizer and survive re-enrollment.

API: `upsert_person(dict)` (creates or updates; generates a slug id from the name if missing;
`embedding` is shorthand for a one-element `embeddings`), `add_summary(person_id, text,
**extra)` (extra fields like `follow_up`/`person_facts` are stored alongside; readers must treat
anything beyond ts/text as optional), `get(person_id)` (KeyError for unknown ids), `people()`.

## `matcher.py` — face matching

`match(embedding, store, threshold=0.4) -> (person_id | None, is_known, confidence)`. Cosine
similarity against every stored embedding, best one wins if it clears the threshold. InsightFace's
`normed_embedding`s are unit-length so the dot product *is* the cosine (the query is re-normalized
defensively anyway). Brute force on purpose — microseconds at this scale. The 0.4 threshold is
the usual verification cut-off for buffalo_l/ArcFace embeddings (same-person pairs ≈ 0.5–0.8,
different people below ~0.3); calibrate against your webcam with `scripts/preview_face_id.py`,
which prints per-face similarity.

## `agent.py` — the shared helper client

Two design rules, enforced here so no consumer can violate them:

1. **Fail-safe**: `GeminiAgent.generate()` returns `None` on ANY failure — rate limit, SDK
   error, timeout, missing key — and logs it. It never raises into a consumer; every consumer has
   a defined degradation for `None` (transcript stays pending; the static identity line is
   served; the model is told the camera helper is unavailable).
2. **One shared rate limit**: `RateLimiter` is a sliding 60 s window (`memory.calls_per_minute`,
   default 14) across all three consumers. Callers that can retry later use `wait=False` (skip
   immediately when the window is full); callers a model is waiting on use `wait=True` (block for
   a slot up to a timeout).

`generate(prompt, image_jpeg=None, json_response=False, wait=False, timeout=30)` supports an
attached JPEG (for the briefing and describe_scene) and a JSON response mode (for the
summarizer). The real SDK call is built lazily on first use and is injectable (`caller`), so
tests never touch the SDK. `parse_json(text)` tolerantly parses a model reply that should be a
JSON object (stripping a markdown code fence), returning None when it isn't one.

## `logger.py` — the interaction transcript

`InteractionLogger(bus, dir, people_now)` records one JSONL file per live conversation — the raw
record memory is built from (not to be confused with the per-run developer log). Record
vocabulary is owned by `parse.py`: `start` (wake reason + context), `person` (everyone face-ID
sees — seeded at open from the injected `people_now` getter since the bus won't replay a sighting
that happened before the wake, then updated live from `PersonIdentified` events), `turn`, `tool`,
`end`.

Turns arrive as **streamed pieces**: `final=False` deltas carrying the words, then a `final=True`
marker closing the turn (carrying empty text on the OpenAI path; possibly the last piece on
Gemini's). The logger accumulates deltas per speaker and writes one turn record at each final; a
turn still open at shutdown is flushed, not lost.

Crash tolerance: every line is flushed as written; the file keeps a `.part` suffix while the
conversation is live — a clean end renames it to `.jsonl` (pending, the summarizer's cue), while
a crash leaves `.part` behind for `rescue_stale_transcripts()` to promote at the next boot, so
even a conversation the app died in gets remembered. A write error drops that line, never the
agent.

## `parse.py` — the mechanical digest

Before the summarizer spends a rate-limited model call, everything extractable *without* a model
is extracted mechanically. `digest(records)` folds a transcript into facts: start/end times and
duration, wake and end reasons (a missing end record is reported as "did not end cleanly"),
deduplicated participants, the turns, `user_turns` count, and the knowledge-base queries the
assistant made. `render_facts(d)` renders those as one deterministic line per fact for the
prompt's `{facts}` slot; `render_transcript(d)` renders the turns as a plain "Visitor:/Assistant:"
block. Owning the record vocabulary here means logger and summarizer agree by importing this
module, not each other.

## `summarizer.py` — conversation → memory

`MemorySummarizer(bus, store, agent, dir, guestbook)` watches `InteractionEnded`. After a short
settle delay (the logger finalizes its file on the *same* event, and inter-subscriber order isn't
guaranteed), it `sweep()`s every pending transcript, oldest first:

1. **Digest** it; a transcript with zero user turns (a wake where nobody actually spoke) is
   marked `.skipped` without spending a call — a memory of it would be noise.
2. **One model call** (`wait=True`, 90 s budget — with the window full a slot is guaranteed
   within its 60 s period) with the summarizer prompt (`prompts/memory_summarizer.md`), asking
   for JSON `{summary, follow_up, person_facts}`. A malformed reply degrades to "the whole text
   is the summary" — a rough memory beats a lost one. A `None` reply (quota/network) stops the
   sweep early, leaving the rest pending rather than hammering a dead quota.
3. **Store**: every *enrolled* participant gets the summary appended to their record — this is
   what the next meeting's briefing reads. If no participant was enrolled, the memory goes to the
   **guestbook** (`guestbook.jsonl`, one entry per interaction) instead: the conversation is
   remembered but no face is ever stored for a stranger; the entry carries the transcript
   filename so a later enrollment can re-attach the full conversation.
4. Publish `MemoryWritten` and rename the transcript `.done`.

Boot recovery: `run()` first rescues crashed `.part` files, then sweeps anything still pending —
so transcripts recorded by a run *without* a `GEMINI_API_KEY` are summarized by a later run that
has one.

## `context.py` — the pre-computed wake briefing

The problem is latency: the live session receives its context at open, but a helper call takes
~1 s, and blocking the wake on it would delay HARP's first word. So the briefing is computed
**before** the wake: face-ID announces people the moment they appear (usually seconds before they
wave or speak), and `ContextWriter` reacts by asking the helper to fuse the current clean camera
frame with what the store remembers about the recognized faces into a short "who you are about to
talk to" paragraph — cached and ready. At session open, `context()` hands it over instantly;
`app.py` falls back to the static face-ID identity line when there's nothing fresh.

Freshness contract: regenerate when **who** is in frame changes — the cache key is the identity
set plus the head-count (the count catches a second unknown face joining, which doesn't change
the set) — and every `memory.context_ttl_seconds` (default 120 s) while someone stays in frame.
The refresh loop runs only while HARP is on **standby** with someone present; an empty frame
clears the cache; a session opening stops it. Calls go through the shared limiter **non-blocking**
(`wait=False`) — a briefing is a nice-to-have and must never starve the summarizer; on a full
window the cached briefing is simply served a little staler (up to 2×TTL) or the static line
wins. Each successful generation publishes `ContextPrepared` (visible on the dashboard).

The prompt (`prompts/context_writer.md`) receives a `{people}` block rendered from the store:
per recognized person, their name/role/notes and their last 5 summaries with any open follow-ups;
unknown faces are described as visitors with no stored history.

## `tools.py` — the `search_memory` tool

The mid-session counterpart to the briefing: when a visitor asks "do you remember me?" or refers
to an earlier conversation, the model calls `search_memory(query)` instead of guessing. The
corpus is everything long-term memory holds — every person's enrollment notes and interaction
summaries plus the guestbook — rebuilt per call (it's tiny, and it means a summary written
seconds ago is immediately findable).

Ranking is plain query-token overlap using the **same tokenizer** as the knowledge retriever (so
both search tools agree on what a word is), with a ×2 bonus for query tokens matching the
person's *name* — a name should outrank a passing mention in some summary body. Returns up to 5
entries as `{person, when, text}`, best first, or `{"note": "no matches found"}`. Needs no API
key and no model.
