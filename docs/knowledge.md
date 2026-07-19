# Knowledge (`harp/knowledge/`)

[← Back to index](index.md)

Retrieval-augmented answering: the live model grounds factual answers in the markdown corpus
under `data/` via the `search_knowledge` tool, with an internet fallback (`web_search`) for
questions the local documents can't cover.

| File | Role |
|---|---|
| `retriever.py` | BM25 keyword search over `data/*.md` |
| `tools.py` | The tool bridge: declarations + `dispatch()` for both tools |
| `web_search.py` | DuckDuckGo HTML search, no API key |
| `indexer.py` | Reserved stub for a future embedding/vector index |

## `retriever.py` — BM25 over `data/`

A Python port of the `web-realtime` sandbox's `knowledge.js`, the version proven in live
sessions. **No embeddings, no vector store** — at this corpus size keyword relevance works and
costs nothing; if the corpus outgrows it, swapping in embeddings means rewriting only this module
(`indexer.py` is the reserved seam) while the `search()` interface stays.

- **Chunking** (`_chunk_markdown`): every `data/*.md` file is split at markdown headings; each
  chunk keeps its heading as part of its text and remembers its source filename.
- **Tokenization** (`tokenize`, public): lowercase, keep Latin word characters *plus the
  Arabic-script block* (U+0600–U+06FF) so Urdu queries work; drop one-character tokens and a
  stop-word list. `memory/tools.py` imports this same function so both search tools agree on what
  a word is.
- **Index**: built in memory at construction (milliseconds for a small corpus) — term
  frequencies per chunk, document frequencies, average chunk length. Construct once at app
  start, not per query.
- **`search(query, k=3)`**: standard BM25 (k1=1.5, b=0.75) over the query's token set; returns
  up to k chunks as `{"source", "heading", "text", "score"}` — the exact result shape the
  sandbox returned to the model.

## `tools.py` — the tool bridge

The seam connecting retrieval to the live model. Provides:

- `declarations(provider)` — the tool declarations to place in `SessionConfig.tools`, shaped per
  provider: OpenAI's flat `{"type": "function", ...}` entries, or Gemini's
  `{"function_declarations": [...]}` wrapper. Both tools take a single `query` string ("a few
  English keywords").
- `dispatch(name, arguments)` — runs a requested tool and returns its output for
  `respond_tool`. `search_knowledge` runs the retriever via `asyncio.to_thread` (the first call
  may build the index — file reads stay off the event loop); `web_search` runs the blocking HTTP
  search off-thread and converts `WebSearchError` into an `{"error": ...}` payload. Empty results
  return an explicit `{"note": "no matches found"}` so the model knows to admit uncertainty.
  An unknown tool name returns `{"error": "unknown tool: ..."}`.
- `index_size()` — the chunk count, used for the startup line ("N chunks indexed from data/")
  and to warm the index before the first real query.

The retriever singleton is built lazily on first use so importing the module stays free.

The tool *descriptions* — the text that teaches the model when to call each tool — live in
`prompts/search_knowledge_tool.md` and `prompts/web_search_tool.md`. Their behavioral levers,
proven in the sandbox: query in concise **English** keywords (the corpus is English even when the
visitor speaks Urdu), call **before** answering factual questions, base the spoken reply on what
comes back, and admit uncertainty on no hits. The web-search description tells the model to reach
for it **only** when `search_knowledge` came up empty.

## `web_search.py` — the internet fallback

Backend: DuckDuckGo's plain-HTML endpoint (`html.duckduckgo.com`) — no API key, no new
dependency, just stdlib urllib + regex over a page layout that has been stable for years.

- `search(query, k=3, timeout=6)` — blocking (callers wrap in `asyncio.to_thread`); returns up
  to k `{title, url, snippet}` results with snippets capped at 300 characters (short, quotable,
  keeps the model's context small).
- The request must present a real-browser User-Agent — urllib's default (and even a polite
  bot-style UA) gets served DDG's anomaly page with zero results.
- `_parse` pairs each `result__a` title anchor with the `result__snippet` that follows it in
  document order, skips ad slots (links routed through `y.js`), and unwraps DDG's redirect links
  (`uddg=` parameter) so the model quotes real source URLs.
- Failure model: a drifted page layout yields `[]` (the model says it found nothing); network
  trouble raises `WebSearchError`, which `tools.dispatch` turns into an `{"error": ...}` payload
  the model can apologize with — never a crashed session.

## `indexer.py` — reserved stub

The documented seam for a future persistent vector index (loaders per file type, an embedding
model, incremental re-index). Currently `NotImplementedError`; the BM25 retriever makes it
unnecessary at today's corpus size.

## Feeding the corpus

`data/` is plain markdown; anything placed there is indexed at next start. The helper
`scripts/scrape_site.py` crawls a website's same-domain content pages and writes one clean
markdown file per page into `data/` (used to ingest the expo/university site content). See
[Data, prompts & on-disk state](data-and-state.md).
