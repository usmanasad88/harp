"""Keyword search over data/ — the read side of search_knowledge.

A Python port of the web-realtime sandbox's knowledge.js, the version proven
in live sessions: chunk every data/*.md at headings, rank chunks against the
query with BM25 (a standard keyword-relevance score). No embeddings, no vector
store — at this corpus size keyword relevance works and costs nothing. If the
corpus outgrows it, swapping in embeddings means rewriting only this module
(indexer.py is the reserved seam); the search() interface stays.

The index is built in memory at construction (milliseconds for a small
corpus), so construct once at app start, not per query. Tokenization keeps
Latin word characters and the Arabic-script range so Urdu queries work too.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Common words that add noise to keyword matching.
_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been being with from by as it its this "
    "that these those you your we our they i he she him her them not no do does did how what when "
    "where who why which will would can could should about into over under more most there here".split()
)

# Latin word chars plus the Urdu/Arabic script block (U+0600–U+06FF).
_TOKEN_RE = re.compile(r"[a-z0-9؀-ۿ]+")

# BM25 constants — the standard defaults, same as the sandbox.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """Search tokens for a query or a chunk: lowercased words minus stopwords.
    Public because memory/tools.py ranks memory entries with the same
    tokenization, so both search tools agree on what a 'word' is."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOP]


@dataclass
class _Chunk:
    source: str
    heading: str
    text: str
    tf: Counter = field(default_factory=Counter)
    length: int = 0


def _chunk_markdown(text: str, source: str) -> list[_Chunk]:
    """Split a markdown file into chunks at headings, each keeping its heading."""
    chunks: list[_Chunk] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        body_text = "\n".join(body).strip()
        if body_text or heading:
            chunk_text = (f"{heading}\n" if heading else "") + body_text
            if chunk_text.strip():
                chunks.append(_Chunk(source=source, heading=heading, text=chunk_text))

    for line in text.split("\n"):
        if re.match(r"^#{1,6}\s", line):
            flush()
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            body = []
        else:
            body.append(line)
    flush()
    return chunks


class Retriever:
    def __init__(self, data_dir) -> None:
        self._chunks: list[_Chunk] = []
        self._df: Counter = Counter()  # term -> number of chunks containing it
        self._avg_len = 0.0

        data_path = Path(data_dir)
        files = sorted(data_path.glob("*.md")) if data_path.is_dir() else []
        for file in files:
            text = file.read_text(encoding="utf-8")
            self._chunks.extend(_chunk_markdown(text, file.name))

        total_len = 0
        for chunk in self._chunks:
            tokens = tokenize(chunk.text)
            chunk.tf = Counter(tokens)
            chunk.length = len(tokens)
            total_len += chunk.length
            self._df.update(chunk.tf.keys())
        self._avg_len = total_len / len(self._chunks) if self._chunks else 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return up to k relevant chunks for `query`, best first, each as
        {"source", "heading", "text", "score"} — the exact result shape the
        sandbox's search_knowledge tool returned to the model."""
        n = len(self._chunks) or 1
        terms = set(tokenize(query or ""))
        if not terms:
            return []

        scored: list[tuple[float, _Chunk]] = []
        for chunk in self._chunks:
            score = 0.0
            for term in terms:
                tf = chunk.tf.get(term)
                if not tf:
                    continue
                df = self._df.get(term, 1)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * (
                    (tf * (_K1 + 1))
                    / (tf + _K1 * (1 - _B + (_B * chunk.length) / self._avg_len))
                )
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "source": chunk.source,
                "heading": chunk.heading,
                "text": chunk.text,
                "score": round(score, 3),
            }
            for score, chunk in scored[:k]
        ]
