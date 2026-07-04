"""Build / refresh the vector index from data/ — run once, not per query.

Reads the documents under data/ (markdown, PDF, captioned images...), splits them
into chunks, embeds them, and writes a persistent vector store. Re-run when the
corpus changes. Kept separate from retrieval so the indexing cost is paid up
front, not on every question.

Run standalone:
    python -m harp.knowledge.indexer

To build:
  - loaders per file type + a chunking strategy,
  - an embedding model (local vs API) and a persistent store (Chroma / FAISS /
    sqlite-vec),
  - incremental re-index (skip files whose content hash is unchanged).
"""

from __future__ import annotations

from pathlib import Path


def build_index(data_dir: Path, store_dir: Path) -> None:
    """Index every document under `data_dir` into a persistent store at `store_dir`."""
    raise NotImplementedError


def main() -> None:
    """CLI: index everything under the repo's data/ folder into the store."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
