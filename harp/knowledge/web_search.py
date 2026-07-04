"""Internet-search fallback for when the local store can't answer.

Used only when retrieval comes up empty or weak, so HARP isn't limited to data/.
Returns short, quotable snippets (not whole pages) to keep the model's context
small. Which search API to use is an open choice.

To build:
  - `search(query) -> snippets` over some search API,
  - trim / rank the results, with a timeout and a graceful "no network" path.
"""

from __future__ import annotations


def search(query: str, k: int = 3) -> list:
    """Return up to k short web snippets for `query` (the fallback path)."""
    raise NotImplementedError
