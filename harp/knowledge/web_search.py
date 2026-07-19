"""Internet-search fallback for when the local store can't answer.

Used only when retrieval comes up empty or weak, so HARP isn't limited to data/.
Returns short, quotable snippets (not whole pages) to keep the model's context
small.

Backend: DuckDuckGo's plain-HTML endpoint (html.duckduckgo.com) — no API key,
no new dependency, just stdlib urllib + regex over a page layout that has been
stable for years. The parse is deliberately forgiving: if the layout drifts we
get [] (the model says it found nothing), and network trouble raises
WebSearchError, which tools.dispatch turns into an {"error": ...} payload the
model can apologize with instead of the session crashing.
"""

from __future__ import annotations

import html as html_lib
import re
import urllib.error
import urllib.parse
import urllib.request

_SEARCH_URL = "https://html.duckduckgo.com/html/"
# Must look like a real browser: urllib's default UA — and even a polite
# "bot-style" UA with a contact URL — gets served the anomaly page (0 results).
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_SNIPPET_MAX_CHARS = 300

# One regex per piece, paired by document order: each result title anchor, then
# the snippet anchor that follows it (see _parse for the pairing).
_RESULT_A_RE = re.compile(r'class="result__a"[^>]*\bhref="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)


class WebSearchError(RuntimeError):
    """The search request itself failed (no network, timeout, blocked)."""


def _clean(fragment: str) -> str:
    """Strip tags, unescape entities, collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", fragment)
    return " ".join(html_lib.unescape(text).split())


def _real_url(href: str) -> str:
    """DDG wraps result links as //duckduckgo.com/l/?uddg=<real-url>&rut=...;
    unwrap to the destination so the model quotes a real source."""
    parsed = urllib.parse.urlparse(href)
    uddg = urllib.parse.parse_qs(parsed.query).get("uddg")
    if uddg:
        return uddg[0]
    return "https:" + href if href.startswith("//") else href


def _parse(page: str, k: int) -> list[dict]:
    """Extract up to k {title, url, snippet} results from the HTML page."""
    results: list[dict] = []
    anchors = list(_RESULT_A_RE.finditer(page))
    for i, match in enumerate(anchors):
        href, title = match.group(1), _clean(match.group(2))
        if "duckduckgo.com/y.js" in href:  # ad slots route through y.js
            continue
        # The snippet for this result sits between this anchor and the next one.
        block_end = anchors[i + 1].start() if i + 1 < len(anchors) else len(page)
        snippet_match = _SNIPPET_RE.search(page, match.end(), block_end)
        snippet = _clean(snippet_match.group(1)) if snippet_match else ""
        results.append(
            {
                "title": title,
                "url": _real_url(href),
                "snippet": snippet[:_SNIPPET_MAX_CHARS],
            }
        )
        if len(results) >= k:
            break
    return results


def search(query: str, k: int = 3, timeout: float = 6.0) -> list[dict]:
    """Return up to k short web snippets for `query` (the fallback path).

    Blocking — callers run it via asyncio.to_thread, same as the retriever.
    """
    query = query.strip()
    if not query:
        return []
    url = _SEARCH_URL + "?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            page = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WebSearchError(f"web search unavailable: {exc}") from exc
    return _parse(page, k)
