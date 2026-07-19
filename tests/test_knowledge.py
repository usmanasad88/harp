"""Knowledge: the BM25 retriever (ported from web-realtime/knowledge.js) and
the search_knowledge tool bridge. Pure tmp_path corpora — no app, no model."""

from __future__ import annotations

import pytest

import harp.knowledge.tools as tools
import harp.knowledge.web_search as web_search
from harp.knowledge.retriever import Retriever, _chunk_markdown


def _write_corpus(tmp_path):
    (tmp_path / "expo.md").write_text(
        "# Expo\n"
        "## Tickets\n"
        "Tickets cost 500 rupees at the gate. Students enter free with ID.\n"
        "## Venue\n"
        "The expo is held at the NUST gymnasium in Islamabad.\n",
        encoding="utf-8",
    )
    (tmp_path / "speakers.md").write_text(
        "# Speakers\n"
        "Dr. Ada Lovelace speaks about analytical engines at 3 pm.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_chunks_split_at_headings_and_keep_them():
    chunks = _chunk_markdown("# Top\nintro\n## Sub\ndetail here\n", "f.md")
    assert [c.heading for c in chunks] == ["Top", "Sub"]
    assert chunks[1].text == "Sub\ndetail here"
    assert all(c.source == "f.md" for c in chunks)


def test_search_ranks_the_relevant_chunk_first(tmp_path):
    retriever = Retriever(_write_corpus(tmp_path))
    results = retriever.search("ticket price rupees")
    assert results, "expected at least one hit"
    assert results[0]["heading"] == "Tickets"
    assert results[0]["source"] == "expo.md"
    assert results[0]["score"] > 0


def test_search_empty_query_and_no_match_return_empty(tmp_path):
    retriever = Retriever(_write_corpus(tmp_path))
    assert retriever.search("") == []
    assert retriever.search("zeppelin quantum marmalade") == []


def test_search_respects_k(tmp_path):
    retriever = Retriever(_write_corpus(tmp_path))
    results = retriever.search("expo speakers tickets venue", k=1)
    assert len(results) == 1


def test_missing_data_dir_yields_empty_index(tmp_path):
    retriever = Retriever(tmp_path / "does-not-exist")
    assert len(retriever) == 0
    assert retriever.search("anything") == []


def test_declarations_per_provider():
    knowledge_decl, web_decl = tools.declarations("openai")
    assert knowledge_decl["type"] == "function"
    assert knowledge_decl["name"] == tools.TOOL_NAME
    assert knowledge_decl["parameters"]["required"] == ["query"]
    assert web_decl["type"] == "function"
    assert web_decl["name"] == tools.WEB_SEARCH_TOOL_NAME
    assert web_decl["parameters"]["required"] == ["query"]

    (gemini_decl,) = tools.declarations("gemini")
    fn, web_fn = gemini_decl["function_declarations"]
    assert fn["name"] == tools.TOOL_NAME
    assert web_fn["name"] == tools.WEB_SEARCH_TOOL_NAME

    with pytest.raises(ValueError):
        tools.declarations("nope")


async def test_dispatch_searches_and_flags_no_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_retriever", Retriever(_write_corpus(tmp_path)))

    hits = await tools.dispatch(tools.TOOL_NAME, {"query": "tickets"})
    assert isinstance(hits, list) and hits[0]["heading"] == "Tickets"

    assert await tools.dispatch(tools.TOOL_NAME, {"query": "marmalade zeppelin"}) == {
        "note": "no matches found"
    }


async def test_dispatch_unknown_tool_reports_error_instead_of_raising():
    output = await tools.dispatch("fly_to_moon", {})
    assert "error" in output


# A trimmed html.duckduckgo.com result page: one organic hit (redirect-wrapped
# URL, entities, bold tags), one ad (routes through y.js), one hit without a
# snippet. The regex parse fails *silently* (returns junk or []) if any of
# this drifts, which is exactly why it's pinned here.
_DDG_PAGE = """
<div class="result results_links results_links_deep web-result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnust.edu.pk%2Fadmissions&amp;rut=abc123">
     NUST <b>Admissions</b> &amp; Aid</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnust.edu.pk%2Fadmissions">
     Apply to <b>NUST</b> &mdash; deadlines &amp; fees.</a>
</div>
<div class="result result--ad">
  <a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_provider=x">Buy Now</a>
  <a class="result__snippet" href="#">Sponsored thing.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.com/plain">Plain link</a>
</div>
"""


def test_web_search_parse_unwraps_urls_skips_ads_and_pairs_snippets():
    first, second = web_search._parse(_DDG_PAGE, k=5)
    assert first == {
        "title": "NUST Admissions & Aid",
        "url": "https://nust.edu.pk/admissions",
        "snippet": "Apply to NUST — deadlines & fees.",
    }
    # The ad is skipped; the snippetless result still comes through, and its
    # snippet is NOT stolen from a neighbouring block.
    assert second["url"] == "https://example.com/plain"
    assert second["snippet"] == ""


def test_web_search_parse_respects_k():
    results = web_search._parse(_DDG_PAGE, k=1)
    assert len(results) == 1


async def test_dispatch_web_search_offline_degrades_to_error_payload(monkeypatch):
    def _no_network(query, k=3, timeout=6.0):
        raise web_search.WebSearchError("web search unavailable: no network")

    monkeypatch.setattr(web_search, "search", _no_network)
    output = await tools.dispatch(tools.WEB_SEARCH_TOOL_NAME, {"query": "weather"})
    assert output == {"error": "web search unavailable: no network"}


async def test_dispatch_web_search_no_hits_returns_note(monkeypatch):
    monkeypatch.setattr(web_search, "search", lambda query, k=3, timeout=6.0: [])
    output = await tools.dispatch(tools.WEB_SEARCH_TOOL_NAME, {"query": "zeppelin"})
    assert output == {"note": "no results found"}
