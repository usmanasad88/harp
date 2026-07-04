"""Knowledge: the BM25 retriever (ported from web-realtime/knowledge.js) and
the search_knowledge tool bridge. Pure tmp_path corpora — no app, no model."""

from __future__ import annotations

import pytest

import harp.knowledge.tools as tools
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
    (openai_decl,) = tools.declarations("openai")
    assert openai_decl["type"] == "function"
    assert openai_decl["name"] == tools.TOOL_NAME
    assert openai_decl["parameters"]["required"] == ["query"]

    (gemini_decl,) = tools.declarations("gemini")
    (fn,) = gemini_decl["function_declarations"]
    assert fn["name"] == tools.TOOL_NAME

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
