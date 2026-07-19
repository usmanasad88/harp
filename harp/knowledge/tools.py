"""The tool bridge: expose knowledge to the voice provider as function calls.

This is the seam that connects retrieval to the live model. It provides
(1) the tool *declarations* to place in SessionConfig.tools, shaped per
provider, and (2) `dispatch(name, arguments)`, which runs a requested tool and
returns its output for VoiceConnection.respond_tool.

Flow the voice bridge wires up:
    provider emits ToolCall
      → tools.dispatch(name, args)          (runs the retriever)
      → conn.respond_tool(call, output)      (answer goes back to the model)
      → bus: ToolRequested / ToolCompleted   (so the dashboard can show it)

The declaration below carries the behavioral levers proven in the
web-realtime sandbox — query in concise English keywords (the corpus is
English even when the visitor speaks Urdu), call BEFORE answering, and admit
uncertainty on no hits — reworded to stay corpus-agnostic per PLAN.md
("nothing is hardcoded to a specific corpus"). Errors come back as an
{"error": ...} payload rather than raising, so a bad tool call degrades into
the model apologizing instead of the session crashing.

web_search is the internet fallback (see web_search.py): advertised alongside
search_knowledge, its description tells the model to reach for it only when
the local store came up empty.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..config import (
    DATA_DIR,
    load_search_tool_description,
    load_web_search_tool_description,
)
from . import web_search
from .retriever import Retriever

TOOL_NAME = "search_knowledge"
WEB_SEARCH_TOOL_NAME = "web_search"

# Wording lives in prompts/search_knowledge_tool.md and prompts/web_search_tool.md
# (see prompts/README.md) — this is what teaches the model when/how to call each tool.
_DESCRIPTION = load_search_tool_description()
_WEB_SEARCH_DESCRIPTION = load_web_search_tool_description()

_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A few English keywords describing what to look up.",
        }
    },
    "required": ["query"],
}

_WEB_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A few English keywords, like a search-engine query.",
        }
    },
    "required": ["query"],
}


def declarations(provider: str) -> list[dict]:
    """Tool declarations for SessionConfig.tools, in `provider`'s format."""
    bases = [
        {"name": TOOL_NAME, "description": _DESCRIPTION, "parameters": _PARAMETERS},
        {
            "name": WEB_SEARCH_TOOL_NAME,
            "description": _WEB_SEARCH_DESCRIPTION,
            "parameters": _WEB_SEARCH_PARAMETERS,
        },
    ]
    if provider == "openai":
        # The GA realtime session format: flat function entries.
        return [{"type": "function", **base} for base in bases]
    if provider == "gemini":
        # google-genai accepts plain dicts shaped like types.Tool (the same
        # declaration format aura's voice_action_bridge proved out).
        return [{"function_declarations": bases}]
    raise ValueError(f"unknown provider: {provider!r} (expected 'gemini' or 'openai')")


# Built lazily on first use so importing this module stays free; the corpus is
# small enough that the one-time index build is milliseconds.
_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(DATA_DIR)
    return _retriever


def index_size() -> int:
    """Number of indexed chunks, building the index if it isn't yet. Handy for
    a startup line ("N chunks indexed from data/") and it warms the index so
    the first real search_knowledge call isn't paying the file-read cost."""
    return len(_get_retriever())


async def dispatch(name: str, arguments: dict) -> Any:
    """Run a tool call by name and return its result for respond_tool."""
    query = str(arguments.get("query", "")).strip()
    if name == WEB_SEARCH_TOOL_NAME:
        try:
            # Blocking HTTP request; keep it off the event loop.
            results = await asyncio.to_thread(web_search.search, query)
        except web_search.WebSearchError as exc:
            return {"error": str(exc)}
        return results if results else {"note": "no results found"}
    if name != TOOL_NAME:
        return {"error": f"unknown tool: {name}"}
    # First call may build the index (file reads); keep it off the event loop.
    results = await asyncio.to_thread(lambda: _get_retriever().search(query))
    # Same shapes the sandbox returned: matches, or an explicit no-match note.
    return results if results else {"note": "no matches found"}
