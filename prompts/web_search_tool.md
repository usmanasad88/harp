# web_search — tool description

> Authoring notes (this title line and blockquotes) are stripped before the
> model sees this — see `harp/config.py` (`load_web_search_tool_description`,
> shared `load_prompt`). This text becomes the `web_search` tool's
> `description` field, sent to the model alongside its name and parameters.
> It is the ONLY thing that teaches the model when and how to call the tool,
> so treat it like an instruction, not documentation. Consumed by
> `harp/knowledge/tools.py`; the search itself is
> `harp/knowledge/web_search.py` (DuckDuckGo, no API key).

Search the internet for current or general information. Use this ONLY when search_knowledge returned nothing useful and the question is about something outside the local documents — news, weather, general facts. Query with a few concise English keywords, like a search-engine query. Base your spoken reply on the returned snippets and keep it brief. If this returns nothing useful or an error, say you could not find out rather than guessing.
