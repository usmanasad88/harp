# search_knowledge — tool description

> Authoring notes (this title line and blockquotes) are stripped before the
> model sees this — see `harp/config.py` (`load_search_tool_description`,
> shared `load_prompt`). This text becomes the `search_knowledge` tool's
> `description` field, sent to the model alongside its name and parameters.
> It is the ONLY thing that teaches the model when and how to call the tool,
> so treat it like an instruction, not documentation. Consumed by
> `harp/knowledge/tools.py`.

Search the local knowledge base for facts before answering. It contains everything HARP has been given documents about. The documents are written in English, so always query with concise English keywords even if the visitor spoke Urdu. Call this BEFORE answering any factual question about the venue, event, schedule, or anything else the documents might cover, and base your spoken reply on what it returns. If it returns nothing useful, say you are not sure rather than guessing.
