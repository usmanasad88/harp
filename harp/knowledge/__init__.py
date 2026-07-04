"""Knowledge: retrieval-augmented answers over whatever is in data/.

Context-agnostic by design — drop any documents into data/ and they become
searchable; nothing is hardcoded to the expo corpus. Exposed to the voice model
as a tool (search_knowledge) so it can look things up mid-conversation, with an
internet-search fallback for when the local store comes up empty.

Pipeline:  indexer (build once) → retriever (query) → tools (bridge to provider).
The `web-realtime/` sandbox is prototyping the RAG/tool-calling shape first; this
package is where the settled design lands for the headless agent.
"""
