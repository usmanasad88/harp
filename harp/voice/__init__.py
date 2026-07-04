"""The real-time conversation core.

One provider-agnostic interface (`provider.py`), one implementation per
backend (`gemini.py`, `openai.py`, and later a Moshi offline fallback).
`get_provider` imports backends lazily so choosing Gemini never requires the
OpenAI dependencies, and vice versa.
"""

from __future__ import annotations


def get_provider(name: str):
    """Return a provider instance by name, importing its backend on demand."""
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider()
    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"unknown provider: {name!r} (expected 'gemini' or 'openai')")
