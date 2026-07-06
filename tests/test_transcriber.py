"""PhraseTranscriber loads faster-whisper offline-first: once the model is
cached it must load with local_files_only=True (no network HEAD check that can
hang on a flaky connection), only falling back to a networked download when the
model genuinely isn't cached yet. faster_whisper is faked here — no model, no
download."""

from __future__ import annotations

import sys
import types

from harp.listener.transcriber import PhraseTranscriber


def _install_fake_whisper(monkeypatch, on_construct):
    """Replace `faster_whisper` with a module whose WhisperModel records how it
    was constructed and can simulate 'not cached' by raising."""
    calls: list[bool | None] = []

    class FakeWhisperModel:
        def __init__(self, name, *, device, compute_type, local_files_only=None):
            calls.append(local_files_only)
            on_construct(local_files_only)  # may raise to simulate a cache miss

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    return calls


def test_cached_model_loads_offline_without_network(monkeypatch):
    calls = _install_fake_whisper(monkeypatch, on_construct=lambda offline: None)

    PhraseTranscriber("base")._load()

    assert calls == [True]  # one construction, offline — never touched the network


def test_uncached_model_falls_back_to_download(monkeypatch):
    def on_construct(offline):
        if offline:  # first, offline attempt: pretend nothing is cached
            raise RuntimeError("model not found locally")

    calls = _install_fake_whisper(monkeypatch, on_construct=on_construct)

    PhraseTranscriber("base")._load()

    # Tried offline first, then downloaded (local_files_only unset → None).
    assert calls == [True, None]
