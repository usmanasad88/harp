# OpenAI input-transcription steering prompt

> Authoring notes stripped before use — see `harp/config.py`
> (`load_openai_transcribe_prompt`). This is NOT sent to the responder model
> that talks to the visitor — it primes the separate OpenAI Realtime *input
> transcriber* (`gpt-4o-mini-transcribe`) that produces the dashboard's
> user-transcript text. Whisper-family transcribers can echo this prompt back
> verbatim on silence/noise; `harp/voice/openai.py` already guards against
> that leaking into the transcript. Override at runtime with the
> `OPENAI_TRANSCRIBE_PROMPT` env var without touching this file.

The speaker uses only English or Urdu. Write the whole transcript in the Latin alphabet: leave English as English, and romanize spoken Urdu into Latin letters (e.g. "aap kaise hain") — never Urdu (Perso-Arabic), Arabic, or Hindi/Devanagari script.
