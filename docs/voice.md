# Voice (`harp/voice/`)

[← Back to index](index.md)

The realtime conversation core: one provider-agnostic interface, one implementation per cloud
backend, the audio hardware wrappers, and two "runners" that drive a session — the bare
standalone runner and the supervised bridge the orchestrator uses. Also home to the experimental
two-agent noise filter.

Files:

| File | Role |
|---|---|
| `provider.py` | The abstraction every backend implements: `SessionConfig` in, normalized `VoiceEvent`s out |
| `gemini.py` | Gemini Live backend |
| `openai.py` | OpenAI Realtime backend |
| `audio_io.py` | `Microphone` and `Speaker` (sounddevice) |
| `loudness_gate.py` | RMS gate that silences quiet room noise before it reaches the model |
| `session.py` | Bare runner for `--voice-only` (no bus/orchestrator) |
| `bridge.py` | `VoiceBridge` — the supervised session that ACTIVE actually runs |
| `two_agent.py` | `TwoAgentBridge` — coordinator of the filter + responder pair |
| `filter_agent.py` | `FilterAgent` — the listening filter (agent 1 of the pair) |
| `__init__.py` | `get_provider(name)` — lazy backend factory ("gemini" / "openai") so choosing one never imports the other's dependencies |

## `provider.py` — the provider abstraction

The whole point of this module is that swapping Gemini for OpenAI must not ripple into audio I/O,
RAG, or the orchestrator.

### `SessionConfig`

Provider-agnostic knobs for one live session:

- `system_instruction` — the persona text (from `prompts/system_instructions.md`).
- `model`, `voice` — resolved per provider (env-var overridable; see [Configuration](configuration.md)).
- `language` — BCP-47 hint; left `None` by default so native-audio models auto-detect (the
  reliable path for an English/Urdu mix).
- `input_rate` / `output_rate` — mic and playback sample rates. Gemini: 16 kHz in / 24 kHz out.
  OpenAI: 24 kHz both ways.
- `tools` — the function declarations advertised to the model (filled by the composition root).
- `vad_threshold`, `vad_silence_ms`, `noise_reduction` — server-VAD tuning; `None` keeps the
  provider default. Each backend maps these onto its own VAD config.

### Normalized events (`VoiceEvent` union)

`AudioOut(pcm)` (16-bit mono PCM at `output_rate`), `UserTranscript(text, final)`,
`AgentTranscript(text, final)`, `ToolCall(id, name, arguments)`, `TurnComplete`, `Interrupted`
(barge-in — stop playing), `ProviderError(message)`.

### The interfaces

`VoiceConnection` (a Protocol): `send_audio(pcm)`, `send_image(jpeg)`, `send_text(text)`,
`respond_tool(call, output)`, `interrupt()`, and `events()` (an async iterator of the normalized
events). `VoiceProvider`: a named factory whose `connect(config)` is an async context manager
yielding a `VoiceConnection`.

## `gemini.py` — Gemini Live backend

- `connect()` requires `GEMINI_API_KEY`, builds a `genai.Client` on the **v1beta** channel (the
  one that exposes Live/native-audio models), and opens `client.aio.live.connect(...)`.
- `_build_live_config()` maps `SessionConfig` onto `LiveConnectConfig`: audio-only responses,
  the system instruction, the prebuilt voice, input **and** output transcription enabled (so both
  sides of the conversation are visible as text), tools, and optional activity-detection tuning.
- `_activity_detection()` maps the VAD knobs best-effort: `vad_silence_ms` →
  `silence_duration_ms`; `vad_threshold >= 0.6` → LOW start/end sensitivity (fewer committed
  turns), else HIGH. It is wrapped in a try/except because the exact types move around between
  google-genai versions — if absent, Gemini keeps its defaults. Gemini has no noise-reduction
  knob, so that field is ignored here (it's an OpenAI lever; the loudness gate is the
  provider-agnostic equivalent).
- `GeminiConnection` starts a `_receive()` task that translates Gemini's message stream into the
  normalized events on an internal queue: `server_content.interrupted` → `Interrupted`,
  input/output transcriptions → `UserTranscript`/`AgentTranscript`, `msg.data` → `AudioOut`,
  function calls → `ToolCall`, `turn_complete` → `TurnComplete`. Any exception becomes a
  `ProviderError` rather than crashing the app; a `_DONE` sentinel ends the `events()` iterator.
- Sending: audio goes as `send_realtime_input(audio=Blob(...))` with the rate in the MIME type;
  text via `send_realtime_input(text=...)`; tool results via `send_tool_response`. `interrupt()`
  is a no-op — barge-in is automatic under Gemini's activity detection.

## `openai.py` — OpenAI Realtime backend

Mirrors the configuration proven in the `web-realtime/` browser sandbox, but over the WebSocket
path of the official SDK (a headless process can't use WebRTC): audio arrives as
`response.output_audio.delta` events and is sent with `input_audio_buffer.append` (base64).

- `_build_session()` produces the GA "realtime" session config: PCM at the configured rates,
  server VAD (threshold 0.5 / silence 500 ms unless `SessionConfig` overrides), input
  transcription via `gpt-4o-mini-transcribe` (model overridable with `OPENAI_TRANSCRIBE_MODEL`),
  optional `noise_reduction` (near_field / far_field), the tools, and `tool_choice: "auto"` when
  tools exist.
- Event translation in `_receive()`: audio deltas → `AudioOut`; agent transcript deltas →
  `AgentTranscript` (with an empty `final=True` marker when done); user transcription deltas →
  the prompt-echo filter below; `input_audio_buffer.speech_started` → `Interrupted`;
  `response.function_call_arguments.done` → `ToolCall`; `response.done` → `TurnComplete`; error
  events → `ProviderError`.

### The transcription prompt, and the prompt-echo filter

HARP wants transcripts in Latin script — English as-is and spoken Urdu *romanized* ("aap kaise
hain"), readable on the dashboard and consistent with the romanized wake words. Rather than
pinning a language (which would mis-transcribe whichever language the visitor switches to), the
input transcriber is steered with a prompt (`prompts/transcription_openai.md`, overridable with
the `OPENAI_TRANSCRIBE_PROMPT` env var).

The price: Whisper-family transcribers **regurgitate their priming prompt verbatim** when the
server VAD commits a turn on silence or breath noise — which would appear on the dashboard as if
the visitor had said it. The fix (`_still_prompt_prefix` + the `_on_user_delta`/
`_finish_user_turn` state machine): user-transcript deltas are *held back* while the accumulated
turn text is still a prefix of the prompt (compared on normalized text so a mid-word delta still
counts). The moment it diverges — real speech diverges within a word or two — the held text is
released and the rest streams normally. A turn that ends while still matching the prompt is
dropped entirely: it never happened.

## `audio_io.py` — microphone and speaker

Both are async context managers over sounddevice raw streams, deliberately vanilla.

- `Microphone(rate)` — `RawInputStream` (16-bit mono, 1024-frame chunks ≈ 64 ms at 16 kHz). The
  PortAudio callback runs on its own thread and hands bytes to the asyncio side with
  `call_soon_threadsafe` into a queue; `chunks()` yields them forever.
- `Speaker(rate)` — `RawOutputStream` plus a playback task: `play(pcm)` enqueues,
  the task writes via `asyncio.to_thread` (the raw write blocks until buffer room frees), and
  `clear()` drains anything queued-but-unplayed — called on barge-in so HARP shuts up immediately.

## `loudness_gate.py` — the proximity/noise gate

`LoudnessGate(level)` takes a *callable* returning the current RMS threshold (0..1), so the
dashboard slider takes effect on the very next chunk; `level() <= 0` disables the gate.
`process(pcm)`:

- at/above threshold → pass, refresh the hangover counter, and if there is buffered pre-roll,
  prepend it (so the onset of a word that *crossed* the threshold isn't clipped);
- below threshold but within the hangover (8 chunks) → still pass (a brief dip mid-word must not
  chop the turn);
- otherwise → buffer the chunk as potential pre-roll (3 chunks) and emit same-length **silence**
  in its place.

Emitting silence rather than nothing keeps the stream continuous, so the provider VAD still sees
trailing quiet and ends turns properly — the same trick push-to-talk uses. The gate is shared by
whichever agent owns a real mic: the single-agent bridge and the two-agent filter.

## `session.py` — the bare runner (`--voice-only`)

`run(provider_name, config, tool_dispatch)` opens provider + mic + speaker, pumps the mic up and
events down, and prints streaming transcripts to the terminal (`_Printer` handles the "You:" /
"HARP:" prefixes). Tool calls are executed through the injected dispatcher with the same
degrade-to-`{"error": ...}` contract as the bridge. This is what `python -m harp --voice-only`
runs — no bus, orchestrator, dashboard, or vision — and it exists as a fast smoke test of the
provider/audio path. Provider, mic, and speaker are injectable for tests.

## `bridge.py` — `VoiceBridge`, what ACTIVE actually runs

The supervised version of the runner: the orchestrator starts `VoiceBridge.run(context)` as a
task when a session opens and cancels it when the session closes. Everything composition-shaped
is injected by `app.py`: which provider, a `make_config` factory (called fresh at every open so
current dashboard tuning is stamped on), the tool dispatcher, the identity-context callable, the
push-to-talk gate, the live loudness threshold, and (for tests) fake providers/mics/speakers.

`run(context)`:

1. connects provider, speaker, and — unless in text-driven mode — the mic, under one
   `AsyncExitStack`;
2. sends the opening text: `context` (the wake explanation from `WakeRequested`) plus the
   identity line, joined;
3. starts the input pump (`_pump_mic` normally, `_pump_text` in two-agent mode) and consumes
   events until the provider's stream ends.

Details worth knowing:

- **Two gates in series** on the mic: first the hard push-to-talk gate (`gated_mic_payload` —
  real audio while the gate is open, else same-length digital silence), then the `LoudnessGate`.
  `gated_mic_payload` is a module-level function shared with the filter agent and the offline
  recorder spike, so all three produce byte-identical streams.
- **Event translation** (`_pump_events`): `AudioOut` → speaker; `AgentTranscript`/`UserTranscript`
  → `AgentSaid`/`UserSaid` on the bus; `Interrupted` → `speaker.clear()`; `ToolCall` →
  `_handle_tool`; `ProviderError` → `ErrorRaised(where="voice.provider")` (which triggers the
  orchestrator's error/backoff path). `TurnComplete` needs no bus mirror — the final transcript
  pieces already close turns for consumers.
- **Tool handling** (`_handle_tool`): publish `ToolRequested`, run the dispatcher (an exception
  degrades to `{"error": str(exc)}` — a failing tool becomes the model apologizing, not a dead
  session), publish `ToolCompleted`, return the result via `respond_tool`.
- **Text-driven mode**: when a `text_inbox` queue is injected (only by `TwoAgentBridge`), the
  session opens *no microphone* and instead forwards each queued message as a user text turn.
  Everything else — tools, transcripts, speaker, barge-in — is unchanged.

## `two_agent.py` + `filter_agent.py` — the two-agent noise filter

Opt-in via `filter_agent.enabled` in `harp.yaml`; built *instead of* the plain bridge by
`app.py`. Exposes the same `run(context)` interface, so the orchestrator cannot tell the
difference.

```
mic ─audio─▶ FilterAgent ─clean text─▶ VoiceBridge (responder) ─▶ speaker
```

### `FilterAgent` (agent 1)

Opens its own live session whose only job is deciding what, if anything, the visitor said *to
HARP*, and relaying that as clean text. Key mechanics:

- It hears the room through the mic like a normal session but has **no speaker** — its spoken
  output is discarded. Instead, its `AgentTranscript` (the transcript of what it "says") *is* the
  relay: the persona (`prompts/filter_instructions.md`) instructs it to output either the
  intended message or the sentinel `[[ignore]]` for noise/background/HARP's own voice.
- `clean_relay(text)` strips the sentinel (matched case/spacing-insensitively, so
  "[[ IGNORE ]]." can't leak) and returns `""` when nothing alphanumeric remains — an
  all-sentinel turn relays nothing.
- `add_context(text)` queues a `CONTEXT: ...` note (the responder's last reply) into the filter's
  session so short follow-ups ("yes", "how much?") remain interpretable; the persona absorbs
  CONTEXT lines and answers `[[ignore]]`.
- Its mic goes through the same two gates as the bridge: the injected half-duplex gate first,
  then the loudness gate.
- Debuggability: what its ASR *heard* and what it chose to relay or ignore are both logged at
  INFO, so a bogus relay can be localized to either the committed audio or the model inventing a
  turn.
- Its errors publish `ErrorRaised(where="voice.filter")`.

### `TwoAgentBridge` (the coordinator)

- Builds a `FilterAgent` and a text-driven `VoiceBridge` (responder) per session, connected by
  two wires: **relay** — each approved message is published as a final `UserSaid` (so the
  dashboard shows the filtered user turn; the responder receives text, so it emits no user
  transcript of its own) and pushed onto the responder's text inbox; **feedback** — a bus watcher
  accumulates the responder's `AgentSaid` stream and, on each final, feeds the finished reply
  back into the filter as context.
- **Half-duplex gate**: while the responder is speaking (`_speaking`, tracked from the AgentSaid
  stream) or within `response_tail_seconds` after its reply finalizes (covering audio still
  draining from the speaker), `_filter_mic_gate()` returns False and the filter's mic feeds
  silence — HARP never relays its own voice to itself. The cost: a visitor cannot interrupt HARP
  mid-reply. A `_RELAY_GRACE_SECONDS` (12 s) cap mutes the mic right after a relay but guarantees
  it reopens even if the responder never answers.
- The filter deliberately gets **no wake context** — that text is written for the responder
  ("someone said hello — greet them") and would prime the filter to hallucinate greetings on
  silence. The responder gets the full context.
- Both sessions run as tasks with first-exit-wins semantics (same shape as `app.py`): whichever
  ends first — provider closed the stream, or a crash — ends the whole interaction, and a crash
  re-raises so the orchestrator's error path runs.
