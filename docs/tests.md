# Tests (`tests/`)

[← Back to index](index.md)

Run with `uv run pytest`. ~290 test functions across 36 files, one file per module under test.
No test needs hardware, a network, an API key, a display, or a sound card — that is a direct
payoff of the architecture's injection seams.

## How hardware/cloud code gets tested

The recurring patterns:

- **Bus-driven testing**: a subsystem is constructed with a real `Bus`, fake events are
  published in, and the events it publishes out are asserted. Works because subsystems depend
  only on the bus (e.g. `test_orchestrator.py`, `test_end_rules.py`, `test_triggers.py`).
- **Injected fakes for the edges**: providers, microphones, speakers, serial factories, key
  listeners, camera backends, the Gemini helper's `caller`, and the status-voice audio sink are
  all constructor parameters. Tests pass fakes; production uses the defaults
  (`test_bridge.py`, `test_two_agent.py`, `test_filter_agent.py`, `test_motion.py`,
  `test_push_to_talk.py`, `test_status_voice.py`, `test_memory_agent.py`).
- **Pure logic extracted on purpose**: the wake detector (sample-count timing → deterministic),
  `match_wake_word`, the BM25 retriever, the retry policy, `follow.steer()` (the sign
  conventions the docstring calls "exactly the kind of silent bug a test should pin down"),
  the transcript digest, and the OpenAI prompt-echo filter (`test_transcript_echo.py`) are all
  plain functions tested with plain values.
- **Injected clocks** where time matters (`RateLimiter`, `ContextWriter`, push-to-talk
  debounce), so no test sleeps its way to a timeout.
- **Real filesystem in tmp dirs** for the store, session log, interaction logger, and
  summarizer — the crash-recovery behaviors (`.part` rescue, pending sweeps, atomic writes) are
  exercised against real files.
- **The dashboard server** is tested by binding an ephemeral port (`_build_server(port=0)`) and
  speaking real websockets/HTTP to it (`test_dashboard.py`).

## Per-area coverage sketch

| Area | Files | Focus |
|---|---|---|
| Core | `test_bus.py`, `test_session_log.py`, `test_retry.py` | fan-out, type filtering, drop-oldest backpressure; log format + pruning + fail-safety; backoff/give-up policy |
| Config | `test_config.py` | settings defaults/unknown keys, prompt loading + stripping + fallbacks, tuning clamps, camera-source mapping |
| Voice | `test_bridge.py`, `test_session.py`, `test_two_agent.py`, `test_filter_agent.py`, `test_transcript_echo.py` | event → bus translation, tool dispatch degradation, mic gating (silence substitution), the half-duplex gate, sentinel stripping, the prompt-echo holdback |
| Listener | `test_listener.py`, `test_transcriber.py` | wake decisions with synthetic audio, standby pause/resume, wake-word matching |
| Orchestrator | `test_orchestrator.py` (25 tests) | the full state machine: boot, wake policy/veto, session open/close per cause, error backoff and give-up, heartbeat |
| Interaction | `test_end_rules.py`, `test_push_to_talk.py`, `test_idle_prompt.py`, `test_session_tools.py` | both end rules incl. presence seeding, debounce/tap-train semantics, exclusive gating |
| Vision | `test_camera.py`, `test_frames.py`, `test_gestures.py`, `test_face_id.py` | backend selection/switching, overlay drawing, debounce, publish-on-change |
| Memory | `test_store.py`, `test_matcher.py`, `test_memory_agent.py`, `test_interaction_logger.py`, `test_memory_parse.py`, `test_summarizer.py`, `test_context_writer.py`, `test_memory_tools.py` | the whole pipeline from transcript to stored memory, rate limiting, TTL/cache-key behavior |
| Knowledge | `test_knowledge.py` | chunking, BM25 ranking, tool declarations/dispatch, web-search parsing |
| Motion | `test_motion.py`, `test_move_around.py`, `test_follow.py` | deadman behavior, patrol stop semantics, steer() signs/hysteresis, controller conflicts |
| Dashboard | `test_dashboard.py` | static serving, event forwarding, all five command handlers, seeding |

## The project's test philosophy

From the project's memory/conventions: only *meaningful* tests are wanted — timing/async
behavior, silent-bug pinning (sign conventions, gating, echo suppression), and regressions.
Tests that merely restate the implementation are deliberately absent.
