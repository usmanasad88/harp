# HARP Realtime

A small, polished **real-time voice console** built on the **OpenAI Realtime API**
over **WebRTC** — low-latency speech in, speech out, with a live transcript and a
sonar-orb that reacts to who's talking. It's the browser-side spike for HARP's
OpenAI Realtime provider, themed around **HARP**, the robot receptionist.

This is *speech-to-speech over a persistent media connection*, not a
record → upload → transcribe → reply text loop. Audio streams continuously in
both directions; the model decides when you've stopped talking (server-side voice
activity detection) and barges in with a spoken reply.

---

## Client vs. server: who does what

The whole point of this split is that **your real OpenAI key never touches the
browser**.

| Server (`server.js`, Node, no deps)                | Browser (`public/`, vanilla JS)                       |
| -------------------------------------------------- | ----------------------------------------------------- |
| Holds `OPENAI_API_KEY`.                            | Never sees the real key.                              |
| Builds the session config (model, voice, persona). | Captures mic, plays HARP's audio.                    |
| `POST /v1/realtime/client_secrets` → ephemeral key.| `POST /session` to get that short-lived `ek_…` key.   |
| Returns only `{ value, expires_at, model, voice }`.| Opens the WebRTC peer connection straight to OpenAI.  |
| Serves the static front-end.                       | Renders transcript, state, telemetry, recovery.       |

Flow:

```
browser ── POST /session ──► your server ── POST /v1/realtime/client_secrets ──► OpenAI
browser ◄──── ek_… key ─────┘                          (real API key here only)
browser ── SDP offer (Bearer ek_…) ──► https://api.openai.com/v1/realtime/calls
browser ◄──── SDP answer ────────────┘   then audio + events flow peer-to-peer
```

---

## Setup

**Prerequisites:** Node 12 or newer (no `npm install` needed — the server uses
only built-in modules), and a modern browser (Chrome/Edge/Safari).

1. **API key.** The server reuses the existing key in the HARP root env file. Make
   sure `harp/.env` contains a real key:

   ```
   OPENAI_API_KEY=sk-...
   ```

   (You can also `export OPENAI_API_KEY=...` in your shell, or drop a local
   `web-realtime/.env` — the shell env wins, then a local `.env`, then `../.env`.)

2. **Run it.**

   ```bash
   cd harp/web-realtime
   npm start          # == node server.js
   ```

3. **Open** http://localhost:3000 and click **Begin session**. Allow the mic when
   prompted, then talk.

> Mic capture needs a "secure context." `http://localhost` counts as secure, so
> local dev works with no TLS. If you serve this on a real host, it must be
> **HTTPS** or the browser will silently refuse the microphone.

### Configuration (optional)

Set these in `harp/.env` (or the shell) to override the defaults:

| Var              | Default            | Notes                                                        |
| ---------------- | ------------------ | ------------------------------------------------------------ |
| `REALTIME_MODEL` | `gpt-realtime-2`   | Reasoning realtime model. `gpt-realtime-1.5` = best plain audio-in/out; `gpt-realtime-mini` = cheaper. |
| `REALTIME_VOICE` | `marin`            | Try `cedar`, `alloy`, `shimmer`, `echo`.                     |
| `PORT`           | `3000`             | Local dev port.                                              |

The spoken persona is loaded from `../prompts/system_instructions.md` (the HARP
prompt). Edit that file to change how HARP behaves — it's the single source of
truth, applied server-side so the browser can't tamper with it.

---

## Knowledge (RAG)

HARP can look things up in `harp/data/*.md` before answering, via a
`search_knowledge` **function tool**. Nothing is hardcoded to a specific corpus —
drop your own markdown into `data/` and restart.

How the round-trip works:

```
model decides to look something up
   └─ emits  response.function_call_arguments.done  { name, call_id, arguments }
browser relays the query ─► POST /search ─► server BM25 search over data/*.md
browser returns result  ─► conversation.item.create { function_call_output } ─► response.create
   └─ HARP speaks an answer grounded in the top passages
```

- **Backend:** `knowledge.js` — dependency-free **keyword (BM25)** search over
  markdown, chunked by heading. Indexed once at server startup.
- **Server-side by design:** the browser only relays the query; the index stays on
  the server. This keeps the client/server split intact and lets you later put an
  embedding key here without touching the browser.
- **Adding knowledge:** drop `.md` files into `harp/data/` and restart the server.
  The startup log prints how many chunks were indexed.
- **Known limits (keyword search):** it matches words, not meaning — "where is it
  located" won't find "Islamabad" unless the word overlaps. Images (`.jpeg` floor
  maps) and the visitor-pass PDF are **not** indexed; those need the vision phase.
- **Upgrading to semantic search:** rewrite only `knowledge.js`'s `search()` to use
  embeddings (e.g. `text-embedding-3-small`) — the tool interface and everything
  else stay the same.

## Developer notes

### Latency
- **WebRTC, not WebSockets**, is the recommended browser transport: audio rides
  the same low-jitter path as a video call, and turn-taking feels conversational.
- Keep `getUserMedia` constraints with `echoCancellation: true` — without it,
  HARP hears herself through your speakers and talks over her own tail.
- The **reply** figure in the footer measures wall-clock from *you stop speaking*
  (`input_audio_buffer.speech_stopped`) to *HARP's first transcript token*. Treat
  it as a felt-latency gauge, not a benchmark; it includes server VAD end-pointing.
- Server-side VAD (`silence_duration_ms: 500`) trades responsiveness against
  cutting you off. Lower it for snappier turns, raise it if HARP interrupts.

### Session lifecycle
- The **ephemeral key expires in ~10 min** (`expires_after.seconds: 600`) and is
  only needed to *open* the connection. Once connected, the live session continues
  on its own; the key expiring mid-call is harmless.
- Realtime **sessions have a maximum duration** server-side. When one ends, the
  peer connection drops — the client's recovery logic mints a fresh key and
  reconnects automatically.
- Each reconnect requests a **new** ephemeral key. Never cache or reuse one.

### Permissions
- First click triggers the mic permission prompt. A denied mic surfaces as a clear
  coral "Microphone blocked…" hint, not a silent failure.
- Audio playback and the `AudioContext` are started **after** the user clicks, to
  satisfy browser autoplay rules. Don't move connection setup to page load.

### Error recovery
- `RTCPeerConnection.onconnectionstatechange` watches for `failed`/`disconnected`.
  On a drop (and only if the user didn't end the call), the client reconnects with
  **exponential backoff** (1s → 2s → 4s → 8s, up to 4 tries) behind a
  "reconnecting…" state, keeping the mic open the whole time.
- A failed `/session` (e.g. missing/invalid key) returns a JSON `error` the UI
  shows verbatim — check the server console too; it prints whether the key loaded.
- Server events of `type: "error"` are appended to the transcript so model-side
  problems are visible during testing.

---

## Validation checklist

Run through this to confirm a build is healthy.

**Audio permissions**
- [ ] First **Begin session** click shows the browser mic prompt.
- [ ] **Allow** → state goes `connecting` → `live`; orb starts reacting to your voice (coral tint).
- [ ] **Block** the mic → coral hint explains how to re-enable; no console crash.
- [ ] Re-allow in site settings, retry → connects normally.
- [ ] On a non-localhost/HTTP origin, confirm the mic is refused (expected — needs HTTPS).

**Connection recovery**
- [ ] Mid-conversation, toggle Wi-Fi/airplane mode off for a few seconds → state shows `reconnecting…`, backoff attempts count up.
- [ ] Restore network → it reconnects and you can keep talking without reloading.
- [ ] Kill the network long enough to exhaust retries → lands in `error` with a clear message and a working **Begin session** retry.
- [ ] Stop the Node server, click Begin → friendly "Could not start a session" error (not a blank page).
- [ ] Empty/invalid `OPENAI_API_KEY` → server console shows `Key: MISSING`; `/session` returns a readable error in the UI.

**Basic conversation quality**
- [ ] Say "Hello" → HARP replies out loud within a second or two; **reply** latency populates.
- [ ] Your words appear under **you**; HARP's reply streams under **harp** as she speaks.
- [ ] **Interrupt** HARP mid-sentence by talking → she yields (barge-in works).
- [ ] Speak Urdu → HARP replies in Urdu (the HARP language test); try a natural Urdu/English mix.
- [ ] No echo/feedback loop (confirms echo cancellation is on).
- [ ] **End session** stops the mic (OS mic indicator turns off) and resets state to `offline`.
- [ ] Session timer counts up while live and resets on end.

**Knowledge / retrieval**
- [ ] Ask something answerable from `data/` (e.g. "what is this expo about?") → a `🔎 searched:` line appears and HARP answers from the docs.
- [ ] Ask in Urdu about the expo → she still searches with English keywords and answers in Urdu.
- [ ] Ask something not in the docs → she says she's not sure rather than inventing an answer.
- [ ] Drop a new `.md` into `harp/data/`, restart, and confirm the startup log's chunk count rises and the new content is findable.

---

## Files

```
web-realtime/
├── server.js            token-minting + static server + /search (Node built-ins only)
├── knowledge.js         BM25 keyword search over ../data/*.md (RAG backend)
├── package.json         `npm start`
├── public/
│   ├── index.html       layout
│   ├── styles.css       deep-ocean theme + state styling
│   └── app.js           WebRTC, data-channel events, transcript, orb, recovery, tool relay
└── README.md
```
