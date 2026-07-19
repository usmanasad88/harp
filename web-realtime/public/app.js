'use strict';

/* HARP Realtime — browser client.
 *
 * Browser responsibilities (everything in this file):
 *   - capture the mic, play HARP's audio,
 *   - fetch a short-lived ephemeral key from our /session endpoint,
 *   - open the WebRTC peer connection directly to OpenAI,
 *   - render transcript + state, and recover from drops.
 * It never sees the real OPENAI_API_KEY — only the ephemeral ek_... secret.
 */

const OPENAI_CALLS_URL = 'https://api.openai.com/v1/realtime/calls';
const MAX_RECONNECTS = 4;

// ---- DOM ------------------------------------------------------------------
const body = document.body;
const callBtn = document.getElementById('callBtn');
const callText = document.getElementById('callText');
const statusText = document.getElementById('statusText');
const turnLabel = document.getElementById('turnLabel');
const hint = document.getElementById('hint');
const transcriptEl = document.getElementById('transcript');
const clearBtn = document.getElementById('clearBtn');
const harpAudio = document.getElementById('harpAudio');
const modelTag = document.getElementById('modelTag');
const voiceTag = document.getElementById('voiceTag');
const latencyEl = document.getElementById('latency');
const sessionTimer = document.getElementById('sessionTimer');
const vadTag = document.getElementById('vadTag');
const orb = document.getElementById('orb');

// ---- Session state --------------------------------------------------------
let pc = null;
let dc = null;
let micStream = null;
let audioCtx = null;
let micAnalyser = null;
let outAnalyser = null;
let intentionalEnd = false;
let reconnects = 0;
let reconnectTimer = null;
let sessionStart = 0;
let sessionTick = null;
let speechStoppedAt = 0;
let userLine = null;
let harpLine = null;

const TURN_LABELS = {
  idle: 'tap to begin',
  connecting: 'connecting…',
  reconnecting: 'reconnecting…',
  listening: 'listening…',
  user_speaking: 'go on…',
  harp_speaking: 'HARP speaking',
  error: 'tap to retry',
};

// ---- UI state helpers -----------------------------------------------------
function setState(state) {
  body.dataset.state = state;
  const labels = {
    idle: 'offline',
    connecting: 'connecting',
    live: 'live',
    reconnecting: 'reconnecting',
    error: 'error',
  };
  statusText.textContent = labels[state] || state;

  if (state === 'live' || state === 'reconnecting') {
    callText.textContent = 'End session';
  } else if (state === 'connecting') {
    callText.textContent = 'Connecting…';
  } else {
    callText.textContent = 'Begin session';
  }

  if (state === 'idle') setTurn('idle');
  if (state === 'connecting') setTurn('connecting');
  if (state === 'reconnecting') setTurn('reconnecting');
}

function setTurn(turn) {
  body.dataset.turn = turn;
  turnLabel.textContent = TURN_LABELS[turn] || turn;
  if (turn === 'user_speaking') vadTag.textContent = 'you';
  else if (turn === 'harp_speaking') vadTag.textContent = 'harp';
  else if (turn === 'listening') vadTag.textContent = 'open';
  else vadTag.textContent = 'idle';
}

function addLine(who, text, cls) {
  const line = document.createElement('div');
  line.className = 'line ' + (cls || who);
  if (who) {
    const w = document.createElement('span');
    w.className = 'who';
    w.textContent = who;
    line.appendChild(w);
  }
  const t = document.createElement('span');
  t.className = 'text';
  t.textContent = text || '';
  line.appendChild(t);
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return line;
}

function setHint(text, isError) {
  hint.textContent = text;
}

// ---- Connection lifecycle -------------------------------------------------
async function startSession() {
  intentionalEnd = false;
  setState('connecting');
  setHint('Allow microphone access to talk with HARP.');

  // 1. Microphone (browser responsibility). Echo cancellation keeps HARP's
  //    voice from looping back into the mic.
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    setState('error');
    if (err && err.name === 'NotAllowedError') {
      setHint('Microphone blocked. Enable it in your browser’s site settings, then retry.');
    } else {
      setHint('Could not access a microphone: ' + (err && err.message ? err.message : err));
    }
    return;
  }

  // 2. Ephemeral key from our server (server holds the real API key).
  let token;
  try {
    const res = await fetch('/session', { method: 'POST' });
    const data = await res.json();
    if (!res.ok || !data.value) throw new Error(data.error || 'No ephemeral key returned.');
    token = data.value;
    modelTag.textContent = data.model || 'realtime';
    voiceTag.textContent = data.voice || '—';
  } catch (err) {
    cleanupMedia();
    setState('error');
    setHint('Could not start a session: ' + (err.message || err));
    return;
  }

  // 3. Peer connection + audio plumbing.
  try {
    await openPeerConnection(token);
  } catch (err) {
    setState('error');
    setHint('Connection failed: ' + (err.message || err));
    teardown(false);
  }
}

async function openPeerConnection(token) {
  pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
  });

  // HARP's audio arrives on a remote track -> play it and analyse it.
  pc.ontrack = (e) => {
    harpAudio.srcObject = e.streams[0];
    attachOutputAnalyser(e.streams[0]);
  };

  // Send our mic.
  micStream.getTracks().forEach((track) => pc.addTrack(track, micStream));
  attachMicAnalyser(micStream);

  // Events + transcripts ride this data channel.
  dc = pc.createDataChannel('oai-events');
  dc.onopen = onChannelOpen;
  dc.onmessage = (e) => {
    try {
      handleServerEvent(JSON.parse(e.data));
    } catch (_) {
      /* ignore non-JSON */
    }
  };

  // Recovery: react to the connection dropping.
  pc.onconnectionstatechange = () => {
    const s = pc && pc.connectionState;
    if (s === 'failed' || s === 'disconnected') {
      if (!intentionalEnd) scheduleReconnect();
    }
  };

  // SDP offer/answer with OpenAI, authenticated by the ephemeral key.
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const sdpRes = await fetch(OPENAI_CALLS_URL, {
    method: 'POST',
    body: offer.sdp,
    headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/sdp' },
  });
  if (!sdpRes.ok) throw new Error('SDP exchange failed (' + sdpRes.status + ')');
  const answer = { type: 'answer', sdp: await sdpRes.text() };
  await pc.setRemoteDescription(answer);
}

function onChannelOpen() {
  reconnects = 0;
  setState('live');
  setTurn('listening');
  setHint('Connected. Just start talking — English or Urdu.');
  if (!sessionStart) startSessionTimer();
  addLine('', 'Session connected with ' + (modelTag.textContent || 'realtime') + '.', 'system');
}

function scheduleReconnect() {
  if (intentionalEnd || reconnectTimer) return;
  if (reconnects >= MAX_RECONNECTS) {
    setState('error');
    setHint('Lost the connection and couldn’t recover. Tap to try again.');
    teardown(false);
    return;
  }
  reconnects += 1;
  const delay = Math.min(1000 * Math.pow(2, reconnects - 1), 8000);
  setState('reconnecting');
  setHint('Connection dropped — reconnecting (attempt ' + reconnects + ')…');
  // Tear down the dead peer connection but keep the mic + timer.
  closePeer();
  reconnectTimer = setTimeout(async () => {
    reconnectTimer = null;
    try {
      const res = await fetch('/session', { method: 'POST' });
      const data = await res.json();
      if (!res.ok || !data.value) throw new Error(data.error || 'token');
      await openPeerConnection(data.value);
    } catch (_) {
      scheduleReconnect();
    }
  }, delay);
}

function endSession() {
  intentionalEnd = true;
  addLine('', 'Session ended.', 'system');
  teardown(true);
  setState('idle');
  setHint('Speak in English or Urdu — HARP listens and replies out loud.');
}

// ---- Server events --------------------------------------------------------
function handleServerEvent(evt) {
  const type = evt.type || '';

  switch (type) {
    case 'input_audio_buffer.speech_started':
      setTurn('user_speaking');
      if (!userLine) userLine = addLine('you', '', 'you pending');
      break;

    case 'input_audio_buffer.speech_stopped':
      speechStoppedAt = performance.now();
      break;

    case 'conversation.item.input_audio_transcription.delta':
      if (!userLine) userLine = addLine('you', '', 'you pending');
      appendText(userLine, evt.delta || '');
      break;

    case 'conversation.item.input_audio_transcription.completed':
      if (!userLine) userLine = addLine('you', '', 'you pending');
      setText(userLine, (evt.transcript || '').trim() || '(unclear)');
      userLine.classList.remove('pending');
      userLine = null;
      break;

    // HARP's spoken reply, transcribed as it streams.
    case 'response.output_audio_transcript.delta':
    case 'response.audio_transcript.delta':
      if (speechStoppedAt) {
        latencyEl.textContent = Math.round(performance.now() - speechStoppedAt) + ' ms';
        speechStoppedAt = 0;
      }
      setTurn('harp_speaking');
      if (!harpLine) harpLine = addLine('harp', '', 'harp pending');
      appendText(harpLine, evt.delta || '');
      break;

    case 'response.output_audio_transcript.done':
    case 'response.audio_transcript.done':
      if (harpLine) {
        if (evt.transcript) setText(harpLine, evt.transcript.trim());
        harpLine.classList.remove('pending');
        harpLine = null;
      }
      break;

    // The model wants to look something up. Arguments have finished streaming.
    case 'response.function_call_arguments.done':
      handleFunctionCall(evt.name, evt.call_id, evt.arguments);
      break;

    case 'response.done':
      setTurn('listening');
      break;

    case 'error':
      {
        const msg = (evt.error && (evt.error.message || evt.error.code)) || 'unknown error';
        addLine('', 'Error: ' + msg, 'system');
      }
      break;
  }
}

// Relay a model function call to our server, then hand the result back so HARP
// can speak an answer grounded in data/.
async function sendEvent(obj) {
  if (dc && dc.readyState === 'open') dc.send(JSON.stringify(obj));
}

async function handleFunctionCall(name, callId, argsStr) {
  let output;
  try {
    if (name !== 'search_knowledge') throw new Error('unknown tool: ' + name);
    const args = JSON.parse(argsStr || '{}');
    const query = (args.query || '').trim();
    vadTag.textContent = 'search';
    turnLabel.textContent = 'searching…';
    addLine('', '🔎 searched: “' + query + '”', 'system');

    const res = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    output = JSON.stringify(data.results && data.results.length ? data.results : { note: 'no matches found' });
  } catch (err) {
    output = JSON.stringify({ error: String(err.message || err) });
  }

  // Return the tool result and ask the model to continue speaking.
  await sendEvent({
    type: 'conversation.item.create',
    item: { type: 'function_call_output', call_id: callId, output: output },
  });
  await sendEvent({ type: 'response.create' });
}

function appendText(line, chunk) {
  const t = line.querySelector('.text');
  t.textContent += chunk;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}
function setText(line, text) {
  line.querySelector('.text').textContent = text;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// ---- Teardown -------------------------------------------------------------
function closePeer() {
  if (dc) {
    try {
      dc.close();
    } catch (_) {}
    dc = null;
  }
  if (pc) {
    pc.onconnectionstatechange = null;
    pc.ontrack = null;
    try {
      pc.close();
    } catch (_) {}
    pc = null;
  }
}

function cleanupMedia() {
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  micAnalyser = null;
  outAnalyser = null;
}

function teardown(full) {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  closePeer();
  cleanupMedia();
  harpAudio.srcObject = null;
  userLine = null;
  harpLine = null;
  setTurn('idle');
  if (full) {
    stopSessionTimer();
    reconnects = 0;
  }
}

// ---- Telemetry: session timer --------------------------------------------
function startSessionTimer() {
  sessionStart = Date.now();
  sessionTick = setInterval(() => {
    const s = Math.floor((Date.now() - sessionStart) / 1000);
    sessionTimer.textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
  }, 1000);
}
function stopSessionTimer() {
  if (sessionTick) clearInterval(sessionTick);
  sessionTick = null;
  sessionStart = 0;
  sessionTimer.textContent = '0:00';
  latencyEl.textContent = '—';
}

// ---- Audio-reactive sonar orb --------------------------------------------
function ensureAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}
function makeAnalyser(stream) {
  const ctx = ensureAudioCtx();
  const src = ctx.createMediaStreamSource(stream);
  const a = ctx.createAnalyser();
  a.fftSize = 512;
  a.smoothingTimeConstant = 0.8;
  src.connect(a); // analysis only — not routed to speakers, so no echo.
  return a;
}
function attachMicAnalyser(stream) {
  micAnalyser = makeAnalyser(stream);
}
function attachOutputAnalyser(stream) {
  outAnalyser = makeAnalyser(stream);
}

function level(analyser) {
  if (!analyser) return 0;
  const buf = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) {
    const v = (buf[i] - 128) / 128;
    sum += v * v;
  }
  return Math.min(1, Math.sqrt(sum / buf.length) * 3.2);
}

// Echolocation ripples emitted on loud moments.
const ripples = [];
function renderOrb() {
  requestAnimationFrame(renderOrb);
  const ctx = orb.getContext('2d');
  const W = orb.width;
  const H = orb.height;
  const cx = W / 2;
  const cy = H / 2;
  ctx.clearRect(0, 0, W, H);

  const live = body.dataset.state === 'live';
  const inLvl = live ? level(micAnalyser) : 0;
  const outLvl = live ? level(outAnalyser) : 0;
  const speaking = outLvl > inLvl;
  const energy = Math.max(inLvl, outLvl);

  // Aqua when HARP speaks, coral when you speak, dim aqua at rest.
  const hue = speaking ? '70, 232, 210' : energy > 0.04 ? '255, 122, 107' : '70, 232, 210';
  const baseR = W * 0.16;
  const r = baseR * (1 + energy * 0.9);

  // Idle breathing if connected but silent.
  const t = performance.now() / 1000;
  const breathe = live ? 1 + Math.sin(t * 1.6) * 0.03 : 1 + Math.sin(t * 1.0) * 0.015;

  // Spawn ripples on speech energy.
  if (live && energy > 0.18 && Math.random() < 0.4) {
    ripples.push({ r: r, a: 0.5, hue: hue });
  }
  for (let i = ripples.length - 1; i >= 0; i--) {
    const rp = ripples[i];
    rp.r += 3.2;
    rp.a -= 0.012;
    if (rp.a <= 0) {
      ripples.splice(i, 1);
      continue;
    }
    ctx.beginPath();
    ctx.arc(cx, cy, rp.r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(' + rp.hue + ',' + rp.a + ')';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Static guide rings.
  for (let i = 1; i <= 3; i++) {
    ctx.beginPath();
    ctx.arc(cx, cy, baseR + i * baseR * 0.55 * breathe, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(' + hue + ',' + (0.10 - i * 0.02) + ')';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Glowing core.
  const grad = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r * breathe);
  grad.addColorStop(0, 'rgba(' + hue + ',' + (0.85 - (speaking ? 0 : 0.2)) + ')');
  grad.addColorStop(0.6, 'rgba(' + hue + ',0.25)');
  grad.addColorStop(1, 'rgba(' + hue + ',0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, r * 1.8 * breathe, 0, Math.PI * 2);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(cx, cy, r * breathe, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(' + hue + ',0.9)';
  ctx.fill();
}
renderOrb();

// ---- Wiring ---------------------------------------------------------------
callBtn.addEventListener('click', () => {
  const s = body.dataset.state;
  if (s === 'live' || s === 'reconnecting' || s === 'connecting') endSession();
  else startSession();
});
clearBtn.addEventListener('click', () => {
  transcriptEl.innerHTML = '';
  userLine = null;
  harpLine = null;
});

setState('idle');
