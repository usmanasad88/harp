// HARP end-user (kiosk) page. Same /ws event stream as the developer
// dashboard, rendered as ONE full-screen state instead of panels:
//
//   offline    — not connected to HARP (or HARP down): "Connecting…"
//   idle       — no one talking: "Hold the green button to talk"
//   listening  — the talk key is held: whole screen green, "Listening"
//   thinking   — key released, reply on its way: pulsing dots
//   responding — the agent's reply, streamed in as it speaks
//
// Priority is exactly that order (a held key always wins; any reply text
// beats the dots). The server seeds a fresh connection with the current app
// state + talk-key hold, so a page opened (or reconnected) mid-conversation
// starts right instead of guessing.

// How long a finished reply stays on screen. The transcript's final piece can
// arrive well before the audio finishes playing (text streams faster than
// speech), so linger for the reply's estimated remaining SPEAKING time —
// ~65 ms per character from the moment the turn started — never less than a
// beat, never forever.
const SPEAK_MS_PER_CHAR = 65;
const LINGER_MIN_MS = 3000;
const LINGER_MAX_MS = 30000;
// Released the key but no reply started → give up on the dots, re-prompt.
const THINKING_TIMEOUT_MS = 12000;
// A reply that stalls mid-stream with no final piece (lost turn, interrupt
// race) must not stick on a kiosk screen forever.
const RESPONSE_STALL_MS = 10000;

let ws = null;
let reconnectDelay = 1000;

let connected = false;
let held = false;          // TalkKeyChanged — is the talk key down
let appActive = false;     // StateChanged — is a live session open
let waiting = false;       // released the key, expecting a reply
let response = { text: "", final: false };
let turnStartedAt = 0;     // when the current reply's first piece arrived
let lastPieceAt = 0;       // when its latest piece arrived

let lingerTimer = null;
let thinkingTimer = null;

// --- rendering ---------------------------------------------------------------

const responseEl = document.getElementById("response");
const respondingScreen = document.getElementById("screen-responding");

function sizeClass(len) {
  if (len <= 90) return "len-s";
  if (len <= 280) return "len-m";
  return "len-l";
}

function render() {
  let mode;
  if (!connected) mode = "offline";
  else if (held) mode = "listening";
  else if (response.text) mode = "responding";
  else if (waiting) mode = "thinking";
  else mode = "idle";

  document.body.className = "mode-" + mode;
  if (mode === "responding") {
    responseEl.textContent = response.text;
    respondingScreen.className = "screen " + sizeClass(response.text.length);
    responseEl.scrollTop = responseEl.scrollHeight; // newest words in view
  }
}

// --- state changes -------------------------------------------------------

function clearResponse() {
  response = { text: "", final: false };
  clearTimeout(lingerTimer);
  lingerTimer = null;
}

function stopWaiting() {
  waiting = false;
  clearTimeout(thinkingTimer);
  thinkingTimer = null;
}

function onTalkKey(isHeld) {
  held = isHeld;
  if (held) {
    // Holding means "I'm talking now" — any old reply text disappears.
    clearResponse();
    stopWaiting();
  } else if (appActive) {
    waiting = true;
    clearTimeout(thinkingTimer);
    thinkingTimer = setTimeout(() => {
      waiting = false;
      render();
    }, THINKING_TIMEOUT_MS);
  }
  render();
}

function onStateChanged(fields) {
  appActive = fields.new === "active";
  if (!appActive) {
    // Session over (or never opened): back to the prompt.
    clearResponse();
    stopWaiting();
  }
  render();
}

function onAgentSaid(fields) {
  stopWaiting();
  if (held) return; // barge-in: the person is talking over it; drop the tail
  const now = Date.now();
  if (response.final) clearResponse(); // a new turn after a closed one
  if (!response.text) turnStartedAt = now;
  clearTimeout(lingerTimer);
  lingerTimer = null;
  response.text += fields.text;
  response.final = fields.final;
  lastPieceAt = now;
  if (fields.final) {
    const speakMs = response.text.length * SPEAK_MS_PER_CHAR - (now - turnStartedAt);
    const linger = Math.min(LINGER_MAX_MS, Math.max(LINGER_MIN_MS, speakMs));
    lingerTimer = setTimeout(() => {
      clearResponse();
      render();
    }, linger);
  }
  render();
}

// Sweep a reply whose stream died without a final piece off the screen.
setInterval(() => {
  if (response.text && !response.final && Date.now() - lastPieceAt > RESPONSE_STALL_MS) {
    clearResponse();
    render();
  }
}, 2000);

// --- connection ------------------------------------------------------------

function handleMessage(msg) {
  switch (msg.type) {
    case "TalkKeyChanged":
      onTalkKey(msg.fields.held);
      break;
    case "StateChanged":
      onStateChanged(msg.fields);
      break;
    case "AgentSaid":
      onAgentSaid(msg.fields);
      break;
    default:
      break; // everything else is dashboard material, not visitor material
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    connected = true;
    reconnectDelay = 1000;
    // Everything below is re-seeded by the server on this connection;
    // start neutral rather than trusting pre-disconnect leftovers.
    held = false;
    appActive = false;
    clearResponse();
    stopWaiting();
    render();
  };
  ws.onclose = () => {
    connected = false;
    render();
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 15000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    try {
      handleMessage(JSON.parse(ev.data));
    } catch (err) {
      console.error("bad message from HARP", err, ev.data);
    }
  };
}

connect();
render();
