// HARP dashboard client. Vanilla JS, no build step: connects to /ws, renders
// every bus event. Unknown event types still show up in the raw log below,
// so new event types need no change here to become visible.

const MAX_TRANSCRIPT_TURNS = 200;
const MAX_TOOL_ROWS = 50;
const MAX_LOG_ROWS = 300;
const MAX_ERROR_ROWS = 50;
const HEARTBEAT_STALE_SECONDS = 10;

let eventCount = 0;
let reconnectDelay = 1000;
let ws = null;
let micMuted = null; // null = not known yet (no MicMuteChanged seen)

// Transcript accumulation: the provider streams transcript pieces as deltas
// (see harp/voice/session.py's _Printer), so consecutive non-final pieces
// from the same speaker are appended to one turn, not replaced.
let currentTurn = null; // { speaker, textEl, final }

const toolRows = new Map(); // id -> row element

function el(tag, opts = {}) {
  const node = document.createElement(tag);
  if (opts.className) node.className = opts.className;
  if (opts.text !== undefined) node.textContent = opts.text;
  return node;
}

function fmtTime(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleTimeString();
}

function clearEmpty(container) {
  const empty = container.querySelector(".empty");
  if (empty) empty.remove();
}

function capRows(container, max) {
  while (container.children.length > max) {
    container.removeChild(container.firstChild);
  }
}

// --- connection -------------------------------------------------------------

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setConnected(true);
    reconnectDelay = 1000;
  };
  ws.onclose = () => {
    setConnected(false);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 15000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    try {
      handleMessage(JSON.parse(ev.data));
    } catch (err) {
      console.error("bad message from dashboard server", err, ev.data);
    }
  };
}

function setConnected(connected) {
  document.getElementById("conn-dot").className = "dot " + (connected ? "dot-on" : "dot-off");
  document.getElementById("conn-text").textContent = connected
    ? "connected to dashboard server"
    : "disconnected — retrying…";
  if (!connected) {
    // Mute state came from this connection — a stale label on reconnect
    // would be a guess, not a fact.
    micMuted = null;
  }
  updateMuteButton();
}

// --- dispatch -----------------------------------------------------------

function handleMessage(msg) {
  eventCount += 1;
  document.getElementById("event-count").textContent = String(eventCount);
  document.getElementById("last-event-at").textContent = fmtTime(msg.server_ts);

  logRaw(msg);

  switch (msg.type) {
    case "StateChanged":
      renderState(msg.fields);
      break;
    case "PresenceChanged":
      renderPresence(msg.fields);
      break;
    case "PersonIdentified":
      renderPerson(msg.fields);
      break;
    case "GestureDetected":
      renderGesture(msg.fields);
      break;
    case "PhraseHeard":
      renderPhraseHeard(msg.fields);
      break;
    case "MicMuteChanged":
      micMuted = msg.fields.muted;
      updateMuteButton();
      break;
    case "InteractionStarted":
      addDivider(`interaction started (${msg.fields.reason})`);
      currentTurn = null;
      break;
    case "InteractionEnded":
      addDivider(`interaction ended (${msg.fields.reason})`);
      currentTurn = null;
      break;
    case "UserSaid":
      addTranscriptPiece("user", msg.fields.text, msg.fields.final);
      break;
    case "AgentSaid":
      addTranscriptPiece("agent", msg.fields.text, msg.fields.final);
      break;
    case "ToolRequested":
      renderToolRequested(msg.fields);
      break;
    case "ToolCompleted":
      renderToolCompleted(msg.fields);
      break;
    case "Heartbeat":
      renderHeartbeat(msg.fields);
      break;
    case "ErrorRaised":
      renderError(msg.fields);
      break;
    default:
      // No dedicated panel — the raw log above already recorded it.
      break;
  }
}

// --- mic mute button -----------------------------------------------------
// The dashboard's one write action (see server.py's docstring): clicking
// sends {type: "SetMicMuted", muted} over the same WS connection; the button
// only actually flips its label once the server echoes back MicMuteChanged
// (confirmation-based, not optimistic) — so it also stays correct if muted
// from a different tab/phone.

const muteBtn = document.getElementById("mute-btn");

function updateMuteButton() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    muteBtn.disabled = true;
    muteBtn.textContent = "mic — connecting…";
    muteBtn.className = "mute-btn";
    return;
  }
  if (micMuted === null) {
    muteBtn.disabled = true;
    muteBtn.textContent = "mic — state unknown";
    muteBtn.className = "mute-btn";
    return;
  }
  muteBtn.disabled = false;
  muteBtn.textContent = micMuted ? "MIC MUTED — tap to unmute" : "mic live — tap to mute";
  muteBtn.className = "mute-btn " + (micMuted ? "muted" : "live");
}

muteBtn.addEventListener("click", () => {
  if (micMuted === null || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "SetMicMuted", muted: !micMuted }));
});

// --- state panel -------------------------------------------------------

function renderState(fields) {
  const body = document.getElementById("state-body");
  body.innerHTML = "";
  const row = el("div", { className: "kv" });
  row.appendChild(el("span", { text: fields.old + " → " + fields.new }));
  body.appendChild(row);
}

// --- presence panel ------------------------------------------------------

function renderPresence(fields) {
  const target = document.getElementById("presence-text");
  target.classList.remove("empty");
  target.textContent = fields.present ? `present (${fields.count})` : "nobody present";
}

function renderPerson(fields) {
  const target = document.getElementById("person-text");
  target.classList.remove("empty");
  const name = fields.name || fields.person_id;
  target.textContent = `${name} (${fields.is_known ? "known" : "new"}, ${Math.round(fields.confidence * 100)}%)`;
}

function renderGesture(fields) {
  const target = document.getElementById("gesture-text");
  target.classList.remove("empty");
  target.textContent = `${fields.kind} — ${new Date().toLocaleTimeString()}`;
}

// --- heard while idle (wake listener) ------------------------------------

function renderPhraseHeard(fields) {
  const container = document.getElementById("heard-list");
  clearEmpty(container);
  const row = el("div", { className: "log-row" });
  row.appendChild(el("span", { className: "ts", text: new Date().toLocaleTimeString() }));
  row.appendChild(el("span", {
    className: "badge " + (fields.wake_word ? "badge-ok" : "badge-warn"),
    text: fields.wake_word ? `wake: ${fields.wake_word}` : "no wake word",
  }));
  row.appendChild(el("span", { className: "fields", text: fields.text }));
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  capRows(container, MAX_LOG_ROWS);
}

// --- health panel --------------------------------------------------------

let lastHeartbeatTs = null;

function renderHeartbeat(fields) {
  lastHeartbeatTs = fields.ts;
  updateHeartbeatText();
}

function updateHeartbeatText() {
  const target = document.getElementById("heartbeat-text");
  if (lastHeartbeatTs === null) {
    target.classList.add("empty");
    target.textContent = "no heartbeat received";
    return;
  }
  target.classList.remove("empty");
  const ageSeconds = Date.now() / 1000 - lastHeartbeatTs;
  target.innerHTML = "";
  const badge = el("span", {
    className: "badge " + (ageSeconds > HEARTBEAT_STALE_SECONDS ? "badge-err" : "badge-ok"),
    text: `${ageSeconds.toFixed(0)}s ago`,
  });
  target.appendChild(badge);
}
setInterval(updateHeartbeatText, 1000);

function renderError(fields) {
  const list = document.getElementById("errors-list");
  const row = el("div", { className: "error-row" + (fields.fatal ? "" : " non-fatal") });
  row.appendChild(el("div", { text: `${fields.where}: ${fields.message}` }));
  row.appendChild(el("div", { className: "dim", text: fields.fatal ? "fatal" : "non-fatal" }));
  list.prepend(row);
  capRows(list, MAX_ERROR_ROWS);
}

// --- transcript ------------------------------------------------------------

function addDivider(text) {
  const container = document.getElementById("transcript");
  clearEmpty(container);
  container.appendChild(el("div", { className: "turn divider", text }));
  capRows(container, MAX_TRANSCRIPT_TURNS);
}

function addTranscriptPiece(speaker, text, final) {
  const container = document.getElementById("transcript");
  clearEmpty(container);

  if (!currentTurn || currentTurn.speaker !== speaker || currentTurn.final) {
    const turnEl = el("div", { className: `turn ${speaker}` + (final ? "" : " partial") });
    turnEl.appendChild(el("span", { className: "who", text: speaker === "user" ? "user" : "harp" }));
    const textEl = el("span", { className: "text", text: "" });
    turnEl.appendChild(textEl);
    container.appendChild(turnEl);
    currentTurn = { speaker, textEl, turnEl, final: false };
  }

  currentTurn.textEl.textContent += text;
  currentTurn.final = final;
  currentTurn.turnEl.className = `turn ${speaker}` + (final ? "" : " partial");
  if (final) currentTurn = null;

  container.scrollTop = container.scrollHeight;
  capRows(container, MAX_TRANSCRIPT_TURNS);
}

// --- tool calls --------------------------------------------------------

function renderToolRequested(fields) {
  const container = document.getElementById("tools-list");
  clearEmpty(container);

  const row = el("div", { className: "tool-row" });
  const header = el("div");
  header.appendChild(el("span", { className: "name", text: fields.name }));
  header.appendChild(document.createTextNode(" "));
  header.appendChild(el("span", { className: "badge badge-warn", text: "pending" }));
  row.appendChild(header);
  const args = el("pre", { text: JSON.stringify(fields.arguments, null, 2) });
  row.appendChild(args);

  container.prepend(row);
  toolRows.set(fields.id, { row, header, name: fields.name });
  capRows(container, MAX_TOOL_ROWS);
}

function renderToolCompleted(fields) {
  const container = document.getElementById("tools-list");
  clearEmpty(container);

  const existing = toolRows.get(fields.id);
  const row = existing ? existing.row : el("div", { className: "tool-row" });
  if (!existing) {
    row.appendChild(el("div", { className: "name", text: `(unknown tool, id=${fields.id})` }));
    container.prepend(row);
  }

  const badge = row.querySelector(".badge");
  if (badge) {
    badge.textContent = "done";
    badge.className = "badge badge-ok";
  }

  const outputPre = el("pre", { text: "→ " + JSON.stringify(fields.output, null, 2) });
  row.appendChild(outputPre);
  capRows(container, MAX_TOOL_ROWS);
}

// --- camera feed ---------------------------------------------------------
// Frames are NOT bus events (too big); the server exposes the shared camera's
// latest frame read-only at /camera.jpg. Poll it, keeping the last good image
// up; 404 means no camera is attached this run.

const CAMERA_REFRESH_MS = 250;
const CAMERA_RETRY_MS = 2000;

function pollCamera() {
  const img = document.getElementById("camera-img");
  const empty = document.getElementById("camera-empty");
  const probe = new Image();
  probe.onload = () => {
    img.src = probe.src;
    img.hidden = false;
    empty.hidden = true;
    setTimeout(pollCamera, CAMERA_REFRESH_MS);
  };
  probe.onerror = () => {
    img.hidden = true;
    empty.hidden = false;
    empty.textContent = "No camera feed — camera off or not attached.";
    setTimeout(pollCamera, CAMERA_RETRY_MS);
  };
  probe.src = "/camera.jpg?t=" + Date.now();
}

// --- raw event log -----------------------------------------------------

function logRaw(msg) {
  const container = document.getElementById("raw-log");
  clearEmpty(container);

  const row = el("div", { className: "log-row" });
  row.appendChild(el("span", { className: "ts", text: fmtTime(msg.server_ts) }));
  row.appendChild(el("span", { className: "type", text: msg.type }));
  row.appendChild(el("span", { className: "fields", text: JSON.stringify(msg.fields) }));
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  capRows(container, MAX_LOG_ROWS);
}

connect();
pollCamera();
