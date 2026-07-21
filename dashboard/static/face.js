// HARP animated face — same behavior as the original harp/motion/face.html +
// face_server.py (poll /state every 300ms for {face: bool}), but sourced from
// the dashboard's own /ws bus stream instead of a second local HTTP server.
//
// face_server.py's `on_face_change` was wired to the GIMBAL's own tracking
// state (Gimbal.track()/tick() in harp/motion/gimbal.py — "is the neck
// currently locked onto a face"), not the main agent's face-ID. That distinct
// signal doesn't cross the process boundary onto harp.app's bus, so it can't
// be reproduced here without wiring harp.motion onto the bus (a bigger change
// than this page should make on its own). The nearest equivalent already on
// the bus is face-ID's PresenceChanged (harp/vision/face_id.py) — "is anyone
// in the shared camera's frame" — which drives the same sad<->surprised/happy
// transition below. If harp.motion's gimbal is what should drive this face
// instead, wire Gimbal(on_face_change=...) to publish PresenceChanged on the
// shared bus and this page needs no further changes.

let ws = null;
let reconnectDelay = 1000;

const face = document.getElementById("face");
let lastPresent = null;
let excitedTimer = null;

function setEmotion(name) {
  face.className = "face " + name;
}

function onPresenceChanged(present) {
  if (present === lastPresent) return;
  lastPresent = present;
  clearTimeout(excitedTimer);
  if (present) {
    setEmotion("surprised");
    setTimeout(() => {
      if (lastPresent) setEmotion("happy");
    }, 700);
    excitedTimer = setTimeout(() => {
      if (lastPresent) {
        setEmotion("excited");
        setTimeout(() => {
          if (lastPresent) setEmotion("happy");
        }, 900);
      }
    }, 6000 + Math.random() * 4000);
  } else {
    setEmotion("sad");
  }
}

function handleMessage(msg) {
  if (msg.type === "PresenceChanged") {
    onPresenceChanged(!!(msg.fields && msg.fields.present));
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    reconnectDelay = 1000;
  };
  ws.onclose = () => {
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

// Start neutral (idle blink) until the first real PresenceChanged arrives,
// same spirit as the original page starting on "sad" until its first poll.
setEmotion("neutral");
connect();
