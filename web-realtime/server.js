'use strict';

/**
 * Laila Realtime — token + static server.
 *
 * Responsibilities (server-only, never the browser):
 *   1. Hold the real OPENAI_API_KEY.
 *   2. Build the session config (model, voice, persona, transcription, VAD).
 *   3. Mint a short-lived ephemeral client secret (ek_...) for the browser.
 *   4. Serve the static front-end.
 *
 * The browser only ever receives the ephemeral secret, which expires in minutes
 * and is scoped to a single realtime session.
 *
 * Zero runtime dependencies — uses Node built-ins only, so `npm install` is not
 * required. Runs on Node 12+.
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const knowledge = require('./knowledge');

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const PROMPT_FILE = path.join(ROOT, '..', 'prompts', 'system_instructions.md');

// ---------------------------------------------------------------------------
// Env loading: prefer the live shell env, then fall back to the harp root .env.
// We do NOT create our own .env — we reuse the one you already filled in.
// ---------------------------------------------------------------------------
function loadEnvFile(file) {
  try {
    const text = fs.readFileSync(file, 'utf8');
    for (const rawLine of text.split('\n')) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#')) continue;
      const eq = line.indexOf('=');
      if (eq === -1) continue;
      const key = line.slice(0, eq).trim();
      let val = line.slice(eq + 1).trim();
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.slice(1, -1);
      }
      if (!(key in process.env) && val !== '') process.env[key] = val;
    }
  } catch (_) {
    /* file absent is fine */
  }
}

// Local override first (if a web-realtime/.env exists), then the shared harp .env.
loadEnvFile(path.join(ROOT, '.env'));
loadEnvFile(path.join(ROOT, '..', '.env'));

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.REALTIME_MODEL || 'gpt-realtime-2';
const VOICE = process.env.REALTIME_VOICE || 'marin';
const PORT = parseInt(process.env.PORT, 10) || 3000;

// ---------------------------------------------------------------------------
// Persona: single source of truth is harp/prompts/system_instructions.md.
// We strip the markdown title + the leading "> note" blockquote so the model
// gets the real instructions, not the authoring notes.
// ---------------------------------------------------------------------------
const FALLBACK_PERSONA =
  "You are Laila, a friendly robot dolphin at the reception of a robotics expo. " +
  "Speak out loud in short, warm, spoken sentences (no markdown). Mirror the user's " +
  "language: reply in English to English, Urdu to Urdu, and match a natural Urdu/English " +
  "mix. Do not speak other languages. Be honest when you don't know something. You can " +
  'talk and listen but cannot move. Keep it appropriate for all ages.';

function loadPersona() {
  try {
    const raw = fs.readFileSync(PROMPT_FILE, 'utf8');
    const kept = raw
      .split('\n')
      .filter((l) => !l.trim().startsWith('>') && !l.trim().startsWith('# '))
      .join('\n')
      .trim();
    return kept.length > 40 ? kept : FALLBACK_PERSONA;
  } catch (_) {
    return FALLBACK_PERSONA;
  }
}

// ---------------------------------------------------------------------------
// Mint an ephemeral client secret from the Realtime API.
// POST https://api.openai.com/v1/realtime/client_secrets
// ---------------------------------------------------------------------------
function createClientSecret() {
  const body = JSON.stringify({
    // The session config is bound to the ephemeral secret here, server-side,
    // so the browser cannot tamper with the model, voice, or persona.
    expires_after: { anchor: 'created_at', seconds: 600 },
    session: {
      type: 'realtime',
      model: MODEL,
      instructions: loadPersona(),
      audio: {
        input: {
          // Turn on input transcription so we can show what the user said.
          transcription: { model: 'gpt-4o-mini-transcribe' },
          // Server-side voice activity detection: the model decides when a turn ends.
          turn_detection: {
            type: 'server_vad',
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
          },
        },
        output: { voice: VOICE },
      },
      // The model can call this to look things up in data/ before answering.
      tools: [
        {
          type: 'function',
          name: 'search_knowledge',
          description:
            'Search the expo knowledge base for facts about the event: schedule, ' +
            'speakers, venue, tickets, exhibitors, and general info. The knowledge ' +
            'base is written in English, so always query with concise English ' +
            'keywords even if the visitor spoke Urdu. Call this BEFORE answering any ' +
            'question about the expo, and base your spoken reply on what it returns. ' +
            'If it returns nothing useful, say you are not sure rather than guessing.',
          parameters: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'A few English keywords describing what to look up.',
              },
            },
            required: ['query'],
          },
        },
      ],
      tool_choice: 'auto',
    },
  });

  const options = {
    method: 'POST',
    hostname: 'api.openai.com',
    path: '/v1/realtime/client_secrets',
    headers: {
      Authorization: 'Bearer ' + OPENAI_API_KEY,
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body),
    },
  };

  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (c) => (data += c));
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          return reject(new Error('OpenAI ' + res.statusCode + ': ' + data));
        }
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error('Bad JSON from OpenAI: ' + data));
        }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Static file serving for public/.
// ---------------------------------------------------------------------------
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function serveStatic(req, res) {
  let urlPath = decodeURIComponent(req.url.split('?')[0]);
  if (urlPath === '/') urlPath = '/index.html';
  const filePath = path.normalize(path.join(PUBLIC_DIR, urlPath));
  // Prevent path traversal outside public/.
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    return res.end('Forbidden');
  }
  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(404);
      return res.end('Not found');
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' });
    res.end(content);
  });
}

function sendJson(res, code, obj) {
  const payload = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(payload);
}

function readJsonBody(req) {
  return new Promise((resolve) => {
    let data = '';
    req.on('data', (c) => {
      data += c;
      if (data.length > 1e5) req.destroy(); // guard against oversized bodies
    });
    req.on('end', () => {
      try {
        resolve(JSON.parse(data || '{}'));
      } catch (_) {
        resolve({});
      }
    });
    req.on('error', () => resolve({}));
  });
}

// ---------------------------------------------------------------------------
// Routes.
// ---------------------------------------------------------------------------
const server = http.createServer(async (req, res) => {
  // POST /session -> mint and return an ephemeral client secret to the browser.
  if (req.url === '/session' && req.method === 'POST') {
    if (!OPENAI_API_KEY) {
      return sendJson(res, 500, {
        error: 'OPENAI_API_KEY is not set. Add it to harp/.env (or export it) and restart.',
      });
    }
    try {
      const data = await createClientSecret();
      // Return only what the browser needs. Different API versions nest the
      // secret differently, so normalise it here.
      const value =
        data.value ||
        data.client_secret ||
        (data.client_secret && data.client_secret.value);
      return sendJson(res, 200, {
        value: value,
        expires_at: data.expires_at,
        model: MODEL,
        voice: VOICE,
      });
    } catch (e) {
      return sendJson(res, 502, { error: String(e.message || e) });
    }
  }

  // POST /search -> keyword search over data/, used by the search_knowledge tool.
  // The browser relays the model's function call here; the index stays server-side.
  if (req.url === '/search' && req.method === 'POST') {
    const body = await readJsonBody(req);
    const results = knowledge.search(body.query || '', 3);
    return sendJson(res, 200, { query: body.query || '', results });
  }

  if (req.method === 'GET') return serveStatic(req, res);

  res.writeHead(405);
  res.end('Method not allowed');
});

const chunkCount = knowledge.buildIndex();

server.listen(PORT, () => {
  console.log('\n  Laila Realtime');
  console.log('  ─────────────────────────────────────────');
  console.log('  Local:  http://localhost:' + PORT);
  console.log('  Model:  ' + MODEL + '   Voice: ' + VOICE);
  console.log('  Key:    ' + (OPENAI_API_KEY ? 'loaded ✓' : 'MISSING ✗  (set OPENAI_API_KEY in harp/.env)'));
  console.log('  Data:   ' + chunkCount + ' chunks indexed from ../data/*.md');
  console.log('  ─────────────────────────────────────────\n');
});
