'use strict';

/**
 * knowledge.js — tiny, dependency-free keyword search over harp/data/*.md.
 *
 * This is the backend for the `search_knowledge` function tool. It chunks the
 * markdown by heading, then ranks chunks against a query with BM25 (a standard
 * keyword-relevance score). No embeddings, no vector DB — appropriate for a
 * small corpus. The search() interface is what the tool calls; swapping in
 * embeddings later means rewriting only this file.
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');

// Common words that add noise to keyword matching.
const STOP = new Set(
  ('a an the of to in on at for and or is are was were be been being with from by as it its this ' +
    'that these those you your we our they i he she him her them not no do does did how what when ' +
    'where who why which will would can could should about into over under more most there here')
    .split(/\s+/)
);

// Keep Latin word chars and the Urdu/Arabic script range so Urdu queries tokenize too.
function tokenize(s) {
  const matches = s.toLowerCase().match(/[a-z0-9؀-ۿ]+/g) || [];
  return matches.filter((t) => t.length > 1 && !STOP.has(t));
}

// Split a markdown file into chunks at headings, each keeping its heading.
function chunkMarkdown(text, source) {
  const lines = text.split('\n');
  const chunks = [];
  let cur = { source, heading: '', body: [] };
  const flush = () => {
    const bodyText = cur.body.join('\n').trim();
    if (bodyText || cur.heading) {
      chunks.push({
        source,
        heading: cur.heading,
        text: (cur.heading ? cur.heading + '\n' : '') + bodyText,
      });
    }
  };
  for (const line of lines) {
    if (/^#{1,6}\s/.test(line)) {
      flush();
      cur = { source, heading: line.replace(/^#{1,6}\s+/, '').trim(), body: [] };
    } else {
      cur.body.push(line);
    }
  }
  flush();
  return chunks.filter((c) => c.text.trim().length > 0);
}

let CHUNKS = [];
let DF = {}; // term -> number of chunks containing it
let AVG_LEN = 0;

function buildIndex() {
  CHUNKS = [];
  DF = {};
  let files = [];
  try {
    files = fs.readdirSync(DATA_DIR).filter((f) => f.toLowerCase().endsWith('.md'));
  } catch (_) {
    return 0;
  }
  for (const f of files) {
    const text = fs.readFileSync(path.join(DATA_DIR, f), 'utf8');
    for (const c of chunkMarkdown(text, f)) CHUNKS.push(c);
  }
  let totalLen = 0;
  for (const c of CHUNKS) {
    c.tokens = tokenize(c.text);
    c.len = c.tokens.length;
    c.tf = Object.create(null);
    for (const t of c.tokens) c.tf[t] = (c.tf[t] || 0) + 1;
    totalLen += c.len;
    for (const t of Object.keys(c.tf)) DF[t] = (DF[t] || 0) + 1;
  }
  AVG_LEN = CHUNKS.length ? totalLen / CHUNKS.length : 0;
  return CHUNKS.length;
}

// BM25 ranking. Returns the top-k chunks above zero relevance.
function search(query, k) {
  k = k || 3;
  const N = CHUNKS.length || 1;
  const k1 = 1.5;
  const b = 0.75;
  const terms = Array.from(new Set(tokenize(query || '')));
  if (!terms.length) return [];

  const scored = [];
  for (const c of CHUNKS) {
    let score = 0;
    for (const t of terms) {
      const tf = c.tf[t];
      if (!tf) continue;
      const df = DF[t] || 1;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * c.len) / AVG_LEN)));
    }
    if (score > 0) scored.push({ c, score });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map((x) => ({
    source: x.c.source,
    heading: x.c.heading,
    text: x.c.text,
    score: Math.round(x.score * 1000) / 1000,
  }));
}

module.exports = { buildIndex, search };
