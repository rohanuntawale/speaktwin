/**
 * ============================================================
 *  SpeakTwin — application logic
 * ============================================================
 *  Captures microphone audio in the browser, ships 2.5 s WAV
 *  chunks to the backend, and renders what comes back.
 *
 *  Audio is tapped twice from one stream: an AnalyserNode feeds
 *  the voiceprint at frame rate, while a ScriptProcessor
 *  accumulates raw samples for the chunk uploads.
 */

// ── Configuration ────────────────────────────────────────────
// Same-origin: the backend serves this page, so hard-coding a
// port only creates a mismatch to keep in sync.
const API_BASE = window.SPEAKTWIN_API_BASE || "";
const CHUNK_MS = 2500;
const TARGET_RATE = 16000;

// ── Elements ─────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const el = {
  mic: $("mic"),
  micGlyph: $("mic-glyph"),
  prompt: $("prompt"),
  stateText: $("state-text"),
  state: $("state"),

  faderEnergy: $("fader-energy"),
  faderPitch: $("fader-pitch"),
  readEnergy: $("read-energy"),
  readPitch: $("read-pitch"),

  scoreNum: $("score-num"),
  bdWpm: $("bd-wpm"),
  bdPitch: $("bd-pitch"),
  bdEnergy: $("bd-energy"),
  bdFiller: $("bd-filler"),

  insight: $("insight"),
  notes: $("notes"),

  valWpm: $("val-wpm"),
  valFiller: $("val-filler"),
  valClarity: $("val-clarity"),
  valPause: $("val-pause"),

  transcript: $("transcript"),
  transcriptEmpty: $("transcript-empty"),
  transcriptScroll: $("transcript-scroll"),

  fillers: $("fillers"),
  keywords: $("keywords"),

  ai: $("ai"),
  aiRows: $("ai-rows"),
  footEngine: $("foot-engine"),
};

// ── State ────────────────────────────────────────────────────
let recording = false;
let sessionId = null;
let stream = null;
let audioCtx = null;
let source = null;
let processor = null;
let analyser = null;
let voiceprint = null;

let totals = { fillers: {}, keywords: {} };

// ── Utilities ────────────────────────────────────────────────
function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text == null ? "" : String(text);
  return d.innerHTML;
}

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Count toward a number instead of snapping to it. */
function countTo(node, value, suffix = "") {
  const from = parseFloat(node.dataset.v || "0");
  const to = Number(value) || 0;
  if (from === to) return;
  node.dataset.v = String(to);

  const start = performance.now();
  const dur = 450;

  const step = (now) => {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    const current = Math.round(from + (to - from) * eased);
    node.innerHTML = current + suffix;
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function setStatus(text, tone) {
  if (el.stateText) el.stateText.textContent = text;
  if (el.state) el.state.classList.toggle("error", tone === "error");
}

// ── Recording ────────────────────────────────────────────────
el.mic.addEventListener("click", () => {
  recording ? stopListening() : startListening();
});

function setLive(live) {
  recording = live;
  document.body.classList.toggle("live", live);
  el.mic.setAttribute("aria-label", live ? "Stop listening" : "Start listening");
  el.prompt.textContent = live ? "Listening — press to stop" : "Press to start listening";
  setStatus(live ? "Listening" : "Idle");

  el.micGlyph.innerHTML = live
    ? `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="2.5"/></svg>`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
         stroke-linecap="round" stroke-linejoin="round">
         <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
         <path d="M19 11v1a7 7 0 0 1-14 0v-1"/>
         <line x1="12" y1="19" x2="12" y2="22"/>
       </svg>`;
}

async function startListening() {
  try {
    setStatus("Requesting microphone");
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    resetSession();
    sessionId = await apiCreateSession();

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: TARGET_RATE,
    });
    source = audioCtx.createMediaStreamSource(stream);

    // Fast clock: drives the voiceprint at frame rate.
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.72;
    source.connect(analyser);
    voiceprint.listen(analyser);

    // Slow clock: accumulate raw samples for chunk uploads.
    processor = audioCtx.createScriptProcessor(4096, 1, 1);

    // The browser may ignore the sample-rate hint, so send what we
    // actually got — the backend resamples rather than assuming.
    const rate = audioCtx.sampleRate;
    const needed = Math.round(rate * (CHUNK_MS / 1000));
    let buffer = [];

    processor.onaudioprocess = async (e) => {
      if (!recording) return;
      const pcm = e.inputBuffer.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) buffer.push(pcm[i]);

      if (buffer.length >= needed) {
        const slice = new Float32Array(buffer.slice(0, needed));
        buffer = buffer.slice(needed);
        const result = await apiAnalyze(encodeWav(slice, rate));
        if (result) render(result);
      }
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    setLive(true);
  } catch (err) {
    console.error("[SpeakTwin]", err);
    setStatus("Microphone blocked", "error");
    say("Microphone access is blocked. Allow it in your browser's site settings, then press again.", "poor");
  }
}

async function stopListening() {
  setLive(false);
  if (voiceprint) voiceprint.stop();

  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (processor) processor.disconnect();
  if (source) source.disconnect();
  if (analyser) analyser.disconnect();
  if (audioCtx && audioCtx.state !== "closed") audioCtx.close();
  stream = audioCtx = source = processor = analyser = null;

  setFader(el.faderEnergy, 0);
  setFader(el.faderPitch, 0);

  if (sessionId) {
    const report = await apiEndSession(sessionId);
    sessionId = null;
    if (report) summarise(report);
  }
}

function resetSession() {
  totals = { fillers: {}, keywords: {} };
  el.transcript.innerHTML = "";
  el.transcriptEmpty.style.display = "";
  el.fillers.innerHTML = `<p class="empty">No fillers yet.</p>`;
  el.keywords.innerHTML = `<p class="empty">Say a target word to collect it.</p>`;
  el.notes.innerHTML = `<p class="empty">Listening…</p>`;
  el.ai.hidden = true;
  if (voiceprint) voiceprint.reset();
  ["valWpm", "valFiller", "valClarity"].forEach((k) => {
    el[k].dataset.v = "0";
    el[k].textContent = "0";
  });
  el.valPause.dataset.v = "0";
  el.valPause.innerHTML = `0<i>%</i>`;
  el.scoreNum.textContent = "—";
  document.querySelectorAll(".cell").forEach((c) => c.removeAttribute("data-tone"));
}

/** Float32 PCM → 16-bit WAV. */
function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const str = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };

  str(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// ── API ──────────────────────────────────────────────────────
async function apiCreateSession() {
  try {
    const res = await fetch(`${API_BASE}/api/session`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()).session_id;
  } catch (err) {
    console.warn("[SpeakTwin] No session:", err);
    return null;   // analysis still works, just without smoothing
  }
}

async function apiEndSession(id) {
  try {
    const res = await fetch(`${API_BASE}/api/session/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("[SpeakTwin] Session not closed:", err);
    return null;
  }
}

async function apiAnalyze(blob) {
  try {
    const form = new FormData();
    form.append("audio_file", blob, "chunk.wav");
    if (sessionId) form.append("session_id", sessionId);

    const res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: form });

    if (!res.ok) {
      let message = `Server error ${res.status}`;
      try {
        message = (await res.json()).message || message;
      } catch (_) { /* non-JSON body */ }
      throw new Error(message);
    }
    if (recording) setStatus("Listening");
    return await res.json();
  } catch (err) {
    console.error("[SpeakTwin]", err);
    setStatus("Connection lost", "error");
    say(err.message, "poor");
    return null;
  }
}

// ── Rendering ────────────────────────────────────────────────
function render(d) {
  if (!d || d.status === "error") return;

  // Instrument — the smoothed score trends instead of jumping
  const score = d.confidence_smoothed ?? d.confidence_score ?? 0;
  voiceprint.update({
    score,
    energyDb: d.energy_db,
    tooLoud: d.energy_db > -12,
  });
  el.scoreNum.textContent = Math.round(score);
  el.scoreNum.style.color = `rgb(${voiceprint.colour().join(",")})`;

  // Live acoustic rails
  setFader(el.faderEnergy, clamp01((d.energy_db + 60) / 48));
  el.readEnergy.textContent = `${Math.round(d.energy_db)} dB`;

  const pitchNorm = d.pitch > 0 ? clamp01((d.pitch - 60) / 300) : 0;
  setFader(el.faderPitch, pitchNorm);
  el.readPitch.textContent = d.pitch > 0 ? `${Math.round(d.pitch)} Hz` : "—";

  // Breakdown
  const bd = d.confidence_breakdown || {};
  setBar(el.bdWpm, bd.wpm);
  setBar(el.bdPitch, bd.pitch_variation);
  setBar(el.bdEnergy, bd.energy);
  setBar(el.bdFiller, bd.filler_usage);

  // Readouts
  countTo(el.valWpm, Math.round(d.wpm));
  countTo(el.valClarity, Math.round(d.clarity || 0));
  countTo(el.valPause, Math.round((d.pause_ratio || 0) * 100), "<i>%</i>");

  tone(el.valWpm.closest(".cell"), wpmTone(d.wpm));
  tone(el.valClarity.closest(".cell"), d.clarity >= 70 ? "good" : d.clarity >= 40 ? null : "warn");

  // Headline
  say(d.message, d.status);
  notes(d.feedback);

  // Transcript
  if (d.transcript) appendTranscript(d);

  // Totals: prefer the server session, fall back to local
  if (d.session) {
    totals.fillers = d.session.filler_details || {};
    totals.keywords = d.session.keyword_details || {};
  } else {
    accumulate(totals.fillers, d.fillers && d.fillers.details);
    accumulate(totals.keywords, d.keywords && d.keywords.found_keywords);
  }
  const fillerCount = Object.values(totals.fillers).reduce((a, b) => a + b, 0);
  countTo(el.valFiller, fillerCount);
  tone(el.valFiller.closest(".cell"), fillerCount > 6 ? "over" : fillerCount > 2 ? "warn" : null);

  renderTags(el.fillers, totals.fillers, "fill", "No fillers yet.");
  renderTags(el.keywords, totals.keywords, "key", "Say a target word to collect it.");

  neural(d);

  if (d.degraded && d.warnings?.includes("stt_unavailable")) {
    setStatus("Transcription unavailable", "error");
  }
}

const clamp01 = (v) => Math.max(0, Math.min(1, v || 0));

function setFader(fader, value) {
  const fill = fader.querySelector(".fader-fill");
  if (fill) fill.style.right = `${100 - clamp01(value) * 100}%`;
}

function setBar(node, pct) {
  if (node) node.style.width = `${Math.round(clamp01((pct || 0) / 100) * 100)}%`;
}

function tone(cell, value) {
  if (!cell) return;
  value ? cell.setAttribute("data-tone", value) : cell.removeAttribute("data-tone");
}

function wpmTone(wpm) {
  if (!wpm) return null;
  if (wpm < 100 || wpm > 175) return "over";
  if (wpm < 120 || wpm > 160) return "warn";
  return "good";
}

/** Swap the headline with a blur transition rather than a hard cut. */
function say(text, status) {
  if (!text || el.insight.textContent === text) return;
  el.insight.classList.add("swap");
  setTimeout(() => {
    el.insight.textContent = text;
    el.insight.className = `insight ${status || ""}`;
  }, 300);
}

function notes(messages) {
  if (!Array.isArray(messages) || !messages.length) return;
  const order = { warning: 0, info: 1, success: 2 };
  el.notes.innerHTML = "";
  [...messages]
    .sort((a, b) => (order[a.type] ?? 3) - (order[b.type] ?? 3))
    .forEach((m, i) => {
      const row = document.createElement("div");
      row.className = `note ${m.type || "info"}`;
      row.style.animationDelay = `${i * 40}ms`;
      row.innerHTML =
        `<span class="note-tag">${escapeHtml(m.category || "")}</span>` +
        `<span>${escapeHtml(m.text || "")}</span>`;
      el.notes.appendChild(row);
    });
}

function appendTranscript(d) {
  // Escape first — anything the STT engine returns is untrusted.
  let html = escapeHtml(d.transcript);

  const wrap = (phrases, cls) => {
    (phrases || []).forEach((p) => {
      if (!p) return;
      const re = new RegExp(`\\b${escapeRegex(escapeHtml(p))}\\b`, "gi");
      html = html.replace(re, `<mark class="${cls}">$&</mark>`);
    });
  };
  wrap(d.keywords && d.keywords.keywords_list, "key");
  wrap(d.fillers && Object.keys(d.fillers.details || {}), "fill");

  el.transcriptEmpty.style.display = "none";
  const seg = document.createElement("div");
  seg.innerHTML = html;
  el.transcript.appendChild(seg);
  el.transcriptScroll.scrollTo({
    top: el.transcriptScroll.scrollHeight,
    behavior: "smooth",
  });
}

function accumulate(target, source) {
  Object.entries(source || {}).forEach(([k, v]) => {
    target[k] = (target[k] || 0) + v;
  });
}

function renderTags(host, map, cls, emptyText) {
  const entries = Object.entries(map || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    host.innerHTML = `<p class="empty">${emptyText}</p>`;
    return;
  }
  host.innerHTML = "";
  entries.forEach(([word, count], i) => {
    const tag = document.createElement("span");
    tag.className = `tag ${cls}`;
    tag.style.animationDelay = `${i * 30}ms`;
    tag.innerHTML = `${escapeHtml(word)} <b>${count}</b>`;
    host.appendChild(tag);
  });
}

/** Neural fields are null unless the models are enabled. */
function neural(d) {
  const rows = [];

  if (d.engines) {
    rows.push(
      `<div class="ai-engines">${Object.entries(d.engines)
        .map(([k, v]) => `${escapeHtml(k)} · ${escapeHtml(v)}`)
        .join("   ")}</div>`
    );
  }

  if (d.emotion) {
    rows.push(row("Tone", escapeHtml(d.emotion.label)));
    rows.push(meter("Vocal tension", d.emotion.tension, d.emotion.tension > 0.5));
  }
  if (d.speech_ratio != null) rows.push(meter("Speech vs silence", d.speech_ratio));
  if (d.pitch_confidence != null) rows.push(meter("Pitch certainty", d.pitch_confidence));

  if (d.prosody) {
    const parts = [];
    if (d.prosody.jitter !== undefined) parts.push(`jitter ${d.prosody.jitter}`);
    if (d.prosody.shimmer !== undefined) parts.push(`shimmer ${d.prosody.shimmer}`);
    if (parts.length) rows.push(row("Voice quality", parts.join(" · ")));
  }

  if (d.disfluency) {
    rows.push(row("Fillers heard / written",
      `${d.disfluency.acoustic_total} / ${d.disfluency.text_total}`));
  }

  if (d.alignment) {
    rows.push(row("Articulation", `${d.alignment.articulation_rate_wpm} wpm`));
    if (d.alignment.pause_count) {
      rows.push(row("Pauses placed", String(d.alignment.pause_count)));
    }
  }

  if (d.speaker_similarity != null) {
    rows.push(meter("Same speaker", d.speaker_similarity, d.speaker_similarity < 0.6));
  }

  if (!rows.length) {
    el.ai.hidden = true;
    return;
  }
  el.ai.hidden = false;
  el.aiRows.innerHTML = rows.join("");
}

const row = (label, value) =>
  `<div class="ai-row"><span>${label}</span><b>${value}</b></div>`;

const meter = (label, value, hot) => {
  const pct = Math.round(clamp01(value) * 100);
  return `<div class="ai-row"><span>${label}</span><b>${pct}%</b>` +
         `<span class="ai-bar"><i class="${hot ? "hot" : ""}" style="width:${pct}%"></i></span></div>`;
};

/** Closing summary in the headline slot. */
function summarise(report) {
  if (!report.analysed_chunks) {
    say("Nothing to score — no speech was detected.", "info");
    el.prompt.textContent = "Press to start listening";
    return;
  }
  const bits = [];
  if (report.avg_confidence != null) bits.push(`${report.avg_confidence} confidence`);
  if (report.avg_wpm) bits.push(`${Math.round(report.avg_wpm)} wpm`);
  bits.push(`${report.total_fillers} filler${report.total_fillers === 1 ? "" : "s"}`);
  bits.push(`${report.total_words} words`);

  say(bits.join("  ·  "), "info");
  el.prompt.textContent = "Press to run it again";
  if (report.avg_confidence != null) {
    voiceprint.update({ score: report.avg_confidence });
    el.scoreNum.textContent = report.avg_confidence;
  }
}

// ── Boot ─────────────────────────────────────────────────────
voiceprint = new Voiceprint($("voiceprint"));
setLive(false);

// Tell the user which engine is actually running, rather than
// claiming privacy the configuration may not provide.
fetch(`${API_BASE}/api/status`)
  .then((r) => (r.ok ? r.json() : null))
  .then((s) => {
    if (!s || !el.footEngine) return;
    el.footEngine.textContent =
      s.stt_engine === "local"
        ? "Running locally — audio never leaves this machine"
        : `Transcribing via ${s.stt_engine} — audio is sent to that provider`;
  })
  .catch(() => {});
