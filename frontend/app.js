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
const TARGET_RATE = 16000;

// ── Utterance segmentation ───────────────────────────────────
// Audio is cut on natural pauses, not on a fixed clock. Whisper was
// trained on 30 s windows; handing it arbitrary 2.5 s slices chops words
// in half and strips the context it needs, which is the single largest
// cause of transcripts that do not resemble what was said.
const SILENCE_RMS = 0.012;      // below this a block counts as quiet
const SILENCE_HOLD_MS = 650;    // quiet for this long ends an utterance
const MIN_UTTERANCE_MS = 700;   // shorter than this is a cough, not speech
// Decoding runs at roughly 1x realtime on CPU, so this also bounds the
// worst-case wait: an 8 s sentence costs about 8 s of transcription.
const MAX_UTTERANCE_MS = 8000;  // force a cut so feedback never stalls
const PREROLL_MS = 300;         // keep audio from just before speech began

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

  // Camera / posture
  camBtn: $("cam-btn"),
  mirror: $("mirror"),
  mirrorFrame: $("mirror-frame"),
  camVideo: $("cam-video"),
  camOverlay: $("cam-overlay"),
  camHint: $("cam-hint"),
  camResTag: $("cam-res-tag"),
  camCalibBanner: $("cam-calib-banner"),
  camCalibText: $("cam-calib-text"),
  mirrorCue: $("mirror-cue"),
  postureRead: $("posture-read"),
  postureNum: $("posture-num"),
  postureBars: $("posture-bars"),
  postureNotes: $("posture-notes"),
  cellPresence: $("cell-presence"),
  valPresence: $("val-presence"),

  // Camera controls
  camResetBtn: $("cam-reset-btn"),
  camBrightness: $("cam-brightness"),
  camBrightnessVal: $("cam-brightness-val"),
  camContrast: $("cam-contrast"),
  camContrastVal: $("cam-contrast-val"),
  camControls: $("cam-controls"),
  camFpsBadge: $("cam-fps-badge"),
  camRecDot: $("cam-rec-dot"),
  camDeviceRow: $("cam-device-row"),
  camDeviceSelect: $("cam-device-select"),

  // Posture ring
  postureRing: $("posture-ring"),
  postureRingFill: $("posture-ring-fill"),
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
let pendingFlush = null;   // sends the in-progress utterance on stop

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

    // Slow clock: accumulate raw samples and cut them on natural pauses.
    processor = audioCtx.createScriptProcessor(4096, 1, 1);

    // The browser may ignore the sample-rate hint, so send what we
    // actually got — the backend resamples rather than assuming.
    const rate = audioCtx.sampleRate;
    const blockMs = (4096 / rate) * 1000;
    const prerollBlocks = Math.max(1, Math.round(PREROLL_MS / blockMs));

    let speech = [];        // audio of the utterance in progress
    let preroll = [];       // rolling buffer of the moments before it
    let quietMs = 0;
    let speaking = false;

    const flush = async () => {
      const samples = new Float32Array(speech);
      speech = [];
      preroll = [];
      quietMs = 0;
      speaking = false;

      if (samples.length < (MIN_UTTERANCE_MS / 1000) * rate) return;
      setStatus("Transcribing");
      const result = await apiAnalyze(encodeWav(samples, rate));
      if (result) render(result);
    };

    processor.onaudioprocess = (e) => {
      if (!recording) return;
      const pcm = e.inputBuffer.getChannelData(0);

      let sum = 0;
      for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
      const rms = Math.sqrt(sum / pcm.length);
      const loud = rms >= SILENCE_RMS;

      if (loud) {
        if (!speaking) {
          // Splice in the pre-roll so the first consonant is not clipped.
          speaking = true;
          preroll.forEach((b) => { for (let i = 0; i < b.length; i++) speech.push(b[i]); });
          preroll = [];
        }
        quietMs = 0;
      }

      if (speaking) {
        for (let i = 0; i < pcm.length; i++) speech.push(pcm[i]);
        if (!loud) quietMs += blockMs;

        const heldMs = (speech.length / rate) * 1000;
        // Cut on a settled pause, or force one so feedback never stalls
        // during a long unbroken stretch.
        if (quietMs >= SILENCE_HOLD_MS || heldMs >= MAX_UTTERANCE_MS) flush();
      } else {
        preroll.push(new Float32Array(pcm));
        if (preroll.length > prerollBlocks) preroll.shift();
      }
    };

    // Send whatever is still buffered when the user stops mid-sentence.
    pendingFlush = flush;

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

  // Don't discard the sentence in progress just because they pressed stop.
  if (pendingFlush) {
    const flush = pendingFlush;
    pendingFlush = null;
    try { await flush(); } catch (_) { /* best effort */ }
  }

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

// ── Camera / posture ─────────────────────────────────────────
let poseTracker = null;

// ── Camera video filter & adjustment state ───────────────────
let camBrightness = 100;  // %
let camContrast   = 100;  // %
let selectedDeviceId = null;

function cameraErrorMessage(err) {
  switch (err?.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Camera permission is blocked. Allow camera access for localhost in your browser settings, then try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera was found. Connect a webcam and try again.";
    case "NotReadableError":
    case "TrackStartError":
      return "The camera is being used by another app. Close that app and try again.";
    case "OverconstrainedError":
      return "That camera mode is unavailable. Check the camera selector and try again.";
    case "InsecureContextError":
      return "Camera access requires http://localhost:8000 or an HTTPS address.";
    case "PoseModelError":
      return "The camera opened, but the posture model could not be downloaded. Check your internet connection and try again.";
    default:
      return "Camera could not start. Check browser permissions and try again.";
  }
}

/** Apply current brightness + contrast as CSS filter on the video. */
function applyCamFilter() {
  if (!el.camVideo) return;
  el.camVideo.style.filter = `brightness(${camBrightness / 100}) contrast(${camContrast / 100})`;
}

/** Show/hide the camera controls panel and icon buttons when camera is on/off. */
function setCamControlsVisible(visible) {
  if (el.camControls)      el.camControls.hidden = !visible;
  if (el.camFpsBadge)      el.camFpsBadge.hidden = !visible;
  if (el.camResTag)        el.camResTag.hidden = !visible;
  if (!visible && el.camCalibBanner) el.camCalibBanner.hidden = true;
}




/** Reset brightness and contrast to defaults. */
function resetAdjustments() {
  camBrightness = 100;
  camContrast = 100;
  if (el.camBrightness)    el.camBrightness.value = 100;
  if (el.camContrast)      el.camContrast.value = 100;
  if (el.camBrightnessVal) el.camBrightnessVal.textContent = "100%";
  if (el.camContrastVal)   el.camContrastVal.textContent = "100%";
  applyCamFilter();
}

/** Enumerate connected video input devices and populate dropdown if >1. */
async function loadCameraDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === "videoinput");
    if (el.camDeviceSelect && el.camDeviceRow) {
      if (videoDevices.length > 1) {
        el.camDeviceSelect.innerHTML = videoDevices
          .map((d, i) => `<option value="${d.deviceId}">${d.label || `Camera ${i + 1}`}</option>`)
          .join("");
        if (selectedDeviceId) el.camDeviceSelect.value = selectedDeviceId;
        el.camDeviceRow.hidden = false;
      } else {
        el.camDeviceRow.hidden = true;
      }
    }
  } catch (err) {
    console.warn("[SpeakTwin] enumerateDevices", err);
  }
}


if (el.camBtn) {
  el.camBtn.addEventListener("click", async () => {
    if (poseTracker && poseTracker.active) {
      poseTracker.stop();
      el.camBtn.textContent = "Turn on camera";
      el.camBtn.setAttribute("aria-pressed", "false");
      el.mirror.classList.remove("on");
      el.mirrorCue.classList.remove("show");
      if (el.camHint) el.camHint.textContent = "Off";
      setCamControlsVisible(false);
      resetAdjustments();
      if (el.mirrorFrame)      el.mirrorFrame.classList.remove("flipped", "is-fullscreen");
      applyCamFilter();
      return;
    }

    if (!poseTracker) {
      poseTracker = new PoseTracker({
        video: el.camVideo,
        canvas: el.camOverlay,
        onBatch: sendPose,
        onStatus: (text) => { if (el.camHint) el.camHint.textContent = text; },
        onGuidance: showCue,
        onFps: (fps) => {
          if (el.camFpsBadge) el.camFpsBadge.textContent = `${fps} fps`;
        },
        onCalibration: ({ active, progress }) => {
          if (el.camCalibBanner) {
            el.camCalibBanner.hidden = !active;
            if (active && el.camCalibText) {
              el.camCalibText.textContent = `Calibrating baseline... ${progress}%`;
            }
          }
        },
        onResolution: (res) => {
          if (el.camResTag) {
            el.camResTag.textContent = res;
            el.camResTag.hidden = false;
          }
        },
      });
    }

    try {
      el.camBtn.disabled = true;
      await poseTracker.start(selectedDeviceId);
      el.camBtn.textContent = "Turn off camera";
      el.camBtn.setAttribute("aria-pressed", "true");
      el.mirror.classList.add("on");
      if (el.camHint) el.camHint.textContent = "Watching";
      setCamControlsVisible(true);
      applyCamFilter();
      loadCameraDevices();
    } catch (err) {
      console.error("[SpeakTwin] camera", err);
      // A model/CDN failure can happen after the camera has opened. Ensure
      // the stream is released so the next attempt is not blocked by this
      // stale request.
      if (poseTracker) poseTracker.stop();
      if (el.camHint) el.camHint.textContent = "Blocked";
      el.postureNotes.innerHTML = `<p class="empty">${cameraErrorMessage(err)}</p>`;
    } finally {
      el.camBtn.disabled = false;
    }
  });
}

// ── Camera control event listeners ───────────────────────────
if (el.camBrightness) {
  el.camBrightness.addEventListener("input", () => {
    camBrightness = Number(el.camBrightness.value);
    if (el.camBrightnessVal) el.camBrightnessVal.textContent = `${camBrightness}%`;
    applyCamFilter();
  });
}
if (el.camContrast) {
  el.camContrast.addEventListener("input", () => {
    camContrast = Number(el.camContrast.value);
    if (el.camContrastVal) el.camContrastVal.textContent = `${camContrast}%`;
    applyCamFilter();
  });
}
if (el.camResetBtn)      el.camResetBtn.addEventListener("click", resetAdjustments);

if (el.camDeviceSelect) {
  el.camDeviceSelect.addEventListener("change", async () => {
    selectedDeviceId = el.camDeviceSelect.value;
    if (poseTracker && poseTracker.active) {
      poseTracker.stop();
      await poseTracker.start(selectedDeviceId);
    }
  });
}

// ── Global Keyboard Shortcuts ─────────────────────────────────
document.addEventListener("keydown", (e) => {
  if (
    e.target.tagName === "INPUT" ||
    e.target.tagName === "TEXTAREA" ||
    e.target.tagName === "SELECT" ||
    e.metaKey ||
    e.ctrlKey ||
    e.altKey
  ) {
    return;
  }
  const key = e.key.toLowerCase();
  if (key === "c") {
    if (el.camBtn) el.camBtn.click();
  }
});

/** The per-frame nudge. Distinct from the server's considered coaching. */
function showCue(cue) {
  if (!el.mirrorCue) return;
  if (!cue || !cue.text) {
    el.mirrorCue.classList.remove("show");
    return;
  }
  el.mirrorCue.textContent = cue.text;
  el.mirrorCue.className = `mirror-cue show ${cue.tone}`;
}

async function sendPose(batch) {
  try {
    const res = await fetch(`${API_BASE}/api/pose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...batch, session_id: sessionId }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderPosture(await res.json());
  } catch (err) {
    console.warn("[SpeakTwin] pose", err);
  }
}

const POSTURE_LABELS = {
  alignment: "Alignment",
  head: "Head position",
  openness: "Openness",
  steadiness: "Steadiness",
};

function renderPosture(p) {
  if (!p) return;

  if (!p.detected) {
    el.postureRead.hidden = true;
    el.postureNotes.hidden = false;
    el.postureNotes.innerHTML =
      `<p class="empty">${escapeHtml(p.message)}</p>`;
    // Reset ring
    if (el.postureRingFill) {
      el.postureRingFill.style.strokeDashoffset = "113";
      el.postureRingFill.className = "posture-ring-fill";
    }
    if (el.postureRing) el.postureRing.hidden = false;
    return;
  }

  el.postureRead.hidden = false;
  el.postureNum.textContent = p.score == null ? "—" : p.score;
  el.postureNum.className =
    "posture-num" + (p.score >= 75 ? " good" : p.score >= 50 ? "" : " poor");

  // Animate the posture score ring
  if (el.postureRingFill && p.score != null) {
    const circumference = 113; // 2π × 18 ≈ 113
    const offset = circumference - (clamp01(p.score / 100) * circumference);
    el.postureRingFill.style.strokeDashoffset = String(Math.round(offset));
    const ringClass = p.score >= 75 ? "good" : p.score >= 50 ? "" : "poor";
    el.postureRingFill.className = `posture-ring-fill${ringClass ? " " + ringClass : ""}`;
    el.postureRing.hidden = false;
  }

  // Only the dimensions the camera could actually see are shown.
  el.postureBars.innerHTML = (p.measured || [])
    .map((key) => {
      const v = p.breakdown[key] ?? 0;
      const tone = v >= 75 ? "good" : v >= 50 ? "" : "poor";
      return `<div class="pbar"><span>${POSTURE_LABELS[key] || key}</span>` +
             `<i><b class="${tone}" style="width:${v}%"></b></i></div>`;
    })
    .join("");

  el.postureNotes.hidden = false;
  const order = { warning: 0, info: 1, success: 2 };
  el.postureNotes.innerHTML = [...(p.feedback || [])]
    .sort((a, b) => (order[a.type] ?? 3) - (order[b.type] ?? 3))
    .slice(0, 4)
    .map((m, i) =>
      `<div class="note ${m.type}" style="animation-delay:${i * 40}ms">` +
      `<span class="note-tag">${escapeHtml(m.category)}</span>` +
      `<span>${escapeHtml(m.text)}</span></div>`)
    .join("");

  if (p.presence_score != null && el.cellPresence) {
    el.cellPresence.hidden = false;
    countTo(el.valPresence, p.presence_score);
  }
}

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
