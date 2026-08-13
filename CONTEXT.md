# SpeakTwin — Working Context

> Living document. Update it when state changes, not on a schedule.
> Full architecture lives in [docs/SpeakTwin-Technical-Guide.pdf](docs/SpeakTwin-Technical-Guide.pdf);
> this file is the *current state and open threads*, not a re-explanation.

**Last updated:** 2026-08-12 · **Version:** 1.1.0 · **Tests:** 158 passing

---

## 1. Where the project stands

| Layer | State | Notes |
|---|---|---|
| Core backend | ✅ Solid | Analysis pipeline, sessions, middleware, schemas — all tested |
| DSP analysis | ✅ Solid | Energy/pitch/pause, all thresholds on the dBFS axis |
| STT | ✅ Working | Hybrid cloud/local; local `tiny.en` requires faster-whisper installed |
| Scoring | ⚠️ Partly fitted | Fitting pipeline exists and runs; corpus can only support 2 of 4 weights — T3 |
| DL layer | ⚠️ Partial | 8 models wired, 5 verified, 1 has no checkpoint, 2 need external access |
| Frontend | ✅ Rebuilt | New visual identity; renders every ML field; panel hidden when models are off |
| Deployment | ✅ Both paths | `Dockerfile` (base, ~1 GB) and `Dockerfile.ml` (~3–4 GB) |
| Datasets | ⚠️ Blocked | 1 of 14 downloaded; the important one needs a request form |

---

## 2. Verified vs. assumed

Anything in this table marked **verified** was actually executed on this machine
(Windows 11, CPU-only, no NVIDIA GPU) — not inferred from documentation.

| Component | Status | Evidence |
|---|---|---|
| CREPE pitch | ✅ verified | 110/220/330 Hz tones tracked within 1% |
| Silero VAD | ✅ verified | Correctly rejects synthetic tone as non-speech |
| Emotion (wav2vec2) | ✅ verified | `ang` at 0.98 → tension 0.987 after label fix |
| ECAPA embeddings | ✅ verified | 192-dim, self-similarity 1.0, cross-source 0.27 |
| openSMILE eGeMAPS | ✅ verified | 88 features, ~1600 ms |
| Acoustic disfluency | ⚠️ logic only | Merge + event collapsing tested; **no checkpoint exists** |
| pyannote diarization | ❌ unverified | Gated model, needs HF token |
| Word alignment | ❌ unverified | Package never installed |

### Measured latency (CPU, warm, 2.5 s chunk budget)

| Component | Cost | Share |
|---|---|---|
| DSP baseline | ~90 ms | 4% |
| CREPE (`tiny`) | ~780 ms | 31% |
| Silero VAD | ~270 ms | 11% |
| Emotion | ~385 ms | 15% |
| ECAPA | ~375 ms | 15% |
| openSMILE | ~1600 ms | 64% |

**All models together ≈ 3400 ms — exceeds the 2500 ms budget.** Roughly two
models fit. CREPE + Silero (~1050 ms) is the pairing that earns its cost.

---

## 3. Open threads

### T1 — Frontend ML rendering  ✅ DONE
Added a "Neural Insights" card (`#ai-panel`) that renders `engines`, `emotion`
+ tension, `speech_ratio`, `pitch_confidence`, `prosody` voice-quality,
acoustic-vs-text filler counts, and located pauses. The card hides itself
entirely when no neural fields are present, so a base install is unchanged.
Also wired the two previously dead DOM refs: `feedback-list` now shows the
rule-based messages (warnings first) and `transcript-toggle` collapses the
transcript.

### T2 — Speaker models reachable  ✅ DONE
- **Per chunk:** `Session.track_speaker()` compares each chunk's ECAPA
  embedding against a session reference. Returns `speaker_similarity` and adds
  a `speaker_changed` warning below `ML_SPEAKER_THRESHOLD` (0.6).
  Counts *transitions*, not frames.
- **Session level:** new `POST /api/diarize` takes a full recording (not a
  chunk — 2.5 s cannot separate voices) and returns pyannote segments.
  Returns 503 with actionable guidance when disabled.

Verified live: same voice → 1.0, different voice → 0.379 + warning, session
`speaker_changes` = 1.

### T3 — Confidence weights  ⚠️ PARTLY RESOLVED
`training/train_confidence_scorer.py` fits the weights against speechocean762
human ratings, keeping the sub-score functions and fitting only the
combination. **Ran it: R² = 0.20 on 400 utterances.**

The honest result is that this corpus supports only half the answer:

| Weight | Verdict |
|---|---|
| `wpm` 0.25 → 0.39 | trustworthy |
| `pitch_variation` 0.25 → 0.38 | trustworthy |
| `energy` | **unreliable** — σ = 0.010, studio recordings all at one level |
| `filler_usage` | **unreliable** — 95% of clips have zero fillers |

The script detects near-constant features and refuses to endorse their
weights, keeping the current values and renormalising. Weights are therefore
**not yet adopted in `helpers.py`** — doing so would import two meaningless
numbers. Needs a spontaneous-speech corpus with delivery ratings; none is
openly available.

### T4 — ML deployment  ✅ DONE
`Dockerfile.ml` adds CPU torch and the model stack, pre-downloading Whisper,
CREPE, and Silero weights into image layers (each guarded so a build-time
network failure degrades to on-demand loading). Base `Dockerfile` unchanged at
~1 GB; the ML image is ~3–4 GB, which is why they are separate.

### T5 — No trained filler detector  🔴 BLOCKED
`disfluency.py` is a complete, tested inference wrapper with no checkpoint.
Blocked on PodcastFillers access (request form).
**Why it matters:** Whisper deletes "um"/"uh" from transcripts, so text-based
filler counting under-counts *structurally*. Only audio recovers it.

### T6 — Gated datasets  🔴 BLOCKED
Needs a Hugging Face account, terms accepted on 4 dataset pages, and `HF_TOKEN`.
Unblocks AMI, TED-LIUM, Common Voice, VoxPopuli, and the pyannote model.

### T7 — Architectural ceilings (not yet worth fixing)
- Single worker only — in-process session store and rate limiter
- No persistence, auth, or cross-session history
- Chunked POST has a hard ~2.5 s latency floor; WebSocket would remove it
- `ScriptProcessorNode` is deprecated in favour of AudioWorklet

---

## 3b. Design language

Concept: **a recording booth at night.** The room is cold blue-black at rest
and warms to tungsten while you speak, so colour encodes state rather than
decorating. `body.live` drives the shift.

| Token | Value | Role |
|---|---|---|
| `--ink` | `#06070e` | The room |
| `--ice` | `#7fd4ff` | At rest |
| `--ember` | `#ff8f4d` | Live / speaking |
| `--mint` / `--amber` / `--rose` | | Good / attention / over |

**Type:** Bricolage Grotesque (display, variable width), Instrument Sans
(body), JetBrains Mono (data). Deliberately not the Inter/Space Grotesk
default.

**Signature element — the voiceprint** (`frontend/voiceprint.js`). One
circular instrument that is simultaneously the button, the live meter, and
the score. Driven by two clocks:

- **60 fps** — radial bars read the browser's own `AnalyserNode`, so the
  instrument responds the instant you make a sound
- **2.5 s** — the outer arc and colour come from the server's analysis

That split is the point. The old UI only moved when the server answered every
2.5 s, which read as broken. Bars also mirror inward at low opacity — the
reflection is the product's premise.

Files: `index.html`, `style.css`, `app.js`, `voiceprint.js`. Still zero build
step, still served straight from the FastAPI static mount.

---

## 4. Decisions worth not re-litigating

| Decision | Reason |
|---|---|
| Capture in the browser, not the server | A deployed server has no microphone |
| Neural results override DSP, code stays wired | Degrade to working, never to an error |
| All ML off by default | Base `pip install` must stay lightweight and valid |
| dBFS not linear RMS for loudness | Linear thresholds swing wildly with mic gain |
| MATTR not TTR or root TTR for clarity | TTR is length-dependent; root TTR under-rewards short chunks |
| Filler counts merge by **max**, not sum | Both detectors see the same event when Whisper keeps an "um" |
| Adaptive silence gate can only open wider | A quiet mic must still transcribe; a noisy room must not |
| No database for sessions | Ephemeral coaching state; reports go to the client as JSON |
| Prompt carryover, not audio overlap | Overlap duplicates words at seams |
| Single Docker worker | In-process session store and rate limiter |

---

## 5. Bugs found and fixed

Each has a regression test.

| Bug | Impact |
|---|---|
| Sample rate read then discarded | 48 kHz input → pitch and WPM wrong by ~3× |
| Pitch peak pinned to search boundary | 20 Hz desk rumble reported as 400 Hz |
| Session pruned before insert | Store settled one over `max_sessions` |
| Emotion labels only long-spelled | 98%-angry audio scored `tension: 0.003` |
| SpeechBrain symlinks on Windows | `WinError 1314`, ECAPA unusable |
| `registry` instance shadowed `registry` module | Module unreachable by dotted path |
| `or 1.0` on a similarity of 0.0 | Falsy zero made every chunk look like a new speaker |
| Canvas intro driven by frame count | Animation speed tied to frame rate; stalled where frames were scarce. Now time-based |
| Canvas measured once at construction | Box is still 0 while fonts load. Now a `ResizeObserver` |
| Grid/flex children at `min-width:auto` | Refused to shrink, so narrow viewports scrolled sideways |
| Traceback in error responses | Leaked file paths and internals |
| CORS `*` + credentials | Invalid per spec, rejected by browsers |
| Transcript injected unescaped | HTML injection via STT output |
| Blocking work on the event loop | One request stalled the whole process |
| No LLM timeout | A slow provider could hang a request indefinitely |
| Static assets 404 at `/` | Frontend mounted only at `/static` |

---

## 6. Environment

- **Platform:** Windows 11, CPU-only (no NVIDIA GPU)
- **Python:** 3.11.9, venv at `./venv`
- **Installed beyond base:** torch 2.13.0+cpu, transformers 5.15.0,
  datasets 5.0.1, torchcrepe 0.0.24, opensmile, speechbrain 1.1.0
- **Not installed:** pyannote.audio, whisperx / whisper-timestamped
- **`HF_TOKEN`:** not set — no account has been used anywhere in this project
- **Known annoyance:** `~/.bashrc` has a UTF-16 BOM, so every bash call prints
  `$'\377\376eval': command not found`. Harmless; re-save as UTF-8 to fix.

### Data on disk
| Dataset | Size | State |
|---|---|---|
| speechocean762 | 1.2 GB | ✅ extracted, human scores present |
| everything else | — | not downloaded (see T6) |

---

## 7. Commands

```bash
# Run
uvicorn backend.main:app --reload --port 8000     # http://localhost:8000

# Test
./venv/Scripts/python.exe -m pytest                # 146 tests

# Datasets
python scripts/datasets.py check                   # what's ready vs blocked
python scripts/datasets.py list --open-only

# Regenerate the PDF guide
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new \
  --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="docs/SpeakTwin-Technical-Guide.pdf" \
  "file:///C:/Users/Admin/AI-MIRROR/docs/speaktwin-technical-guide.html"
```

---

## 8. Session log

**2026-08-12 — Hardening pass.** Fixed 12 defects (see §5). Added config layer,
Pydantic schemas, middleware, session store, audio I/O with resampling, 146
tests, Docker, CI, README rewrite.

**2026-08-12 — DL layer.** Added 8 optional models behind a lazy registry.
Verified 5 on real audio. Added dataset manager (14 corpora), filler-detector
training pipeline, DATASETS.md.

**2026-08-12 — Technical guide.** 33-page PDF covering architecture, every
module, measured latency, technology choices, and limitations.

**2026-08-12 — Loose ends closed.** T1, T2, T4 done; T3 partly. Added the
Neural Insights UI card, speaker continuity tracking + `/api/diarize`,
`train_confidence_scorer.py` (run: R² 0.20, 2 of 4 weights trustworthy),
`Dockerfile.ml`, LICENSE, and this file. Wired the two dead DOM refs. Found
and fixed a falsy-zero bug in speaker-change counting. 146 → 158 tests.

**2026-08-12 — UI rebuild.** New visual identity (see §3b): the voiceprint
instrument, warm/cold room, Bricolage + Instrument Sans, bento readout rail,
transcript with keyword/filler marks, neural panel. Verified with headless
screenshots at 1440 / 760 / 560 px. Fixed three animation and layout bugs
found by looking at the render rather than assuming it worked.

> Screenshot note: headless Chrome will not render a window narrower than
> ~500 px — it lays out wider and clips the capture. Sub-500 px checks need
> real device emulation; 560 px renders cleanly.

**Next up:** T5 and T6 are the blockers — both need external access (see §3).
Everything else buildable is done. The remaining unblocked work is T7-class
architecture (WebSocket, Redis, AudioWorklet), which is not yet worth doing at
this scale.
