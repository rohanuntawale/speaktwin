# 🎙️ SpeakTwin – AI Communication Mirror

> Real-time AI-powered speech analysis and coaching feedback.

SpeakTwin captures your microphone input in the browser, analyzes your speaking patterns chunk by chunk, and provides actionable feedback to help you become a better communicator.

![SpeakTwin](https://img.shields.io/badge/SpeakTwin-v1.1-6c5ce7?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Real-time Audio Analysis** | Energy (dBFS), pitch via windowed autocorrelation with voiced gating and octave correction, and pause detection |
| 🗣️ **Hybrid Speech-to-Text** | Groq or OpenAI Whisper when a key is configured, faster-whisper (`tiny.en`, int8) locally otherwise |
| 🔍 **Context-Aware Filler Detection** | "um" and "uh" always count; "so", "like", "well", "right" only count when used as fillers |
| 🎯 **Keyword Detection** | Positive reinforcement for impactful language, including multi-word phrases |
| 📖 **Clarity Score** | MATTR-based lexical variety, stable as the transcript grows |
| 💬 **Coaching Feedback** | Threshold rules, optionally sharpened by an LLM insight line |
| 📊 **Confidence Score** | Composite 0-100 from WPM, pitch variation, loudness, and fluency |
| 📈 **Sessions** | Cross-chunk totals, smoothed trends, decoder context carryover, and exportable reports |
| 🎨 **Premium Dark UI** | Glassmorphism design with animated meters, aurora background, and micro-animations |

---

## 🏗️ Project Structure

```
AI-MIRROR/
│
├── backend/
│   ├── main.py                    # FastAPI entry point, CORS, static, health
│   ├── middleware.py              # Request IDs, rate limiting, API-key auth
│   ├── schemas.py                 # Pydantic response models
│   ├── routes/
│   │   └── analyze.py             # /analyze + session endpoints
│   ├── services/
│   │   ├── audio_io.py            # Decode → mono → float32 → resample
│   │   ├── audio_analysis.py      # Energy, pitch, pause extraction
│   │   ├── speech_to_text.py      # Hybrid cloud/local Whisper
│   │   ├── filler_detection.py    # Context-aware filler scanner
│   │   ├── keyword_detection.py   # Target keyword scanner
│   │   ├── clarity_analysis.py    # MATTR-based clarity scoring
│   │   ├── feedback_engine.py     # Coaching message generator
│   │   ├── confidence_score.py    # Composite scoring
│   │   ├── session_store.py       # Session state, smoothing, reports
│   │   ├── audio_capture.py       # ⚠️ Legacy server-side mic (unused)
│   │   ├── camera_placeholder.py  # 🔮 Future: MediaPipe body language
│   │   └── ml/                    # 🧠 Optional deep learning layer
│   │       ├── registry.py        #    Lazy loading, device, status
│   │       ├── enrichment.py      #    Orchestrates all models per chunk
│   │       ├── pitch.py           #    CREPE
│   │       ├── vad.py             #    Silero
│   │       ├── disfluency.py      #    Acoustic filler detection
│   │       ├── emotion.py         #    wav2vec2 SER
│   │       ├── speaker.py         #    ECAPA embeddings + pyannote
│   │       ├── prosody.py         #    openSMILE eGeMAPS
│   │       └── alignment.py       #    Word-level timestamps
│   └── utils/
│       ├── config.py              # All environment settings
│       ├── helpers.py             # Thresholds, logger
│       └── text.py                # Shared tokenizer
│
├── frontend/
│   ├── index.html                 # Semantic HTML layout
│   ├── style.css                  # Dark theme + glassmorphism
│   └── app.js                     # Web Audio capture + UI animations
│
├── scripts/
│   ├── datasets.py                # Dataset manager CLI
│   └── dataset_registry.py        # 14 corpora: size, licence, access
├── training/
│   └── train_filler_detector.py   # Fine-tune the acoustic filler model
│
├── tests/                         # pytest suite (146 tests)
├── data/                          # Downloaded corpora (gitignored)
├── Dockerfile
├── .env.example
├── DATASETS.md                    # Dataset & training guide
├── requirements.txt               # Runtime
├── requirements-ml.txt            # + torch, transformers, the models
├── requirements-dev.txt           # + pytest, httpx
└── requirements-legacy.txt        # + sounddevice, librosa (prototypes only)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Microphone** and a browser that allows mic access on your origin
- **Windows / macOS / Linux**

### Installation

```bash
git clone https://github.com/yourusername/AI-MIRROR.git
cd AI-MIRROR

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Optional: start from the documented config surface
cp .env.example .env
```

### Running

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# or
python -m backend.main
```

Open **http://localhost:8000**. Interactive API docs are at **/docs**.

For the posture-data download, synthetic trainer, and the complete PowerShell
command list, see [docs/POSTURE_TRAINING.md](docs/POSTURE_TRAINING.md).

> Browsers only grant microphone access on `localhost` or over HTTPS. Serving the page from `file://` or a plain-HTTP LAN address will fail the permission check.

### Docker

```bash
# Base image (~1 GB) — DSP pipeline + Whisper, no neural extras
docker build -t speaktwin .
docker run -p 8000:8000 --env-file .env speaktwin

# With the deep learning layer (~3–4 GB) — CPU torch + all models
docker build -f Dockerfile.ml -t speaktwin:ml .
docker run -p 8000:8000 --env-file .env speaktwin:ml
```

The base image installs only `requirements.txt`, so every `ML_*_ENABLED` flag
is inert there. Use `Dockerfile.ml` if you want the neural layer in a container.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## ⚙️ Configuration

Every setting has a working default — an empty `.env` runs local-only with no cloud calls. See [.env.example](.env.example) for the full list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` / `HOST` | `8000` / `0.0.0.0` | Bind address |
| `DEBUG` | `false` | Include error detail in responses |
| `LOG_LEVEL` | `INFO` | Logger verbosity |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `STT_ENGINE` | `auto` | `auto` \| `local` \| `groq` \| `openai` |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | – | Enables the matching cloud engine |
| `WHISPER_MODEL` | `tiny.en` | Local faster-whisper model |
| `WHISPER_VAD_FILTER` | `true` | Trim silence before decoding |
| `OPENROUTER_API_KEY` | – | Enables the LLM coaching line |
| `LLM_MIN_INTERVAL_SECONDS` | `8` | Minimum gap between LLM calls per session |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size cap |
| `MAX_AUDIO_SECONDS` | `30` | Decoded duration cap |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-client cap; `0` disables |
| `API_KEY` | – | When set, `/api` requires `X-API-Key` |
| `SESSION_TTL_SECONDS` | `3600` | Idle session expiry |
| `SMOOTHING_ALPHA` | `0.4` | EMA factor for smoothed metrics |

**Before deploying:** set `CORS_ORIGINS` to your real origin, and set `API_KEY` if the instance is reachable publicly — `/api/analyze` can spend money at a cloud provider on every call.

---

## 🎯 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Analyze one audio chunk (multipart: `audio_file`, optional `session_id`) |
| `POST` | `/api/diarize` | Segment a **full recording** by speaker — 2.5 s is far too short to separate voices, so this is deliberately separate from `/analyze`. Needs `ML_DIARIZATION_ENABLED` |
| `POST` | `/api/session` | Open a session |
| `GET` | `/api/session/{id}` | Rolling totals and smoothed averages |
| `GET` | `/api/session/{id}/report` | Full report including the stitched transcript |
| `DELETE` | `/api/session/{id}` | Close a session, returning its final report |
| `GET` | `/api/status` | Lightweight readiness probe |
| `GET` | `/api/health` | Deep health check: engine, model, LLM, sessions |

Audio may be any format `libsndfile` reads (WAV, FLAC, OGG) at any sample rate — it is resampled to 16 kHz server-side.

### Sample Response: `POST /api/analyze`

```json
{
    "message": "Great vocal variety! Your speech sounds expressive.",
    "pitch": 185.42,
    "pitch_std": 38.7,
    "voiced_ratio": 0.82,
    "energy": 0.0423,
    "energy_db": -27.47,
    "wpm": 142.0,
    "transcript": "I think this project is really great and like very innovative",
    "fillers": {
        "total_fillers": 1,
        "filler_rate": 0.0909,
        "details": { "like": 1 },
        "total_words": 11
    },
    "keywords": {
        "total_keywords": 2,
        "found_keywords": { "project": 1, "innovative": 1 },
        "keywords_list": ["project", "innovative"]
    },
    "clarity": 73,
    "lexical_diversity": 0.91,
    "confidence_score": 82,
    "confidence_breakdown": {
        "wpm": 100, "pitch_variation": 85, "energy": 100, "filler_usage": 43
    },
    "confidence_smoothed": 79,
    "feedback": [
        { "text": "Volume is perfect.", "type": "success", "category": "energy" }
    ],
    "status": "excellent",
    "pause_ratio": 0.12,
    "longest_pause_sec": 0.29,
    "degraded": false,
    "warnings": [],
    "source_sample_rate": 48000,
    "session_id": "9f2c...",
    "session": { "chunk_count": 4, "total_fillers": 3, "avg_confidence": 79 }
}
```

**`degraded`** is `true` when part of the pipeline failed but the response is still usable — for example, transcription was unavailable while acoustic metrics remain valid. `warnings` says which part (`stt_unavailable`, `silence_skipped`, `confidence_acoustic_only`, `session_not_found`).

---

## 🧠 Deep Learning Layer

Optional neural models, all **off by default**. Install with
`pip install -r requirements-ml.txt` and enable individually.

| Model | What it does | Setting |
|---|---|---|
| **CREPE** | Neural pitch tracking; beats autocorrelation on real speech | `ML_PITCH_ENABLED` |
| **Silero VAD** | Real speech/silence detection instead of an energy threshold | `ML_VAD_ENABLED` |
| **Filler detector** | Finds `um`/`uh` **in the audio** — see below | `ML_DISFLUENCY_ENABLED` |
| **wav2vec2 SER** | Emotion and vocal tension | `ML_EMOTION_ENABLED` |
| **ECAPA-TDNN** | Speaker embeddings for continuity across sessions | `ML_SPEAKER_ENABLED` |
| **pyannote** | Speaker diarization (multi-speaker practice) | `ML_DIARIZATION_ENABLED` |
| **openSMILE** | 88 eGeMAPS prosody features (jitter, shimmer, HNR) | `ML_PROSODY_ENABLED` |
| **WhisperX** | Word-level timestamps → located pauses, articulation rate | `ML_ALIGNMENT_ENABLED` |

Neural results **override** the DSP metrics when enabled; the signal-processing
path stays wired as the fallback, so a missing package costs one field rather
than the response. `GET /api/health` reports per-model load state and any load
error. Which engine produced each metric is returned in `engines`.

### Why the acoustic filler detector matters

Whisper was trained largely on cleaned transcripts and **routinely deletes
"um" and "uh"** — and its VAD filter trims the hesitation regions too. So
counting fillers in the transcript under-counts structurally, and no regex
improvement fixes it. The evidence only survives in the waveform.

No suitable public checkpoint exists, so
[training/train_filler_detector.py](training/train_filler_detector.py)
fine-tunes one on PodcastFillers. The backend merges acoustic and text counts
by taking the **maximum** per label, not the sum — when Whisper does keep an
"um", both detectors see the same event.

### Latency budget — measured, not estimated

CPU (no GPU), 2.5 s chunks, after warmup. **The budget is 2500 ms per chunk.**

| Component | Warm cost | Share of budget |
|---|---|---|
| DSP baseline (no ML) | ~90 ms | 4% |
| CREPE pitch (`tiny`) | ~780 ms | 31% |
| Silero VAD | ~270 ms | 11% |
| Emotion (wav2vec2) | ~385 ms | 15% |
| Speaker embedding (ECAPA) | ~375 ms | 15% |
| **openSMILE eGeMAPS** | **~1600 ms** | **64%** |

**You cannot enable everything at once on CPU** — the sum is ~3.4 s against a
2.5 s budget. Pick deliberately. The pairing that earns its cost is
**CREPE + Silero (~1050 ms, 42%)**: better pitch plus honest speech detection.

openSMILE is the outlier. Run it offline to build training features
(`ML_PROSODY_FULL_VECTOR=true`) rather than on the live path.

First call per model is far slower — weights download and load (CREPE ~3.6 s,
ECAPA ~31 s, emotion ~87 s, Silero ~176 s from GitHub). `ML_WARMUP=true`
(default) pays this at startup instead of inside the first request.

A GPU changes all of this (`ML_DEVICE=cuda`).

Diarization and speaker verification are **session-level**, not per-chunk:
2.5 s is too short to separate voices reliably.

### Windows notes

Two platform-specific issues are handled in code, both worth knowing about:

- **SpeechBrain symlinks.** It links model files out of the HF cache, and
  Windows refuses without Developer Mode (`WinError 1314`). The loader
  requests real copies instead.
- **Emotion label vocabularies differ by checkpoint.** SUPERB/IEMOCAP models
  emit `ang`/`hap`/`neu`/`sad`; RAVDESS-trained ones emit full words. Both are
  mapped, and an unrecognised label is reported in `unmapped_labels` rather
  than silently scoring a furious delivery as calm.

---

## 📚 Datasets & Training

See **[DATASETS.md](DATASETS.md)** for the full guide.

```bash
python scripts/datasets.py list      # 14 corpora, with size and licence
python scripts/datasets.py check     # what's ready vs. blocked on you
python scripts/datasets.py download speechocean762
```

Four corpora sit behind a click-through licence agreement (Common Voice,
TED-LIUM, AMI, VoxPopuli via Hugging Face; PodcastFillers and IEMOCAP via
request forms). Accepting a licence is a legal act under your own name, so it
can't be scripted — `check` prints exactly which ones are waiting on you.
Everything after that acceptance is automated.

Starting points: **PodcastFillers** (filler detection — the one that matters),
**speechocean762** (open, 409 MB, real human delivery ratings — supervision for
replacing the hand-weighted confidence score), **AMI** (CC BY, commercial-safe),
**TED-LIUM 3** (450 h of actual public speaking).

---

## ⚡ Performance & Design Notes

- **Model loading** – Whisper loaded once at startup behind a lock; inference is serialised because `WhisperModel` is not thread-safe
- **Quantization** – int8 inference for minimal memory and fast processing
- **Non-blocking** – all CPU-bound work runs in the threadpool, never on the event loop
- **Audio chunking** – 2.5s chunks; the previous chunk's transcript tail is passed as decoder context so words are not mangled at the seams
- **Loudness in dBFS** – log-scale thresholds degrade gracefully across microphone gains, with a per-session adaptive silence gate that can only ever open wider than the absolute threshold, never narrower
- **Smoothing** – per-chunk scores are noisy at 2.5s granularity, so sessions expose EMA-smoothed values

### Scaling

The session store and rate limiter are in-process, so the container runs a single worker. Running multiple workers or replicas requires moving both to a shared backend such as Redis.

---

## 🔮 Roadmap

- [x] **Session History** – cross-chunk aggregation and smoothing
- [x] **Export Reports** – JSON session summaries
- [x] **Custom Thresholds** – configurable via environment
- [ ] **Camera Module** – MediaPipe-based posture & eye contact tracking
- [ ] **Real-time Video Overlay** – visual feedback on webcam feed
- [ ] **Multimodal Fusion** – combined audio + video confidence score
- [ ] **WebSocket transport** – below the ~2.5s latency floor of chunked POST
- [ ] **Word-level timestamps** – precise pause placement and per-word highlighting

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Audio Capture | Web Audio API (browser) |
| Audio Decoding | soundfile + scipy |
| Audio Analysis | numpy + scipy |
| Speech-to-Text | faster-whisper (CTranslate2) / Groq / OpenAI |
| Coaching Insight | OpenRouter (optional) |
| Frontend | Vanilla HTML/CSS/JS |
| Design | Glassmorphism, Inter + JetBrains Mono |

---

## 📦 Legacy Files

`server.py`, `audio_engine.py`, `static/`, and `backend/services/audio_capture.py` are the original prototype: they captured audio server-side with `sounddevice` and estimated pitch with `librosa`. That approach cannot work on a server with no microphone, so capture moved to the browser. The files are kept for reference and are not imported by the running app; their extra dependencies live in `requirements-legacy.txt`.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>SpeakTwin</strong> – Speak better, speak smarter 🎙️
</p>
