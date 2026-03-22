# 🎙️ SpeakTwin – AI Communication Mirror

> Real-time AI-powered speech analysis and coaching feedback.

SpeakTwin captures your microphone input, analyzes your speaking patterns in real-time, and provides actionable feedback to help you become a better communicator.

![SpeakTwin](https://img.shields.io/badge/SpeakTwin-v1.0-6c5ce7?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Real-time Audio Analysis** | Captures microphone input, extracts energy (volume), pitch (frequency), and pause detection |
| 🗣️ **Speech-to-Text** | Uses faster-whisper (tiny.en, int8) for efficient local transcription |
| 🔍 **Filler Word Detection** | Detects "um", "uh", "like", "basically", "you know" and more |
| 💬 **Smart Feedback Engine** | Generates real-time coaching messages based on thresholds |
| 📊 **Confidence Score** | Composite 0-100 score combining WPM, pitch variation, energy, and fluency |
| 🎨 **Premium Dark UI** | Glassmorphism design with animated meters, aurora background, and micro-animations |

---

## 🏗️ Project Structure

```
AI-MIRROR/
│
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── routes/
│   │   └── analyze.py             # API endpoints (/start, /stop, /analyze, /status)
│   ├── services/
│   │   ├── audio_capture.py       # Microphone input (singleton, non-blocking)
│   │   ├── audio_analysis.py      # Energy, pitch, pause extraction
│   │   ├── speech_to_text.py      # faster-whisper STT
│   │   ├── filler_detection.py    # Filler word scanner
│   │   ├── feedback_engine.py     # Coaching message generator
│   │   ├── confidence_score.py    # Composite scoring
│   │   └── camera_placeholder.py  # 🔮 Future: MediaPipe body language
│   └── utils/
│       └── helpers.py             # Config, thresholds, logger
│
├── frontend/
│   ├── index.html                 # Semantic HTML layout
│   ├── style.css                  # Dark theme + glassmorphism
│   └── app.js                     # Async polling + UI animations
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Microphone** connected to your system
- **Windows / macOS / Linux**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AI-MIRROR.git
cd AI-MIRROR

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the server (from the project root)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser at: **http://localhost:8000**

---

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/start` | Start microphone recording |
| `POST` | `/api/stop` | Stop microphone recording |
| `GET`  | `/api/status` | Check recording status |
| `GET`  | `/api/analyze` | Get latest analysis results |

### Sample Response: `GET /api/analyze`

```json
{
    "message": "Great vocal variety! Your speech sounds expressive.",
    "pitch": 185.42,
    "pitch_std": 38.7,
    "energy": 0.0423,
    "wpm": 142.0,
    "fillers": {
        "total_fillers": 1,
        "filler_rate": 0.0167,
        "details": { "like": 1 }
    },
    "transcript": "I think this project is really great and like very innovative",
    "confidence_score": 82,
    "confidence_breakdown": {
        "wpm": 100,
        "pitch_variation": 85,
        "energy": 100,
        "filler_usage": 89
    },
    "feedback": [...],
    "status": "excellent"
}
```

---

## ⚡ Performance

- **Model loading**: Whisper model loaded once at startup (singleton pattern)
- **Quantization**: int8 inference for minimal memory and fast processing
- **Audio chunking**: 2.5-second chunks for sub-3s latency
- **Non-blocking**: Async audio capture with threading
- **Polling**: ~2.8s interval matching chunk duration

---

## 🔮 Roadmap (Future Features)

- [ ] **Camera Module** – MediaPipe-based posture & eye contact tracking
- [ ] **Real-time Video Overlay** – Visual feedback on webcam feed
- [ ] **Multimodal Fusion** – Combined audio + video confidence score
- [ ] **Session History** – Track improvement over time
- [ ] **Export Reports** – PDF/JSON session summaries
- [ ] **Custom Thresholds** – User-configurable feedback sensitivity

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Audio Capture | sounddevice |
| Audio Analysis | librosa + numpy + scipy |
| Speech-to-Text | faster-whisper (CTranslate2) |
| Frontend | Vanilla HTML/CSS/JS |
| Design | Glassmorphism, Inter + JetBrains Mono |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>SpeakTwin</strong> – Speak better, speak smarter 🎙️
</p>
