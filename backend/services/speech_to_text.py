"""
SpeakTwin - Speech-to-Text Service
=====================================
Uses faster-whisper (CTranslate2 backend) for efficient local
speech recognition. The model is loaded once (singleton) to avoid
repeated initialization overhead.

Supports int8 quantization for reduced memory and faster inference.
"""

import os
import io
import threading
import numpy as np # type: ignore
import traceback
import soundfile as sf # type: ignore
from dotenv import load_dotenv # type: ignore

from backend.utils.helpers import get_logger, SAMPLE_RATE # type: ignore

# Initialise logger
logger = get_logger(__name__)

# Load environment for API keys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import typing

_model: typing.Any = None
_model_lock: threading.Lock = threading.Lock()

STT_ENGINE = os.getenv("STT_ENGINE", "local").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger.info(f"STT Engine: {STT_ENGINE} | Groq Key Set: {bool(GROQ_API_KEY)}")

# ---------------------------------------------------------------------------
# API-Based STT Helpers
# ---------------------------------------------------------------------------

def _transcribe_groq(audio: np.ndarray, sr: int) -> dict:
    """Uses Groq's high-speed Whisper Large API."""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}
    
    try:
        from groq import Groq # type: ignore
        client = Groq(api_key=GROQ_API_KEY)

        # Convert numpy array to WAV bytes in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)

        # Groq expects a tuple (filename, file-like object)
        transcription = client.audio.transcriptions.create(
            file=("chunk.wav", buffer),
            model="whisper-large-v3",
            response_format="verbose_json",
            language="en"
        )

        return {
            "text": transcription.text,
            "word_count": len(transcription.text.split()),
            "segments": transcription.segments if hasattr(transcription, 'segments') else []
        }
    except Exception as e:
        logger.error("Groq STT error: %s", e)
        return {"error": str(e)}

def _transcribe_openai(audio: np.ndarray, sr: int) -> dict:
    """Uses OpenAI's official Whisper API."""
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not configured"}

    try:
        from openai import OpenAI # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)

        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)
        buffer.name = "chunk.wav" # Required by OpenAI

        transcription = client.audio.transcriptions.create(
            file=buffer,
            model="whisper-1",
            response_format="verbose_json",
        )

        return {
            "text": transcription.text,
            "word_count": len(transcription.text.split()),
            "segments": transcription.segments if hasattr(transcription, 'segments') else []
        }
    except Exception as e:
        logger.error("OpenAI STT error: %s", e)
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Local STT Fallback (faster-whisper)
# ---------------------------------------------------------------------------

def _load_model():
    """Lazily load the faster-whisper model."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        logger.info("Loading local faster-whisper model (tiny.en)...")
        try:
            from faster_whisper import WhisperModel # type: ignore
            _model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            logger.info("✅ Local Whisper model ready")
        except Exception as e:
            logger.warning("Local Whisper load failed: %s (Check MSVC Redistributable)", e)
            _model = None
    return _model

def _transcribe_local(audio: np.ndarray) -> dict:
    """Local inference using faster-whisper with full isolation."""
    try:
        model = _load_model()
        if model is None:
            return {"error": "Local STT engine unavailable (DLL/Import error). Try setting STT_ENGINE=groq in .env."}

        segments_gen, _ = model.transcribe(audio, beam_size=1, language="en")
        full_text = " ".join([seg.text.strip() for seg in segments_gen])
        return {
            "text": full_text,
            "word_count": len(full_text.split()),
            "segments": []
        }
    except Exception as e:
        logger.error(f"FATAL Local STT Crash: {traceback.format_exc()}")
        return {"error": f"Local STT Engine Error: {str(e)}"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Main transcription entry point (Hybrid).
    Routes to Cloud API or Local Engine based on config.
    """
    # 1. Try Groq (Preferred for Speed/Deployment)
    if STT_ENGINE == "groq" or (STT_ENGINE == "local" and not _load_model() and GROQ_API_KEY):
        res = _transcribe_groq(audio, sr)
        if "error" not in res: return res
        logger.warning("Groq failed, falling back...")

    # 2. Try OpenAI
    if STT_ENGINE == "openai":
        res = _transcribe_openai(audio, sr)
        if "error" not in res: return res
        logger.warning("OpenAI failed, falling back...")

    # 3. Fallback to Local
    return _transcribe_local(audio)
