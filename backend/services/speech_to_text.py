"""
SpeakTwin - Speech-to-Text Service
===================================
Hybrid transcription: a cloud Whisper API when a key is configured, with
faster-whisper (CTranslate2, int8) as the local fallback. The local model is
loaded once behind a lock and reused.

Engine selection comes from config (`STT_ENGINE`), and `auto` resolves to
the best available option rather than the old chained condition that
re-entered model loading from inside the routing test.

Concurrency note: `WhisperModel.transcribe` is not safe to call from several
threads at once, and this service runs inside FastAPI's threadpool, so
inference is serialised behind `_inference_lock`.
"""

from __future__ import annotations

import io
import threading
import traceback
from typing import Any, Dict, Optional

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore

from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import get_logger, SAMPLE_RATE  # type: ignore

logger = get_logger(__name__)

_model: Any = None
_model_load_attempted = False
_model_lock = threading.Lock()       # guards loading
_inference_lock = threading.Lock()   # guards decoding

_settings = get_settings()
logger.info(
    "STT engine: %s (resolved: %s) | groq_key=%s openai_key=%s",
    _settings.stt_engine, _settings.resolved_stt_engine,
    bool(_settings.groq_api_key), bool(_settings.openai_api_key),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_float32(audio: np.ndarray) -> np.ndarray:
    """Whisper expects contiguous float32; soundfile can hand us float64."""
    return np.ascontiguousarray(audio, dtype=np.float32)


def _to_wav_bytes(audio: np.ndarray, sr: int) -> io.BytesIO:
    """Encode a float array as an in-memory WAV for the cloud APIs."""
    buffer = io.BytesIO()
    sf.write(buffer, _as_float32(audio), sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer


def _result(text: str, segments: Any = None) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    return {
        "text": cleaned,
        "word_count": len(cleaned.split()),
        "segments": segments or [],
    }


# ---------------------------------------------------------------------------
# Cloud engines
# ---------------------------------------------------------------------------
def _transcribe_groq(audio: np.ndarray, sr: int,
                     initial_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Groq's hosted Whisper (fast, large-v3 quality)."""
    settings = get_settings()
    if not settings.groq_api_key:
        return {"error": "GROQ_API_KEY not configured"}

    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=settings.groq_api_key,
                      timeout=settings.llm_timeout_seconds * 3,
                      max_retries=1)

        kwargs: Dict[str, Any] = {
            "file": ("chunk.wav", _to_wav_bytes(audio, sr)),
            "model": settings.groq_model,
            "response_format": "verbose_json",
            "language": "en",
        }
        if initial_prompt:
            kwargs["prompt"] = initial_prompt

        transcription = client.audio.transcriptions.create(**kwargs)
        return _result(transcription.text, getattr(transcription, "segments", None))

    except Exception as exc:
        logger.error("Groq STT error: %s", exc)
        return {"error": str(exc)}


def _transcribe_openai(audio: np.ndarray, sr: int,
                       initial_prompt: Optional[str] = None) -> Dict[str, Any]:
    """OpenAI's hosted Whisper."""
    settings = get_settings()
    if not settings.openai_api_key:
        return {"error": "OPENAI_API_KEY not configured"}

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=settings.openai_api_key,
                        timeout=settings.llm_timeout_seconds * 3,
                        max_retries=1)

        buffer = _to_wav_bytes(audio, sr)
        buffer.name = "chunk.wav"  # the SDK infers the format from this

        kwargs: Dict[str, Any] = {
            "file": buffer,
            "model": settings.openai_model,
            "response_format": "verbose_json",
            "language": "en",
        }
        if initial_prompt:
            kwargs["prompt"] = initial_prompt

        transcription = client.audio.transcriptions.create(**kwargs)
        return _result(transcription.text, getattr(transcription, "segments", None))

    except Exception as exc:
        logger.error("OpenAI STT error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Local engine (faster-whisper)
# ---------------------------------------------------------------------------
def _load_model():
    """Load the faster-whisper model once. Returns None if unavailable."""
    global _model, _model_load_attempted

    if _model is not None or _model_load_attempted:
        return _model

    with _model_lock:
        if _model is not None or _model_load_attempted:
            return _model

        settings = get_settings()
        _model_load_attempted = True
        logger.info("Loading local faster-whisper model (%s, %s/%s)...",
                    settings.whisper_model, settings.whisper_device,
                    settings.whisper_compute_type)
        try:
            from faster_whisper import WhisperModel  # type: ignore

            _model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            logger.info("Local Whisper model ready")
        except Exception as exc:
            logger.warning(
                "Local Whisper load failed: %s "
                "(on Windows this is usually a missing MSVC redistributable). "
                "Set GROQ_API_KEY or OPENAI_API_KEY to use a cloud engine.",
                exc,
            )
            _model = None

    return _model


def local_model_available() -> bool:
    """Whether the local engine loaded, loading it if it has not been tried."""
    return _load_model() is not None


def model_state() -> Dict[str, bool]:
    """
    Non-blocking view of the local model, for health checks.

    Deliberately does not trigger a load - a probe should never be the
    thing that spends 10 seconds pulling model weights.
    """
    return {"loaded": _model is not None, "load_attempted": _model_load_attempted}


def engine_ready() -> bool:
    """Whether the resolved engine has what it needs to transcribe."""
    settings = get_settings()
    engine = settings.resolved_stt_engine
    if engine == "groq":
        return bool(settings.groq_api_key)
    if engine == "openai":
        return bool(settings.openai_api_key)
    return _model is not None


def _run_local(model, audio: np.ndarray, initial_prompt: Optional[str],
               use_vad: bool):
    """One faster-whisper decode pass."""
    settings = get_settings()

    segments_gen, _info = model.transcribe(
        audio,
        beam_size=settings.whisper_beam_size,
        language=settings.whisper_language,
        vad_filter=use_vad,
        # Prevents the decoder from looping on its own previous output,
        # which is the classic cause of repeated-phrase hallucinations.
        condition_on_previous_text=False,
        initial_prompt=initial_prompt or None,
        # Temperature fallback: retry a segment at rising temperature when
        # the greedy result trips one of the guards below. Whisper's own
        # reference implementation does this; skipping it is a large part
        # of why short-chunk transcription reads as gibberish.
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        no_speech_threshold=settings.whisper_no_speech_threshold,
        log_prob_threshold=settings.whisper_logprob_threshold,
        compression_ratio_threshold=settings.whisper_compression_threshold,
    )
    return " ".join(seg.text.strip() for seg in segments_gen)


def _transcribe_local(audio: np.ndarray,
                      initial_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Local inference, fully isolated so a crash cannot take down a request."""
    model = _load_model()
    if model is None:
        return {"error": "Local STT engine unavailable. "
                         "Configure GROQ_API_KEY or OPENAI_API_KEY."}

    settings = get_settings()
    audio = _as_float32(audio)

    try:
        with _inference_lock:
            try:
                text = _run_local(model, audio, initial_prompt,
                                  settings.whisper_vad_filter)
            except Exception as vad_exc:
                if not settings.whisper_vad_filter:
                    raise
                # The bundled Silero VAD needs onnxruntime; fall back rather
                # than failing the whole request if it is missing.
                logger.warning("VAD pass failed (%s), retrying without VAD", vad_exc)
                text = _run_local(model, audio, initial_prompt, False)
        return _result(text)

    except Exception as exc:
        logger.error("Local STT crash:\n%s", traceback.format_exc())
        return {"error": f"Local STT engine error: {exc}"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def transcribe(audio: np.ndarray, sr: int = SAMPLE_RATE,
               initial_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe one audio chunk.

    `initial_prompt` carries the tail of the previous chunk's transcript,
    which gives the decoder context across chunk boundaries and noticeably
    reduces words being mangled at the seams.

    Returns {"text", "word_count", "segments"} or {"error": ...}.
    """
    settings = get_settings()
    engine = settings.resolved_stt_engine
    errors = []

    if engine == "groq":
        result = _transcribe_groq(audio, sr, initial_prompt)
        if "error" not in result:
            return result
        errors.append(f"groq: {result['error']}")
        logger.warning("Groq STT failed, falling back to local")

    elif engine == "openai":
        result = _transcribe_openai(audio, sr, initial_prompt)
        if "error" not in result:
            return result
        errors.append(f"openai: {result['error']}")
        logger.warning("OpenAI STT failed, falling back to local")

    result = _transcribe_local(audio, initial_prompt)
    if "error" in result and errors:
        result["error"] = "; ".join(errors + [f"local: {result['error']}"])
    return result
