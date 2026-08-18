"""
SpeakTwin - Application Configuration
=====================================
Single source of truth for every environment-derived setting.

`.env` is loaded exactly once, resolved against the project root, so the
server behaves identically no matter which directory it was launched from.
Import `get_settings()` instead of reading `os.environ` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv  # type: ignore

# ---------------------------------------------------------------------------
# Paths - resolved from this file, never from the working directory
# ---------------------------------------------------------------------------
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(UTILS_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

load_dotenv(ENV_PATH)

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

VALID_STT_ENGINES = ("auto", "local", "groq", "openai")


# ---------------------------------------------------------------------------
# Typed environment readers
# ---------------------------------------------------------------------------
def _raw(key: str) -> Optional[str]:
    value = os.getenv(key)
    return value.strip() if value is not None else None


def _str(key: str, default: str = "") -> str:
    return _raw(key) or default


def _opt(key: str) -> Optional[str]:
    return _raw(key) or None


def _bool(key: str, default: bool) -> bool:
    value = _raw(key)
    if not value:
        return default
    lowered = value.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    return default


def _int(key: str, default: int, minimum: Optional[int] = None,
         maximum: Optional[int] = None) -> int:
    value = _raw(key)
    try:
        parsed = int(value) if value else default
    except ValueError:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _float(key: str, default: float, minimum: Optional[float] = None,
           maximum: Optional[float] = None) -> float:
    value = _raw(key)
    try:
        parsed = float(value) if value else default
    except ValueError:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _list(key: str, default: List[str]) -> List[str]:
    value = _raw(key)
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    # -- Server ------------------------------------------------------------
    host: str
    port: int
    debug: bool
    log_level: str
    cors_origins: List[str]

    # -- Request limits ----------------------------------------------------
    max_upload_bytes: int
    max_audio_seconds: float
    rate_limit_per_minute: int
    api_key: Optional[str]

    # -- Speech-to-text ----------------------------------------------------
    stt_engine: str
    groq_api_key: Optional[str]
    groq_model: str
    openai_api_key: Optional[str]
    openai_model: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_vad_filter: bool
    whisper_beam_size: int
    whisper_language: str
    whisper_no_speech_threshold: float
    whisper_logprob_threshold: float
    whisper_compression_threshold: float

    # -- LLM coaching insight ---------------------------------------------
    openrouter_api_key: Optional[str]
    openrouter_model: str
    llm_timeout_seconds: float
    llm_min_interval_seconds: float
    llm_max_transcript_chars: int
    _llm_enabled: bool = field(repr=False, default=True)

    # -- Sessions ----------------------------------------------------------
    session_ttl_seconds: int = 3600
    max_sessions: int = 500
    smoothing_alpha: float = 0.4

    # -- Deep learning -----------------------------------------------------
    # Every model is off by default: they are optional dependencies, and a
    # base install must behave identically without them.
    hf_token: Optional[str] = None
    ml_device: str = "auto"           # auto | cpu | cuda | mps
    ml_torch_threads: int = 2
    ml_warmup: bool = True

    ml_pitch_enabled: bool = False
    ml_crepe_capacity: str = "tiny"   # tiny | full
    ml_crepe_confidence: float = 0.5

    ml_vad_enabled: bool = False
    ml_vad_threshold: float = 0.5

    ml_disfluency_enabled: bool = False
    ml_disfluency_model: Optional[str] = None
    ml_disfluency_threshold: float = 0.5

    ml_emotion_enabled: bool = False
    ml_emotion_model: str = "superb/wav2vec2-base-superb-er"

    ml_speaker_enabled: bool = False
    ml_speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    # Cosine similarity below this means a different voice is speaking.
    ml_speaker_threshold: float = 0.6

    ml_diarization_enabled: bool = False
    ml_diarization_model: str = "pyannote/speaker-diarization-3.1"

    ml_prosody_enabled: bool = False
    ml_prosody_full_vector: bool = False

    ml_alignment_enabled: bool = False
    ml_alignment_backend: str = "auto"      # auto | whisperx | whisper_timestamped
    ml_alignment_whisper_model: str = "tiny.en"
    ml_alignment_min_pause: float = 0.3

    # ------------------------------------------------------------------
    @property
    def any_ml_enabled(self) -> bool:
        return any((
            self.ml_pitch_enabled, self.ml_vad_enabled,
            self.ml_disfluency_enabled, self.ml_emotion_enabled,
            self.ml_speaker_enabled, self.ml_diarization_enabled,
            self.ml_prosody_enabled, self.ml_alignment_enabled,
        ))

    @property
    def warmup_models(self) -> List[str]:
        """Models to load at startup so the first chunk is not slow."""
        if not self.ml_warmup:
            return []
        wanted = []
        if self.ml_pitch_enabled:
            wanted.append("crepe")
        if self.ml_vad_enabled:
            wanted.append("silero_vad")
        if self.ml_disfluency_enabled and self.ml_disfluency_model:
            wanted.append("disfluency")
        if self.ml_emotion_enabled:
            wanted.append("emotion")
        if self.ml_prosody_enabled:
            wanted.append("prosody")
        # Speaker, diarization, and alignment are heavy and are not needed
        # on the per-chunk path, so they load on first use instead.
        return wanted

    # ------------------------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        """LLM coaching runs only when explicitly enabled *and* keyed."""
        return self._llm_enabled and bool(self.openrouter_api_key)

    @property
    def resolved_stt_engine(self) -> str:
        """
        Which engine `auto` actually picks.

        Cloud engines are preferred when a key is present because they are
        both faster and more accurate than the local `tiny.en` fallback.
        """
        if self.stt_engine != "auto":
            return self.stt_engine
        if self.groq_api_key:
            return "groq"
        if self.openai_api_key:
            return "openai"
        return "local"

    def describe(self) -> Dict[str, Any]:
        """Redacted view of the config, safe to expose on a health endpoint."""
        return {
            "debug": self.debug,
            "log_level": self.log_level,
            "cors_origins": self.cors_origins,
            "stt_engine": self.stt_engine,
            "stt_engine_resolved": self.resolved_stt_engine,
            "groq_key_configured": bool(self.groq_api_key),
            "openai_key_configured": bool(self.openai_api_key),
            "whisper_model": self.whisper_model,
            "whisper_compute_type": self.whisper_compute_type,
            "llm_enabled": self.llm_enabled,
            "llm_model": self.openrouter_model if self.llm_enabled else None,
            "max_upload_bytes": self.max_upload_bytes,
            "max_audio_seconds": self.max_audio_seconds,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "auth_required": bool(self.api_key),
            "session_ttl_seconds": self.session_ttl_seconds,
            "ml": {
                "device": self.ml_device,
                "any_enabled": self.any_ml_enabled,
                "hf_token_configured": bool(self.hf_token),
                "pitch_crepe": self.ml_pitch_enabled,
                "vad_silero": self.ml_vad_enabled,
                "disfluency_acoustic": self.ml_disfluency_enabled,
                "disfluency_model": self.ml_disfluency_model,
                "emotion": self.ml_emotion_enabled,
                "speaker_embedding": self.ml_speaker_enabled,
                "diarization": self.ml_diarization_enabled,
                "prosody": self.ml_prosody_enabled,
                "word_alignment": self.ml_alignment_enabled,
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (once) and return the process-wide settings object."""
    engine = _str("STT_ENGINE", "auto").lower()
    if engine not in VALID_STT_ENGINES:
        engine = "auto"

    return Settings(
        # Server
        host=_str("HOST", "0.0.0.0"),
        port=_int("PORT", 8000, minimum=1, maximum=65535),
        debug=_bool("DEBUG", False),
        log_level=_str("LOG_LEVEL", "INFO").upper(),
        cors_origins=_list("CORS_ORIGINS", ["*"]),

        # Request limits
        max_upload_bytes=_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024, minimum=1024),
        max_audio_seconds=_float("MAX_AUDIO_SECONDS", 30.0, minimum=0.1, maximum=600.0),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 120, minimum=0),
        api_key=_opt("API_KEY"),

        # Speech-to-text
        stt_engine=engine,
        groq_api_key=_opt("GROQ_API_KEY"),
        groq_model=_str("GROQ_MODEL", "whisper-large-v3"),
        openai_api_key=_opt("OPENAI_API_KEY"),
        openai_model=_str("OPENAI_STT_MODEL", "whisper-1"),
        # base.en is roughly 3x the accuracy of tiny.en for about 2x the
        # CPU. tiny.en mangles ordinary speech badly enough that it reads
        # as broken, which is not a trade worth making by default.
        whisper_model=_str("WHISPER_MODEL", "base.en"),
        whisper_device=_str("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_str("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_vad_filter=_bool("WHISPER_VAD_FILTER", True),
        # Greedy decoding (beam 1) is the fastest and the least accurate.
        whisper_beam_size=_int("WHISPER_BEAM_SIZE", 5, minimum=1, maximum=10),
        whisper_language=_str("WHISPER_LANGUAGE", "en"),
        # Guards that make Whisper decline rather than invent. Without
        # them it will confidently transcribe breath and room tone.
        whisper_no_speech_threshold=_float("WHISPER_NO_SPEECH_THRESHOLD", 0.6,
                                           minimum=0.0, maximum=1.0),
        whisper_logprob_threshold=_float("WHISPER_LOGPROB_THRESHOLD", -1.0),
        whisper_compression_threshold=_float("WHISPER_COMPRESSION_THRESHOLD", 2.4),

        # LLM coaching insight
        openrouter_api_key=_opt("OPENROUTER_API_KEY"),
        # NOTE: must be a slug OpenRouter actually serves, or the insight
        # call fails and SpeakTwin silently falls back to rule-based feedback.
        openrouter_model=_str("OPENROUTER_MODEL", "openai/gpt-5.4"),
        llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 4.0, minimum=0.5, maximum=60.0),
        llm_min_interval_seconds=_float("LLM_MIN_INTERVAL_SECONDS", 8.0, minimum=0.0),
        llm_max_transcript_chars=_int("LLM_MAX_TRANSCRIPT_CHARS", 1200, minimum=100),
        # Rule-based coaching plus the local speech/confidence models are the
        # production default. An external LLM must be explicitly opted in.
        _llm_enabled=_bool("LLM_ENABLED", False),

        # Sessions
        session_ttl_seconds=_int("SESSION_TTL_SECONDS", 3600, minimum=60),
        max_sessions=_int("MAX_SESSIONS", 500, minimum=1),
        smoothing_alpha=_float("SMOOTHING_ALPHA", 0.4, minimum=0.05, maximum=1.0),

        # Deep learning
        # HF_TOKEN is the name huggingface_hub itself reads, so it is
        # honoured whether it is set for us or for the library.
        hf_token=_opt("HF_TOKEN") or _opt("HUGGINGFACE_TOKEN"),
        ml_device=_str("ML_DEVICE", "auto").lower(),
        ml_torch_threads=_int("ML_TORCH_THREADS", 2, minimum=0, maximum=64),
        ml_warmup=_bool("ML_WARMUP", True),

        ml_pitch_enabled=_bool("ML_PITCH_ENABLED", False),
        ml_crepe_capacity=_str("ML_CREPE_CAPACITY", "tiny").lower(),
        ml_crepe_confidence=_float("ML_CREPE_CONFIDENCE", 0.5, minimum=0.0, maximum=1.0),

        ml_vad_enabled=_bool("ML_VAD_ENABLED", False),
        ml_vad_threshold=_float("ML_VAD_THRESHOLD", 0.5, minimum=0.0, maximum=1.0),

        ml_disfluency_enabled=_bool("ML_DISFLUENCY_ENABLED", False),
        ml_disfluency_model=_opt("ML_DISFLUENCY_MODEL"),
        ml_disfluency_threshold=_float("ML_DISFLUENCY_THRESHOLD", 0.5,
                                       minimum=0.0, maximum=1.0),

        ml_emotion_enabled=_bool("ML_EMOTION_ENABLED", False),
        ml_emotion_model=_str("ML_EMOTION_MODEL", "superb/wav2vec2-base-superb-er"),

        ml_speaker_enabled=_bool("ML_SPEAKER_ENABLED", False),
        ml_speaker_model=_str("ML_SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb"),
        ml_speaker_threshold=_float("ML_SPEAKER_THRESHOLD", 0.6,
                                    minimum=0.0, maximum=1.0),

        ml_diarization_enabled=_bool("ML_DIARIZATION_ENABLED", False),
        ml_diarization_model=_str("ML_DIARIZATION_MODEL",
                                  "pyannote/speaker-diarization-3.1"),

        ml_prosody_enabled=_bool("ML_PROSODY_ENABLED", False),
        ml_prosody_full_vector=_bool("ML_PROSODY_FULL_VECTOR", False),

        ml_alignment_enabled=_bool("ML_ALIGNMENT_ENABLED", False),
        ml_alignment_backend=_str("ML_ALIGNMENT_BACKEND", "auto").lower(),
        ml_alignment_whisper_model=_str("ML_ALIGNMENT_WHISPER_MODEL", "tiny.en"),
        ml_alignment_min_pause=_float("ML_ALIGNMENT_MIN_PAUSE", 0.3, minimum=0.05),
    )
