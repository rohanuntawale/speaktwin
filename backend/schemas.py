"""
SpeakTwin - API Schemas
========================
Pydantic models for every response body.

The API used to return bare dicts through `JSONResponse`, which meant there
was no typed contract between backend and frontend and nothing useful in
the OpenAPI docs - exactly how the README drifted out of sync with the
routes it documented.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field  # type: ignore


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class FillerBreakdown(BaseModel):
    total_fillers: int = 0
    filler_rate: float = 0.0
    details: Dict[str, int] = Field(default_factory=dict)
    total_words: int = 0


class KeywordBreakdown(BaseModel):
    total_keywords: int = 0
    found_keywords: Dict[str, int] = Field(default_factory=dict)
    keywords_list: List[str] = Field(default_factory=list)


class ConfidenceBreakdown(BaseModel):
    wpm: int = 0
    pitch_variation: int = 0
    energy: int = 0
    filler_usage: int = 0


class FeedbackMessage(BaseModel):
    text: str
    type: str = "info"       # info | success | warning
    category: str = "general"


class SessionSummary(BaseModel):
    session_id: str
    chunk_count: int = 0
    analysed_chunks: int = 0
    duration_seconds: float = 0.0
    audio_seconds: float = 0.0
    total_words: int = 0
    total_fillers: int = 0
    filler_rate: float = 0.0
    filler_details: Dict[str, int] = Field(default_factory=dict)
    total_keywords: int = 0
    keyword_details: Dict[str, int] = Field(default_factory=dict)
    avg_confidence: Optional[int] = None
    peak_confidence: int = 0
    avg_wpm: Optional[float] = None
    avg_clarity: Optional[int] = None
    avg_pitch: Optional[float] = None
    speaker_similarity: Optional[float] = Field(
        default=None,
        description="Cosine similarity of the latest voice against the session "
                    "reference. Null unless ML_SPEAKER_ENABLED.",
    )
    speaker_changes: int = Field(
        default=0, description="Times the speaker appears to have changed."
    )


class SessionReport(SessionSummary):
    created_at: float
    updated_at: float
    transcript: str = ""
    transcript_segments: List[str] = Field(default_factory=list)
    top_fillers: List[Any] = Field(default_factory=list)
    top_keywords: List[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class AnalysisResponse(BaseModel):
    """One analysed audio chunk."""

    message: str

    # Acoustic
    pitch: float = 0.0
    pitch_std: float = 0.0
    voiced_ratio: float = 0.0
    energy: float = 0.0
    energy_db: float = -90.0
    pause_ratio: float = 0.0
    longest_pause_sec: float = 0.0

    # Linguistic
    wpm: float = 0.0
    transcript: str = ""
    fillers: FillerBreakdown = Field(default_factory=FillerBreakdown)
    keywords: KeywordBreakdown = Field(default_factory=KeywordBreakdown)
    clarity: int = 0
    lexical_diversity: float = 0.0

    # Scoring
    confidence_score: int = 0
    confidence_breakdown: ConfidenceBreakdown = Field(
        default_factory=ConfidenceBreakdown
    )
    confidence_smoothed: Optional[int] = Field(
        default=None,
        description="Session-smoothed confidence; more stable than the "
                    "per-chunk score. Prefer this for display when present.",
    )

    # Coaching
    feedback: List[FeedbackMessage] = Field(default_factory=list)
    status: str = "info"

    # Diagnostics
    degraded: bool = Field(
        default=False,
        description="True when part of the pipeline failed and the result "
                    "is incomplete (for example, transcription was "
                    "unavailable but acoustic metrics are still valid).",
    )
    warnings: List[str] = Field(default_factory=list)
    source_sample_rate: Optional[int] = None

    # Session
    session_id: Optional[str] = None
    session: Optional[SessionSummary] = None

    # ---- Neural extras -------------------------------------------------
    # All null unless the matching model is enabled (see ML_* settings).
    engines: Optional[Dict[str, str]] = Field(
        default=None,
        description="Which engine produced each metric, e.g. "
                    "{'pitch': 'crepe', 'vad': 'silero'}.",
    )
    pitch_confidence: Optional[float] = Field(
        default=None, description="CREPE periodicity confidence (0-1)."
    )
    speech_ratio: Optional[float] = Field(
        default=None, description="Fraction of the chunk Silero VAD called speech."
    )
    emotion: Optional[Dict[str, Any]] = Field(
        default=None, description="Speech emotion: label, scores, tension."
    )
    prosody: Optional[Dict[str, Any]] = Field(
        default=None, description="eGeMAPS functionals (jitter, shimmer, HNR...)."
    )
    disfluency: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Acoustic filler events. Catches the hesitations Whisper "
                    "deletes from the transcript.",
    )
    alignment: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Word-level timings, located pauses, articulation rate.",
    )
    speaker_similarity: Optional[float] = Field(
        default=None,
        description="Voice match against the session reference. Below the "
                    "threshold adds a 'speaker_changed' warning.",
    )


class SessionCreatedResponse(BaseModel):
    session_id: str
    created_at: float


class HealthResponse(BaseModel):
    status: str                # ok | degraded
    version: str
    stt_engine: str
    stt_ready: bool
    local_model_loaded: bool
    llm_enabled: bool
    active_sessions: int
    config: Dict[str, Any] = Field(default_factory=dict)
    ml: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-model load state, device, and any load errors.",
    )


class StatusResponse(BaseModel):
    status: str
    mode: str
    stt_engine: str
    llm_enabled: bool
    active_sessions: int


class ErrorResponse(BaseModel):
    message: str
    status: str = "error"
    request_id: Optional[str] = None
    detail: Optional[str] = Field(
        default=None,
        description="Present only when DEBUG is enabled.",
    )
