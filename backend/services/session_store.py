"""
SpeakTwin - Session State
==========================
Holds everything that has to survive across chunks within one speaking
session. Without it each 2.5s chunk was analysed in isolation and thrown
away, which meant:

  * cumulative filler/keyword totals lived only in browser memory and
    vanished on refresh
  * the confidence score was recomputed from scratch every chunk, so it
    jumped around instead of trending
  * every chunk was transcribed with no knowledge of the previous one
  * the silence gate used a fixed threshold regardless of mic gain

Storage is in-process and TTL-bounded: sessions are ephemeral coaching
state, not records worth a database. Reports are handed to the client as
JSON to persist wherever it likes.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import (  # type: ignore
    get_logger,
    SILENCE_DBFS,
    HARD_SILENCE_FLOOR_DBFS,
)

logger = get_logger(__name__)

# How much of the previous transcript to feed the decoder as context.
PROMPT_CONTEXT_CHARS = 220
# Chunk loudness history kept for the adaptive noise floor.
LOUDNESS_HISTORY = 40
# Chunks needed before the adaptive floor is trusted.
MIN_CHUNKS_FOR_ADAPTIVE_GATE = 4
# Headroom above the observed noise floor.
NOISE_FLOOR_HEADROOM_DB = 8.0


def _ema(previous: Optional[float], value: float, alpha: float) -> float:
    """Exponential moving average, seeded by the first observation."""
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


@dataclass
class Session:
    """Mutable state for a single speaking session."""

    session_id: str
    created_at: float
    updated_at: float

    chunk_count: int = 0
    analysed_chunks: int = 0          # chunks that carried actual speech
    audio_seconds: float = 0.0

    total_words: int = 0
    total_fillers: int = 0
    filler_details: Dict[str, int] = field(default_factory=dict)
    total_keywords: int = 0
    keyword_details: Dict[str, int] = field(default_factory=dict)

    transcript_segments: List[str] = field(default_factory=list)

    confidence_ema: Optional[float] = None
    wpm_ema: Optional[float] = None
    clarity_ema: Optional[float] = None
    pitch_ema: Optional[float] = None

    peak_confidence: int = 0
    loudness_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=LOUDNESS_HISTORY)
    )

    # Speaker continuity. The first embedding of the session becomes the
    # reference; later chunks are compared against it so a different voice
    # taking over is visible rather than silently folded into the averages.
    reference_embedding: Optional[List[float]] = None
    speaker_similarity: Optional[float] = None
    speaker_changes: int = 0

    # ------------------------------------------------------------------
    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.updated_at - self.created_at)

    @property
    def transcript(self) -> str:
        return " ".join(self.transcript_segments).strip()

    def prompt_context(self) -> Optional[str]:
        """Tail of the transcript, used as the decoder's `initial_prompt`."""
        if not self.transcript_segments:
            return None
        tail = self.transcript[-PROMPT_CONTEXT_CHARS:].strip()
        return tail or None

    def silence_gate_db(self) -> float:
        """
        Loudness threshold below which a chunk is treated as silence.

        The absolute threshold is a ceiling, never a floor: a quiet mic
        lowers the gate so speech still reaches the transcriber, but a noisy
        room can never raise it and start transcribing background hiss.
        """
        if len(self.loudness_history) < MIN_CHUNKS_FOR_ADAPTIVE_GATE:
            return SILENCE_DBFS

        ordered = sorted(self.loudness_history)
        index = max(0, int(len(ordered) * 0.1) - 1)
        noise_floor = ordered[index]

        adaptive = noise_floor + NOISE_FLOOR_HEADROOM_DB
        return max(HARD_SILENCE_FLOOR_DBFS, min(SILENCE_DBFS, adaptive))

    # ------------------------------------------------------------------
    def track_speaker(self, embedding: Optional[List[float]],
                      threshold: float) -> Optional[float]:
        """
        Compare this chunk's voice against the session reference.

        Returns the similarity, or None when embeddings are unavailable.
        The first embedding seeds the reference and scores 1.0 by definition.
        """
        if not embedding:
            return None

        if self.reference_embedding is None:
            self.reference_embedding = embedding
            self.speaker_similarity = 1.0
            return 1.0

        from backend.services.ml.speaker import cosine_similarity  # type: ignore

        similarity = cosine_similarity(self.reference_embedding, embedding)
        # Count transitions, not frames: a second speaker talking for six
        # chunks is one change, not six.
        #
        # `self.speaker_similarity or 1.0` would be wrong here - a genuine
        # similarity of 0.0 (orthogonal voices) is falsy, so every chunk
        # would look like a fresh transition.
        previous = self.speaker_similarity
        was_same = previous is None or previous >= threshold
        is_same = similarity >= threshold
        if was_same and not is_same:
            self.speaker_changes += 1

        self.speaker_similarity = similarity
        return similarity

    def record(self, result: Dict[str, Any], audio_seconds: float,
               alpha: float) -> None:
        """Fold one chunk's analysis into the running session state."""
        self.updated_at = time.time()
        self.chunk_count += 1
        self.audio_seconds += audio_seconds

        energy_db = result.get("energy_db")
        if energy_db is not None:
            self.loudness_history.append(float(energy_db))

        transcript = (result.get("transcript") or "").strip()
        if not transcript:
            return

        self.analysed_chunks += 1
        self.transcript_segments.append(transcript)

        fillers = result.get("fillers") or {}
        self.total_words += int(fillers.get("total_words", 0) or 0)
        self.total_fillers += int(fillers.get("total_fillers", 0) or 0)
        for word, count in (fillers.get("details") or {}).items():
            self.filler_details[word] = self.filler_details.get(word, 0) + int(count)

        keywords = result.get("keywords") or {}
        self.total_keywords += int(keywords.get("total_keywords", 0) or 0)
        for word, count in (keywords.get("found_keywords") or {}).items():
            self.keyword_details[word] = self.keyword_details.get(word, 0) + int(count)

        score = result.get("confidence_score")
        if score is not None:
            self.confidence_ema = _ema(self.confidence_ema, float(score), alpha)
            self.peak_confidence = max(self.peak_confidence, int(score))

        wpm = result.get("wpm")
        if wpm:
            self.wpm_ema = _ema(self.wpm_ema, float(wpm), alpha)

        clarity = result.get("clarity")
        if clarity is not None:
            self.clarity_ema = _ema(self.clarity_ema, float(clarity), alpha)

        pitch = result.get("pitch")
        if pitch:
            self.pitch_ema = _ema(self.pitch_ema, float(pitch), alpha)

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Compact rolling view, returned alongside every analysis."""
        filler_rate = (
            round(self.total_fillers / self.total_words, 4)
            if self.total_words else 0.0
        )
        return {
            "session_id": self.session_id,
            "chunk_count": self.chunk_count,
            "analysed_chunks": self.analysed_chunks,
            "duration_seconds": round(self.duration_seconds, 1),
            "audio_seconds": round(self.audio_seconds, 1),
            "total_words": self.total_words,
            "total_fillers": self.total_fillers,
            "filler_rate": filler_rate,
            "filler_details": dict(self.filler_details),
            "total_keywords": self.total_keywords,
            "keyword_details": dict(self.keyword_details),
            "avg_confidence": (
                int(round(self.confidence_ema)) if self.confidence_ema is not None else None
            ),
            "peak_confidence": self.peak_confidence,
            "avg_wpm": round(self.wpm_ema, 1) if self.wpm_ema is not None else None,
            "avg_clarity": (
                int(round(self.clarity_ema)) if self.clarity_ema is not None else None
            ),
            "avg_pitch": round(self.pitch_ema, 1) if self.pitch_ema is not None else None,
            "speaker_similarity": self.speaker_similarity,
            "speaker_changes": self.speaker_changes,
        }

    def report(self) -> Dict[str, Any]:
        """Full exportable record, including the stitched transcript."""
        data = self.summary()
        data.update({
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "transcript": self.transcript,
            "transcript_segments": list(self.transcript_segments),
            "top_fillers": sorted(
                self.filler_details.items(), key=lambda kv: kv[1], reverse=True
            )[:10],
            "top_keywords": sorted(
                self.keyword_details.items(), key=lambda kv: kv[1], reverse=True
            )[:10],
        })
        return data


class SessionStore:
    """Thread-safe, TTL-bounded registry of active sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        # Reentrant so a public method can safely call another one.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def _prune_locked(self) -> None:
        """Drop expired sessions, then the oldest if still over capacity."""
        settings = get_settings()
        now = time.time()

        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.updated_at > settings.session_ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        if expired:
            logger.info("Pruned %d expired session(s)", len(expired))

        overflow = len(self._sessions) - settings.max_sessions
        if overflow > 0:
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1].updated_at)
            for sid, _ in oldest[:overflow]:
                self._sessions.pop(sid, None)
            logger.warning("Session cap reached; evicted %d oldest session(s)", overflow)

    # ------------------------------------------------------------------
    def create(self) -> Session:
        now = time.time()
        session = Session(session_id=uuid.uuid4().hex, created_at=now, updated_at=now)
        with self._lock:
            # Insert first, then prune. Pruning beforehand measured capacity
            # without the incoming session, so the store settled one over
            # `max_sessions` instead of at it.
            self._sessions[session.session_id] = session
            self._prune_locked()
        logger.info("Session %s started", session.session_id)
        return session

    def _get_locked(self, session_id: Optional[str]) -> Optional[Session]:
        """Caller must hold the lock. Expired sessions are dropped on access."""
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.updated_at > get_settings().session_ttl_seconds:
            self._sessions.pop(session_id, None)
            return None
        return session

    def get(self, session_id: Optional[str]) -> Optional[Session]:
        with self._lock:
            return self._get_locked(session_id)

    def record(self, session_id: Optional[str], result: Dict[str, Any],
               audio_seconds: float) -> Optional[Dict[str, Any]]:
        """Fold a chunk into its session and return the updated summary."""
        with self._lock:
            session = self._get_locked(session_id)
            if session is None:
                return None
            session.record(result, audio_seconds, get_settings().smoothing_alpha)
            return session.summary()

    def end(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Close a session and return its final report."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        logger.info("Session %s ended after %d chunk(s)",
                    session_id, session.chunk_count)
        return session.report()

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def clear(self) -> None:
        """Drop every session - used by tests and on shutdown."""
        with self._lock:
            self._sessions.clear()


# Process-wide singleton
session_store = SessionStore()
