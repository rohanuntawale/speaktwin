"""
SpeakTwin - API Routes
=======================
Chunked speech analysis plus session lifecycle.

Design notes:
  * The handler is `async` only to stream the upload. All CPU-bound work
    (Whisper inference, autocorrelation, cloud calls) is pushed onto the
    threadpool, because running it inline on the event loop stalls every
    other request in the process.
  * Uploads are read with a hard byte cap and decoded with a duration cap,
    so an oversized or malicious payload is rejected before it is expanded.
  * A failed transcription no longer fabricates a confidence score. The
    acoustic metrics are still real, so they are returned and the response
    is flagged `degraded` with a machine-readable warning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile  # type: ignore
from fastapi.concurrency import run_in_threadpool  # type: ignore

from backend.schemas import (  # type: ignore
    AnalysisResponse,
    PoseBatchRequest,
    PostureResponse,
    SessionCreatedResponse,
    SessionReport,
    SessionSummary,
    StatusResponse,
)
from backend.services.gesture_analysis import analyse_movement  # type: ignore
from backend.services.gesture_analysis import interpret as interpret_movement  # type: ignore
from backend.services.pose_analysis import PoseFrame, analyse_frames  # type: ignore
from backend.services.pose_analysis import interpret as interpret_pose  # type: ignore
from backend.services.posture_feedback import (  # type: ignore
    generate_posture_feedback,
    posture_status,
    presence_score,
    score_posture,
)
from backend.services.audio_analysis import analyze_audio  # type: ignore
from backend.services.audio_io import AudioDecodeError, duration_seconds, prepare  # type: ignore
from backend.services.clarity_analysis import analyze_clarity  # type: ignore
from backend.services.confidence_score import calculate_confidence  # type: ignore
from backend.services.feedback_engine import generate_feedback  # type: ignore
from backend.services.filler_detection import detect_fillers  # type: ignore
from backend.services.keyword_detection import detect_keywords  # type: ignore
from backend.services.session_store import session_store  # type: ignore
from backend.services.speech_to_text import transcribe  # type: ignore
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import (  # type: ignore
    POSE_MIN_USABLE_FRAMES,
    SAMPLE_RATE,
    SILENCE_DBFS,
    get_logger,
)

logger = get_logger(__name__)
router = APIRouter()

UPLOAD_READ_CHUNK = 64 * 1024
# ~30 fps for 10 s. Enough headroom for any sane batch, low enough that a
# malicious payload cannot make the server chew through millions of points.
MAX_POSE_FRAMES = 300


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------
async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """
    Stream the upload into memory, aborting as soon as it exceeds `limit`.

    Reading incrementally means an oversized body is rejected part-way
    through instead of being buffered in full first.
    """
    buffer = bytearray()
    while True:
        chunk = await upload.read(UPLOAD_READ_CHUNK)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Audio upload exceeds the {limit} byte limit.",
            )
    return bytes(buffer)


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------
def _empty_fillers() -> Dict[str, Any]:
    return {"total_fillers": 0, "filler_rate": 0.0, "details": {}, "total_words": 0}


def _empty_keywords() -> Dict[str, Any]:
    return {"total_keywords": 0, "found_keywords": {}, "keywords_list": []}


def _primary_message(llm_insight: Optional[str],
                     messages: List[Dict[str, str]]) -> str:
    """
    Pick the headline coaching line.

    Preference order: the LLM insight, then the first warning (the most
    actionable rule-based message), then whatever came first.
    """
    if llm_insight:
        return llm_insight

    for message in messages:
        if message.get("type") == "warning":
            return str(message.get("text", ""))

    if messages:
        return str(messages[0].get("text", ""))

    return "Listening..."


def _perform_full_analysis(audio, session_id: Optional[str] = None,
                           source_sample_rate: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the full pipeline over one chunk. Blocking - call via threadpool.
    """
    settings = get_settings()
    session = session_store.get(session_id)
    warnings: List[str] = []
    degraded = False

    # 1. Acoustics -------------------------------------------------------
    silence_gate = session.silence_gate_db() if session else SILENCE_DBFS
    metrics = analyze_audio(audio, SAMPLE_RATE, silence_dbfs=silence_gate)
    chunk_seconds = duration_seconds(audio, SAMPLE_RATE)

    # 1b. Neural upgrades (no-ops unless the models are enabled) ---------
    if settings.any_ml_enabled:
        from backend.services.ml import enrichment  # type: ignore
        metrics = enrichment.enrich_acoustic(audio, metrics, SAMPLE_RATE)

    # 2. Transcription ---------------------------------------------------
    # Prefer the neural VAD verdict when we have one: an energy threshold
    # cannot tell a quiet speaker from a noisy room, which is exactly the
    # case where the gate makes the wrong call.
    if "has_speech" in metrics:
        is_silent = not metrics["has_speech"]
    else:
        is_silent = metrics["energy_db"] < silence_gate

    transcript = ""
    if is_silent:
        warnings.append("silence_skipped")
        logger.debug("Chunk has no speech (gate %.1f dBFS), skipping STT",
                     silence_gate)
    else:
        stt_result = transcribe(
            audio,
            SAMPLE_RATE,
            initial_prompt=session.prompt_context() if session else None,
        )
        if "error" in stt_result:
            degraded = True
            warnings.append("stt_unavailable")
            logger.warning("STT failed: %s", stt_result["error"])
        else:
            transcript = stt_result.get("text", "")

    # 3. Linguistics -----------------------------------------------------
    ml_extra: Dict[str, Any] = {}
    if transcript:
        fillers = detect_fillers(transcript)
        keywords = detect_keywords(transcript)

        # 3b. Acoustic fillers + word timings. Whisper deletes most "um"s,
        # so the audio-side detector is what makes the count honest.
        if settings.any_ml_enabled:
            from backend.services.ml import enrichment  # type: ignore
            ml_extra = enrichment.enrich_linguistic(
                audio, transcript, fillers, SAMPLE_RATE
            )
            fillers = ml_extra.pop("fillers", fillers)

        clarity = analyze_clarity(transcript, float(fillers["filler_rate"]))
        wpm = round(fillers["total_words"] / (chunk_seconds / 60.0), 1) if chunk_seconds > 0 else 0.0
    else:
        fillers = _empty_fillers()
        keywords = _empty_keywords()
        clarity = {"lexical_diversity": 0.0, "mattr": 0.0, "clarity_score": 0}
        wpm = 0.0

    # 4. Coaching --------------------------------------------------------
    feedback = generate_feedback(
        energy_db=metrics["energy_db"],
        mean_pitch=metrics["mean_pitch"],
        pitch_std=metrics["pitch_std"],
        wpm=wpm,
        filler_rate=float(fillers["filler_rate"]),
        pause_ratio=metrics["pause_ratio"],
    )

    confidence = calculate_confidence(
        wpm=wpm,
        pitch_std=metrics["pitch_std"],
        energy_db=metrics["energy_db"],
        filler_rate=float(fillers["filler_rate"]),
    )

    if degraded:
        # WPM and filler rate are both unknown without a transcript, so the
        # sub-scores that depend on them are not evidence of anything.
        warnings.append("confidence_acoustic_only")

    # 5. LLM insight (throttled, best-effort) ----------------------------
    llm_insight: Optional[str] = None
    if transcript and settings.llm_enabled:
        from backend.services import llm_feedback  # type: ignore

        if llm_feedback.should_call(session_id):
            llm_insight = llm_feedback.generate_llm_insight(
                transcript,
                {
                    "wpm": wpm,
                    "pitch_std": metrics["pitch_std"],
                    "total_fillers": fillers["total_fillers"],
                },
                session_id=session_id,
            )

    result: Dict[str, Any] = {
        "message": _primary_message(llm_insight, feedback["messages"]),
        "pitch": metrics["mean_pitch"],
        "pitch_std": metrics["pitch_std"],
        "voiced_ratio": metrics["voiced_ratio"],
        "energy": metrics["energy"],
        "energy_db": metrics["energy_db"],
        "pause_ratio": metrics["pause_ratio"],
        "longest_pause_sec": metrics["longest_pause_sec"],
        "wpm": wpm,
        "transcript": transcript,
        "fillers": fillers,
        "keywords": keywords,
        "clarity": clarity["clarity_score"],
        "lexical_diversity": clarity["lexical_diversity"],
        "confidence_score": confidence["score"],
        "confidence_breakdown": confidence["breakdown"],
        "confidence_smoothed": None,
        "feedback": feedback["messages"],
        "status": feedback["status"],
        "degraded": degraded,
        "warnings": warnings,
        "source_sample_rate": source_sample_rate,
        "session_id": session_id,
        "session": None,

        # Neural extras - present only when the matching model is enabled.
        "engines": metrics.get("engines"),
        "pitch_confidence": metrics.get("pitch_confidence"),
        "speech_ratio": metrics.get("speech_ratio"),
        "emotion": metrics.get("emotion"),
        "prosody": metrics.get("prosody"),
        "disfluency": ml_extra.get("disfluency"),
        "alignment": ml_extra.get("alignment"),
    }

    # 5b. Speaker continuity --------------------------------------------
    # Verifies the same person is still speaking. Without it, a second voice
    # is silently folded into the session averages.
    if session is not None and settings.ml_speaker_enabled:
        from backend.services.ml import enrichment  # type: ignore

        similarity = session.track_speaker(
            enrichment.speaker_fingerprint(audio, SAMPLE_RATE),
            settings.ml_speaker_threshold,
        )
        if similarity is not None:
            result["speaker_similarity"] = similarity
            if similarity < settings.ml_speaker_threshold:
                warnings.append("speaker_changed")

    # 6. Fold into the session ------------------------------------------
    summary = session_store.record(session_id, result, chunk_seconds)
    if summary is not None:
        result["session"] = summary
        result["confidence_smoothed"] = summary.get("avg_confidence")
    elif session_id:
        warnings.append("session_not_found")

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_blob(
    audio_file: UploadFile = File(..., description="Audio chunk (WAV/FLAC/OGG)"),
    session_id: Optional[str] = Form(
        None, description="Optional session to accumulate this chunk into"
    ),
):
    """Analyse one audio chunk and return speech metrics plus coaching."""
    settings = get_settings()

    audio_bytes = await _read_limited(audio_file, settings.max_upload_bytes)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    try:
        audio, source_sr = await run_in_threadpool(
            prepare, audio_bytes, settings.max_audio_seconds
        )
    except AudioDecodeError as exc:
        logger.info("Rejected undecodable upload: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await run_in_threadpool(
        _perform_full_analysis, audio, session_id, source_sr
    )


@router.post("/pose", response_model=PostureResponse)
async def analyze_pose(batch: PoseBatchRequest):
    """
    Score a batch of pose landmarks for posture and gesture.

    MediaPipe runs in the browser, so video never reaches the server —
    only landmark coordinates, which keeps a batch around 25 KB.
    """
    if not batch.frames:
        raise HTTPException(status_code=400, detail="No pose frames supplied.")
    if len(batch.frames) > MAX_POSE_FRAMES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many frames in one batch (max {MAX_POSE_FRAMES}).",
        )

    return await run_in_threadpool(_score_pose_batch, batch)


def _score_pose_batch(batch: PoseBatchRequest) -> Dict[str, Any]:
    """Blocking posture pipeline — call via threadpool."""
    settings = get_settings()
    warnings: List[str] = []

    frames = [
        PoseFrame([lm.model_dump() for lm in frame.landmarks], batch.aspect)
        for frame in batch.frames
    ]
    pose = analyse_frames(frames)

    usable_ratio = (
        pose.get("usable_frames", 0) / len(frames) if frames else 0.0
    )
    if pose.get("detected") and usable_ratio < POSE_MIN_USABLE_FRAMES:
        # Seen in too few frames to say anything responsibly.
        warnings.append("partially_out_of_frame")
        pose = {**pose, "detected": False}

    series = pose.pop("frames", [])
    movement = analyse_movement(series, batch.duration)

    pose_bands = interpret_pose(pose)
    movement_bands = interpret_movement(movement)

    scored = score_posture(pose, movement)
    messages = generate_posture_feedback(pose, movement, pose_bands, movement_bands)
    status = posture_status(scored["score"], messages)

    session = session_store.get(batch.session_id)
    presence = None
    if session is not None:
        session.record_posture(scored["score"], movement, messages,
                               settings.smoothing_alpha)
        presence = presence_score(
            session.summary().get("avg_confidence"), session.posture_score
        )
    elif batch.session_id:
        warnings.append("session_not_found")

    headline = next(
        (m["text"] for m in messages if m["type"] == "warning"),
        messages[0]["text"] if messages else "Looking good.",
    )

    return {
        "detected": bool(pose.get("detected")),
        "message": headline,
        "score": scored["score"],
        "status": status,
        "breakdown": scored["breakdown"],
        "measured": scored["measured"],
        "pose": {k: v for k, v in pose.items() if k != "frames"},
        "movement": movement,
        "bands": {**pose_bands, **movement_bands},
        "feedback": messages,
        "frames_used": pose.get("usable_frames", 0),
        "frames_received": len(frames),
        "presence_score": presence,
        "session_id": batch.session_id,
        "warnings": warnings,
    }


@router.post("/diarize")
async def diarize_audio(
    audio_file: UploadFile = File(..., description="A full recording, not a chunk"),
):
    """
    Segment a recording by speaker.

    Deliberately separate from `/analyze`: 2.5 s is far too short to separate
    voices reliably, so diarization needs a longer span of audio. Send a whole
    practice recording here rather than a live chunk.

    Requires `ML_DIARIZATION_ENABLED=true`, plus an `HF_TOKEN` with the
    pyannote model terms accepted.
    """
    settings = get_settings()
    if not settings.ml_diarization_enabled:
        raise HTTPException(
            status_code=503,
            detail="Diarization is disabled. Set ML_DIARIZATION_ENABLED=true "
                   "and install requirements-ml.txt.",
        )

    audio_bytes = await _read_limited(audio_file, settings.max_upload_bytes)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    try:
        # Diarization needs the whole recording, so the per-chunk duration
        # cap does not apply - only the upload byte cap bounds it.
        audio, source_sr = await run_in_threadpool(prepare, audio_bytes, None)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    from backend.services.ml import enrichment  # type: ignore

    result = await run_in_threadpool(enrichment.analyze_session_audio, audio, SAMPLE_RATE)
    if not result.get("diarization"):
        raise HTTPException(
            status_code=503,
            detail="Diarization model unavailable. Check /api/health for the "
                   "load error - the pyannote model is gated and needs HF_TOKEN.",
        )

    return {
        **result["diarization"],
        "duration_seconds": round(duration_seconds(audio, SAMPLE_RATE), 2),
        "source_sample_rate": source_sr,
    }


@router.post("/session", response_model=SessionCreatedResponse, status_code=201)
async def create_session():
    """Open a session so chunks can be accumulated and smoothed."""
    session = session_store.create()
    return {"session_id": session.session_id, "created_at": session.created_at}


@router.get("/session/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str):
    """Rolling totals and smoothed averages for an open session."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return session.summary()


@router.get("/session/{session_id}/report", response_model=SessionReport)
async def get_session_report(session_id: str):
    """Full exportable report, including the stitched transcript."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return session.report()


@router.delete("/session/{session_id}", response_model=SessionReport)
async def end_session(session_id: str):
    """Close a session and return its final report."""
    report = session_store.end(session_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    from backend.services import llm_feedback  # type: ignore
    llm_feedback.forget_session(session_id)

    return report


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Lightweight readiness probe for the frontend."""
    settings = get_settings()
    return {
        "status": "ready",
        "mode": "deployment",
        "stt_engine": settings.resolved_stt_engine,
        "llm_enabled": settings.llm_enabled,
        "active_sessions": session_store.active_count(),
    }
