"""
SpeakTwin - ML Enrichment Orchestrator
=======================================
Single entry point the analysis route calls to run whichever neural models
are enabled over one chunk.

Two design points worth stating:

  * **Neural results override DSP results, never replace the code.** CREPE
    supersedes autocorrelation pitch and Silero supersedes the energy gate
    *when they are on*; the DSP path stays wired so the backend degrades to
    a working state rather than an error.

  * **Nothing here can fail a request.** Each model is independently
    guarded. One broken model costs one field, not the response.

Cost note: everything here runs inside the same 2.5s chunk budget, so only
per-chunk-appropriate models are called. Diarization and speaker
verification are session-level concerns and are exposed separately.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np  # type: ignore

from backend.services.ml import (  # type: ignore
    alignment,
    disfluency,
    emotion,
    pitch as ml_pitch,
    prosody,
    speaker,
    vad,
)
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import SAMPLE_RATE, get_logger  # type: ignore

logger = get_logger(__name__)


def enrich_acoustic(audio: np.ndarray, metrics: Dict[str, Any],
                    sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """
    Upgrade the DSP acoustic metrics in place with neural equivalents.

    Called *before* transcription, because the VAD result decides whether
    transcription is worth running at all.
    """
    settings = get_settings()
    if not settings.any_ml_enabled:
        return metrics

    metrics = dict(metrics)
    metrics.setdefault("engines", {})
    metrics["engines"]["pitch"] = "autocorrelation"
    metrics["engines"]["vad"] = "energy"

    # -- Pitch: CREPE beats autocorrelation on real speech ---------------
    crepe_result = ml_pitch.estimate_pitch(audio, sr)
    if crepe_result:
        metrics["mean_pitch"] = crepe_result["mean_pitch"]
        metrics["pitch_std"] = crepe_result["pitch_std"]
        metrics["voiced_ratio"] = crepe_result["voiced_ratio"]
        metrics["pitch_confidence"] = crepe_result["pitch_confidence"]
        metrics["engines"]["pitch"] = "crepe"

    # -- VAD: real speech detection beats an energy threshold ------------
    vad_result = vad.detect_speech(audio, sr)
    if vad_result:
        metrics["pause_ratio"] = vad_result["pause_ratio"]
        metrics["longest_pause_sec"] = vad_result["longest_pause_sec"]
        metrics["speech_ratio"] = vad_result["speech_ratio"]
        metrics["speech_seconds"] = vad_result["speech_seconds"]
        metrics["has_speech"] = vad_result["has_speech"]
        metrics["engines"]["vad"] = "silero"

    # -- Prosody: eGeMAPS functionals ------------------------------------
    prosody_result = prosody.extract(audio, sr)
    if prosody_result:
        metrics["prosody"] = prosody_result

    # -- Emotion / delivery affect ---------------------------------------
    emotion_result = emotion.analyze(audio, sr)
    if emotion_result:
        metrics["emotion"] = emotion_result

    return metrics


def enrich_linguistic(audio: np.ndarray, transcript: str,
                      fillers: Dict[str, Any],
                      sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """
    Add transcript-dependent neural results.

    Returns a dict of extra fields plus a possibly-corrected `fillers`
    block, since acoustic detection typically finds fillers that Whisper
    deleted from the transcript.
    """
    settings = get_settings()
    extra: Dict[str, Any] = {}
    if not settings.any_ml_enabled:
        return {"fillers": fillers}

    # -- Acoustic fillers, merged with the text-based count --------------
    acoustic = disfluency.detect(audio, sr)
    merged_fillers = disfluency.merge_with_text(fillers, acoustic)
    if acoustic:
        extra["disfluency"] = {
            "acoustic_total": acoustic["total_fillers"],
            "text_total": fillers.get("total_fillers", 0),
            "events": acoustic["events"],
            "engine": acoustic["engine"],
        }

    # -- Word-level timing -----------------------------------------------
    alignment_result = alignment.align(audio, transcript, sr)
    if alignment_result:
        extra["alignment"] = alignment_result

    return {"fillers": merged_fillers, **extra}


def speaker_fingerprint(audio: np.ndarray,
                        sr: int = SAMPLE_RATE) -> Optional[list]:
    """Speaker embedding for the chunk, for continuity checks."""
    return speaker.embed(audio, sr)


def analyze_session_audio(audio: np.ndarray,
                          sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """
    Session-level analysis over a longer span of audio.

    Diarization needs far more than 2.5s to separate voices, so it is run
    here rather than on the per-chunk path.
    """
    result: Dict[str, Any] = {}

    diarization_result = speaker.diarize(audio, sr)
    if diarization_result:
        result["diarization"] = diarization_result

    return result
