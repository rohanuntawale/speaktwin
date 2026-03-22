"""
SpeakTwin - API Routes
========================
Defines the main API endpoint for real-time speech analysis.
Optimised for Windows by using soundfile instead of librosa.
"""

from fastapi import APIRouter, UploadFile, File # type: ignore
from fastapi.responses import JSONResponse # type: ignore
import io
import soundfile as sf # type: ignore
import numpy as np # type: ignore
import traceback

from backend.services.audio_analysis import analyze_audio # type: ignore
from backend.services.speech_to_text import transcribe # type: ignore
from backend.services.filler_detection import detect_fillers # type: ignore
from backend.services.feedback_engine import generate_feedback # type: ignore
from backend.services.confidence_score import calculate_confidence # type: ignore
from backend.services.keyword_detection import detect_keywords # type: ignore
from backend.services.clarity_analysis import analyze_clarity # type: ignore
from backend.utils.helpers import get_logger, SAMPLE_RATE # type: ignore

logger = get_logger(__name__)
router = APIRouter()

def _perform_full_analysis(audio_data: np.ndarray) -> dict:
    """Helper to run the full analysis pipeline on an audio chunk."""
    # 1. Acoustic Analysis (Refactored to be librosa-free)
    audio_metrics = analyze_audio(audio_data)

    from backend.utils.helpers import ENERGY_SILENCE_THRESHOLD # type: ignore
    
    # 2. Speech-to-Text (Resilient to failure and silence hallucinations)
    if audio_metrics["energy"] < ENERGY_SILENCE_THRESHOLD:
        stt_result = {"text": "", "word_count": 0, "segments": []}
        logger.debug("Silence detected, skipping STT.")
    else:
        stt_result = transcribe(audio_data)
    
    if "error" in stt_result:
        err_msg = stt_result['error']
        logger.warning(f"STT Failed: {err_msg}")
        stt_result = {"text": "", "word_count": 0, "segments": []}
        feedback = {"messages": [{"text": f"STT Error: {err_msg}", "type": "warning", "category": "stt"}], "status": "needs_improvement"}
        filler_result = {"total_fillers": 0, "filler_rate": 0.0, "details": {}}
        keywords_result = {"total_keywords": 0, "found_keywords": {}, "keywords_list": []}
        clarity_result = {"lexical_diversity": 0.0, "clarity_score": 0}
        wpm = 0.0
        confidence = {"score": 50, "breakdown": {"wpm": 0, "pitch_variation": 50, "energy": 50, "filler_usage": 100}}
    else:
        # Normal Pipeline Execution
        chunk_duration_sec = len(audio_data) / SAMPLE_RATE
        chunk_duration_min = chunk_duration_sec / 60.0
        
        wpm = round(float(stt_result.get("word_count", 0)) / float(chunk_duration_min), 1) if chunk_duration_min > 0 else 0.0 # type: ignore
        filler_result = detect_fillers(stt_result["text"])
        keywords_result = detect_keywords(stt_result["text"])
        clarity_result = analyze_clarity(stt_result["text"], float(filler_result["filler_rate"]))
        
        feedback = generate_feedback(
            energy=audio_metrics["energy"],
            mean_pitch=audio_metrics["mean_pitch"],
            pitch_std=audio_metrics["pitch_std"],
            wpm=wpm,
            filler_rate=filler_result["filler_rate"],
            pause_ratio=audio_metrics["pause_ratio"],
        )

        confidence = calculate_confidence(
            wpm=wpm,
            pitch_std=audio_metrics["pitch_std"],
            energy=audio_metrics["energy"],
            filler_rate=filler_result["filler_rate"],
        )

    # Build primary message (Personalized LLM Insight)
    primary_msg = "Analyzing..."
    transcript = stt_result.get("text", "")
    
    if transcript:
        try:
            from backend.services.llm_feedback import generate_llm_insight # type: ignore
            llm_insight = generate_llm_insight(transcript, {
                "wpm": wpm,
                "pitch_std": audio_metrics["pitch_std"],
                "total_fillers": filler_result["total_fillers"]
            })
            if llm_insight:
                primary_msg = llm_insight
        except Exception as e:
            logger.debug(f"LLM Insight skipped: {e}")

    # Fallback to rule-based if no LLM insight
    messages_list = feedback.get("messages", [])
    if primary_msg == "Analyzing..." and isinstance(messages_list, list) and messages_list:
        for msg in messages_list:
            if isinstance(msg, dict) and msg.get("type") == "warning":
                primary_msg = str(msg.get("text", "Analyzing..."))
                break
        else:
            first_msg = messages_list[0]
            if isinstance(first_msg, dict):
                primary_msg = str(first_msg.get("text", "Analyzing..."))

    return {
        "message": primary_msg,
        "pitch": audio_metrics["mean_pitch"],
        "pitch_std": audio_metrics["pitch_std"],
        "energy": audio_metrics["energy"],
        "wpm": wpm,
        "fillers": {
            "total_fillers": filler_result["total_fillers"],
            "filler_rate": filler_result["filler_rate"],
            "details": filler_result.get("details", {}),
        },
        "keywords": keywords_result,
        "clarity": clarity_result["clarity_score"],
        "lexical_diversity": clarity_result["lexical_diversity"],
        "transcript": transcript,
        "confidence_score": confidence["score"],
        "confidence_breakdown": confidence["breakdown"],
        "feedback": feedback.get("messages", []),
        "status": feedback.get("status", "info"),
        "pause_ratio": audio_metrics["pause_ratio"],
    }

@router.post("/analyze")
async def analyze_blob(audio_file: UploadFile = File(...)):
    """Accepts a WAV audio chunk and runs analysis without librosa."""
    try:
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            return JSONResponse({"message": "Empty file", "status": "error"}, status_code=400)
        
        # Robust loading using soundfile
        try:
            y, sr = sf.read(io.BytesIO(audio_bytes))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
        except Exception as read_err:
            logger.error(f"Soundfile read error: {read_err}")
            return JSONResponse({"message": "Audio format error.", "status": "error"}, status_code=400)

        if len(y) == 0:
            return JSONResponse({"message": "No audio.", "status": "warning"}, status_code=400)

        result = _perform_full_analysis(y)
        return JSONResponse(result)

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error("Analysis blob error:\n%s", error_msg)
        return JSONResponse({
            "message": "Acoustic analysis failed internally.",
            "error_detail": error_msg,
            "status": "error"
        }, status_code=500)

@router.get("/status")
async def get_status():
    return JSONResponse({"status": "ready", "mode": "deployment"})
