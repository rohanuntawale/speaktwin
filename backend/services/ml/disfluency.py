"""
SpeakTwin - Acoustic Disfluency Detection
==========================================
Detects filler words from the *audio*, not the transcript.

This exists because Whisper was trained largely on cleaned transcripts and
routinely deletes "um" and "uh" outright - and enabling its VAD filter
trims the hesitation regions as well. Text-based filler counting is
therefore structurally under-counting, and no amount of better regex fixes
it. The signal only survives in the waveform.

The model is a frame-level classifier over a self-supervised speech
backbone (WavLM / wav2vec2). No suitable pretrained checkpoint is widely
published, so `training/train_filler_detector.py` fine-tunes one on
PodcastFillers; this module is the inference half of that pair.

Set `ML_DISFLUENCY_MODEL` to a local directory or a Hub repo id. Until one
is configured the feature stays off and text-based detection carries on
unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

from backend.services.ml.registry import registry  # type: ignore
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import SAMPLE_RATE, get_logger  # type: ignore

logger = get_logger(__name__)

MODEL_KEY = "disfluency"
MODEL_SAMPLE_RATE = 16_000

# Label set produced by the training script. Index 0 must be the negative
# class so `argmax == 0` means "nothing interesting here".
LABELS = ["none", "um", "uh", "repetition", "prolongation"]
FILLER_LABELS = {"um", "uh", "repetition", "prolongation"}


def _load():
    from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification  # type: ignore

    settings = get_settings()
    model_id = settings.ml_disfluency_model
    if not model_id:
        raise RuntimeError("ML_DISFLUENCY_MODEL is not set")

    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioFrameClassification.from_pretrained(model_id)
    model.eval()
    model.to(registry.device())

    # Prefer the checkpoint's own label map over our default.
    id2label = getattr(model.config, "id2label", None) or {}
    labels = [id2label.get(i, LABELS[i] if i < len(LABELS) else str(i))
              for i in range(model.config.num_labels)]

    return {"model": model, "extractor": extractor, "labels": labels}


registry.register(
    MODEL_KEY,
    "Acoustic filler / disfluency detector",
    "requirements-ml.txt",
    _load,
)


def is_enabled() -> bool:
    settings = get_settings()
    return settings.ml_disfluency_enabled and bool(settings.ml_disfluency_model)


def detect(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[Dict[str, Any]]:
    """
    Find filler events in the waveform.

    Returns per-event timings and a per-class tally that can be merged with
    the text-based counts, or None when unavailable.
    """
    if not is_enabled():
        return None

    bundle = registry.get(MODEL_KEY)
    if bundle is None:
        return None

    if sr != MODEL_SAMPLE_RATE:
        logger.debug("Disfluency model expects %d Hz, got %d", MODEL_SAMPLE_RATE, sr)
        return None

    try:
        import torch  # type: ignore

        inputs = bundle["extractor"](
            np.ascontiguousarray(audio, dtype=np.float32),
            sampling_rate=MODEL_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(registry.device()) for k, v in inputs.items()}

        with registry.infer_lock(MODEL_KEY):
            with torch.no_grad():
                logits = bundle["model"](**inputs).logits

        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    except Exception as exc:
        logger.warning("Disfluency inference failed: %s", exc)
        return None

    return _events_from_frames(
        probabilities,
        bundle["labels"],
        duration=len(audio) / float(sr),
        threshold=get_settings().ml_disfluency_threshold,
    )


def _events_from_frames(probabilities: np.ndarray, labels: List[str],
                        duration: float, threshold: float) -> Dict[str, Any]:
    """
    Collapse per-frame class probabilities into discrete events.

    Consecutive frames sharing a label become one event, which is what
    turns "40 frames of um" into a single counted filler.
    """
    num_frames = probabilities.shape[0]
    if num_frames == 0:
        return {"total_fillers": 0, "details": {}, "events": [], "engine": "acoustic"}

    seconds_per_frame = duration / num_frames
    best = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)

    events: List[Dict[str, Any]] = []
    details: Dict[str, int] = {}

    current_label: Optional[str] = None
    start_frame = 0
    scores: List[float] = []

    def close(end_frame: int) -> None:
        if current_label is None or not scores:
            return
        mean_score = float(np.mean(scores))
        if mean_score < threshold:
            return
        events.append({
            "label": current_label,
            "start": round(start_frame * seconds_per_frame, 3),
            "end": round(end_frame * seconds_per_frame, 3),
            "confidence": round(mean_score, 3),
        })
        details[current_label] = details.get(current_label, 0) + 1

    for frame in range(num_frames):
        label = labels[best[frame]] if best[frame] < len(labels) else "none"
        label = label if label in FILLER_LABELS else None

        if label != current_label:
            close(frame)
            current_label = label
            start_frame = frame
            scores = []

        if label is not None:
            scores.append(float(confidence[frame]))

    close(num_frames)

    return {
        "total_fillers": len(events),
        "details": details,
        "events": events,
        "engine": "acoustic",
    }


def merge_with_text(text_result: Dict[str, Any],
                    acoustic: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine transcript-based and audio-based filler counts.

    The acoustic count is taken as the floor rather than summing the two:
    when Whisper *does* keep an "um", both detectors see the same event and
    adding them would double-count it.
    """
    if not acoustic:
        return text_result

    merged = dict(text_result)
    text_total = int(text_result.get("total_fillers", 0))
    acoustic_total = int(acoustic.get("total_fillers", 0))
    total = max(text_total, acoustic_total)

    details = dict(text_result.get("details", {}))
    for label, count in acoustic.get("details", {}).items():
        details[label] = max(details.get(label, 0), count)

    total_words = int(text_result.get("total_words", 0))
    merged.update({
        "total_fillers": total,
        "details": details,
        "filler_rate": round(total / total_words, 4) if total_words else 0.0,
        "acoustic_fillers": acoustic_total,
        "text_fillers": text_total,
        "events": acoustic.get("events", []),
    })
    return merged
