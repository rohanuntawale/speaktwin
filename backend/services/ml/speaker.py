"""
SpeakTwin - Speaker Embeddings & Diarization
=============================================
Two related capabilities, both keyed off speaker identity:

  * **ECAPA-TDNN embeddings** (SpeechBrain) - a 192-dim vector per chunk.
    Comparing it across a session verifies the same person is speaking, and
    comparing across sessions is what would let SpeakTwin track one
    speaker's progress over time.

  * **Diarization** (pyannote) - who spoke when. Needed the moment more
    than one voice is present: practice interviews, panel rehearsals, or
    simply a colleague talking in the room. Without it, every metric is
    computed over a blend of voices.

pyannote checkpoints are gated on the Hugging Face Hub: you must accept
the model terms on the model page and supply `HF_TOKEN`. That is a
click-through licence, so it cannot be automated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

from backend.services.ml.registry import registry  # type: ignore
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import SAMPLE_RATE, get_logger  # type: ignore

logger = get_logger(__name__)

EMBEDDING_KEY = "speaker_embedding"
DIARIZATION_KEY = "diarization"
MODEL_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Speaker embeddings
# ---------------------------------------------------------------------------
def _load_embedding():
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

    settings = get_settings()
    kwargs = {
        "source": settings.ml_speaker_model,
        "savedir": f".models/{settings.ml_speaker_model.replace('/', '_')}",
        "run_opts": {"device": registry.device()},
    }

    # SpeechBrain symlinks from the Hugging Face cache into `savedir`, and
    # Windows refuses to create symlinks without Developer Mode or admin
    # rights ("WinError 1314: A required privilege is not held"). Ask for
    # real copies instead; older SpeechBrain releases lack the option, so
    # fall back to the default behaviour.
    try:
        from speechbrain.utils.fetching import LocalStrategy  # type: ignore
        return EncoderClassifier.from_hparams(
            **kwargs, local_strategy=LocalStrategy.COPY_SKIP_CACHE
        )
    except (ImportError, AttributeError, TypeError):
        return EncoderClassifier.from_hparams(**kwargs)


registry.register(
    EMBEDDING_KEY,
    "ECAPA-TDNN speaker embeddings",
    "requirements-ml.txt",
    _load_embedding,
)


def embeddings_enabled() -> bool:
    return get_settings().ml_speaker_enabled


def embed(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[List[float]]:
    """Return a speaker embedding vector for the chunk, or None."""
    if not embeddings_enabled():
        return None

    encoder = registry.get(EMBEDDING_KEY)
    if encoder is None or sr != MODEL_SAMPLE_RATE:
        return None

    # Too short a window produces an unstable embedding.
    if len(audio) < MODEL_SAMPLE_RATE:
        return None

    try:
        import torch  # type: ignore

        tensor = torch.from_numpy(
            np.ascontiguousarray(audio, dtype=np.float32)
        ).unsqueeze(0)

        with registry.infer_lock(EMBEDDING_KEY):
            with torch.no_grad():
                vector = encoder.encode_batch(tensor)

        return [round(float(v), 6) for v in vector.squeeze().cpu().numpy()]

    except Exception as exc:
        logger.warning("Speaker embedding failed: %s", exc)
        return None


def cosine_similarity(left: List[float], right: List[float]) -> float:
    """Similarity of two embeddings; ~0.75+ typically means same speaker."""
    if not left or not right or len(left) != len(right):
        return 0.0
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return round(float(np.dot(a, b) / denominator), 4)


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------
def _load_diarization():
    from pyannote.audio import Pipeline  # type: ignore

    settings = get_settings()
    if not settings.hf_token:
        raise RuntimeError(
            "HF_TOKEN is required for pyannote. Accept the model terms at "
            f"https://hf.co/{settings.ml_diarization_model} and set HF_TOKEN."
        )

    pipeline = Pipeline.from_pretrained(
        settings.ml_diarization_model,
        use_auth_token=settings.hf_token,
    )

    try:
        import torch  # type: ignore
        pipeline.to(torch.device(registry.device()))
    except Exception:
        pass  # CPU is fine

    return pipeline


registry.register(
    DIARIZATION_KEY,
    "pyannote speaker diarization",
    "requirements-ml.txt",
    _load_diarization,
)


def diarization_enabled() -> bool:
    return get_settings().ml_diarization_enabled


def diarize(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[Dict[str, Any]]:
    """
    Segment the audio by speaker.

    Best run over a whole session rather than per chunk - 2.5s is too short
    to separate voices reliably, and the pipeline is comparatively slow.
    """
    if not diarization_enabled():
        return None

    pipeline = registry.get(DIARIZATION_KEY)
    if pipeline is None:
        return None

    try:
        import torch  # type: ignore

        tensor = torch.from_numpy(
            np.ascontiguousarray(audio, dtype=np.float32)
        ).unsqueeze(0)

        with registry.infer_lock(DIARIZATION_KEY):
            annotation = pipeline({"waveform": tensor, "sample_rate": sr})

    except Exception as exc:
        logger.warning("Diarization failed: %s", exc)
        return None

    segments: List[Dict[str, Any]] = []
    speakers: Dict[str, float] = {}

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        duration = float(turn.end - turn.start)
        segments.append({
            "speaker": str(speaker),
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
        })
        speakers[str(speaker)] = speakers.get(str(speaker), 0.0) + duration

    total = sum(speakers.values()) or 1.0
    dominant = max(speakers.items(), key=lambda kv: kv[1])[0] if speakers else None

    return {
        "speaker_count": len(speakers),
        "segments": segments,
        "speaking_share": {k: round(v / total, 3) for k, v in speakers.items()},
        "dominant_speaker": dominant,
        "engine": "pyannote",
    }
