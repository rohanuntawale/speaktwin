"""
SpeakTwin - Neural Pitch Tracking (CREPE)
==========================================
CREPE is a CNN trained to read f0 directly from the waveform. It is
markedly more reliable than autocorrelation on real speech, which is noisy,
breathy, and full of the octave ambiguities that wreck signal-processing
pitch trackers.

`torchcrepe` ships several capacities. `tiny` runs comfortably inside a
2.5s chunk budget on CPU; `full` is more accurate but slower.

The scipy tracker in `audio_analysis` stays as the fallback, so pitch is
always produced - this only upgrades its quality when available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np  # type: ignore

from backend.services.ml.registry import registry  # type: ignore
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import (  # type: ignore
    PITCH_MAX_HZ,
    PITCH_MIN_HZ,
    SAMPLE_RATE,
    get_logger,
)

logger = get_logger(__name__)

MODEL_KEY = "crepe"
# torchcrepe is trained at 16 kHz, which is already our pipeline rate.
CREPE_SAMPLE_RATE = 16_000
# Frames whose periodicity falls below this are treated as unvoiced.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def _load():
    import torch  # type: ignore  # noqa: F401
    import torchcrepe  # type: ignore

    # torchcrepe loads weights on first use; force it now so the cost is
    # paid at startup rather than inside the first request.
    torchcrepe.load.model(device=registry.device(),
                          capacity=get_settings().ml_crepe_capacity)
    return torchcrepe


registry.register(
    MODEL_KEY,
    "CREPE neural pitch tracker",
    "requirements-ml.txt",
    _load,
)


def is_enabled() -> bool:
    return get_settings().ml_pitch_enabled


def estimate_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[Dict[str, Any]]:
    """
    Estimate pitch statistics with CREPE.

    Returns the same shape as `audio_analysis.compute_pitch` plus a mean
    periodicity confidence, or None when unavailable so the caller can fall
    back to the DSP tracker.
    """
    if not is_enabled():
        return None

    torchcrepe = registry.get(MODEL_KEY)
    if torchcrepe is None:
        return None

    if sr != CREPE_SAMPLE_RATE:
        logger.debug("CREPE expects %d Hz, got %d - skipping", CREPE_SAMPLE_RATE, sr)
        return None

    settings = get_settings()

    try:
        import torch  # type: ignore

        device = registry.device()
        # torchcrepe wants a (1, samples) float32 tensor.
        tensor = torch.from_numpy(
            np.ascontiguousarray(audio, dtype=np.float32)
        ).unsqueeze(0)

        with registry.infer_lock(MODEL_KEY):
            with torch.no_grad():
                frequency, periodicity = torchcrepe.predict(
                    tensor,
                    CREPE_SAMPLE_RATE,
                    hop_length=int(CREPE_SAMPLE_RATE / 100),  # 10 ms frames
                    fmin=float(PITCH_MIN_HZ),
                    fmax=float(PITCH_MAX_HZ),
                    model=settings.ml_crepe_capacity,
                    batch_size=512,
                    device=device,
                    return_periodicity=True,
                )

        f0 = frequency.squeeze(0).cpu().numpy()
        confidence = periodicity.squeeze(0).cpu().numpy()

    except Exception as exc:
        logger.warning("CREPE inference failed: %s", exc)
        return None

    threshold = settings.ml_crepe_confidence
    voiced = (confidence >= threshold) & (f0 >= PITCH_MIN_HZ) & (f0 <= PITCH_MAX_HZ)
    total_frames = int(f0.size)

    if total_frames == 0 or not np.any(voiced):
        return {
            "mean_pitch": 0.0,
            "pitch_std": 0.0,
            "voiced_ratio": 0.0,
            "pitch_confidence": 0.0,
            "engine": "crepe",
        }

    values = f0[voiced]
    return {
        "mean_pitch": round(float(np.mean(values)), 2),
        "pitch_std": round(float(np.std(values)), 2),
        "voiced_ratio": round(float(np.count_nonzero(voiced) / total_frames), 3),
        "pitch_confidence": round(float(np.mean(confidence[voiced])), 3),
        "engine": "crepe",
    }
