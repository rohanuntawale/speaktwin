"""
SpeakTwin - Prosody Features (openSMILE eGeMAPS)
=================================================
eGeMAPS is the standard compact acoustic parameter set for paralinguistics
- 88 features covering loudness, pitch contour, jitter, shimmer, spectral
slope, and formants.

Two uses here:

  * a few features are directly meaningful to a speaker (jitter/shimmer
    read as vocal strain; loudness variability reads as dynamism)
  * the full vector is the natural input to a *learned* confidence scorer,
    which is what would eventually replace the hand-weighted average in
    `confidence_score.py`

openSMILE is a compiled binary wrapped by the `opensmile` package; it is
not a neural model and needs no GPU.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np  # type: ignore

from backend.services.ml.registry import registry  # type: ignore
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import SAMPLE_RATE, get_logger  # type: ignore

logger = get_logger(__name__)

MODEL_KEY = "prosody"

# The handful of eGeMAPS functionals worth showing a speaker directly.
HIGHLIGHT_FEATURES = {
    "jitterLocal_sma3nz_amean": "jitter",
    "shimmerLocaldB_sma3nz_amean": "shimmer",
    "HNRdBACF_sma3nz_amean": "harmonics_to_noise",
    "loudness_sma3_stddevNorm": "loudness_variation",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "pitch_variation_norm",
    "slopeV0-500_sma3nz_amean": "spectral_slope",
}


def _load():
    import opensmile  # type: ignore

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


registry.register(
    MODEL_KEY,
    "openSMILE eGeMAPS prosody features",
    "requirements-ml.txt",
    _load,
)


def is_enabled() -> bool:
    return get_settings().ml_prosody_enabled


def extract(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[Dict[str, Any]]:
    """
    Compute eGeMAPS functionals for the chunk.

    Returns the readable highlights plus, when `ML_PROSODY_FULL_VECTOR` is
    on, the complete 88-dimensional vector for downstream model training.
    """
    if not is_enabled():
        return None

    smile = registry.get(MODEL_KEY)
    if smile is None:
        return None

    try:
        with registry.infer_lock(MODEL_KEY):
            frame = smile.process_signal(
                np.ascontiguousarray(audio, dtype=np.float32), sr
            )
    except Exception as exc:
        logger.warning("openSMILE extraction failed: %s", exc)
        return None

    if frame is None or frame.empty:
        return None

    row = frame.iloc[0]
    values = {name: float(row[name]) for name in frame.columns}

    result: Dict[str, Any] = {
        "engine": "opensmile_egemaps_v02",
        "feature_count": len(values),
    }
    for raw_name, friendly in HIGHLIGHT_FEATURES.items():
        if raw_name in values:
            result[friendly] = round(values[raw_name], 4)

    if get_settings().ml_prosody_full_vector:
        result["features"] = {k: round(v, 6) for k, v in values.items()}

    return result
