"""
SpeakTwin - Clarity Analysis
=============================
Scores how varied and articulate a transcript is.

Plain type-token ratio (unique / total) is strongly length-dependent: a
five-word chunk almost always scores 1.0 while a hundred-word one rarely
clears 0.6, so a per-chunk score and a whole-session score are not
comparable on the same axis. The score is therefore built on MATTR
(moving-average TTR): the average TTR across a sliding fixed-width window,
which is stable as the text grows. Texts shorter than one window fall back
to plain TTR. Raw TTR is still reported for display.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.utils import text as tk  # type: ignore
from backend.utils.helpers import get_logger  # type: ignore

logger = get_logger(__name__)

MATTR_WINDOW = 25         # words per sliding window
DIVERSITY_TARGET = 0.75   # MATTR at or above this scores full marks
FILLER_PENALTY_WEIGHT = 300.0


def moving_average_ttr(words: List[str], window: int = MATTR_WINDOW) -> float:
    """
    Average type-token ratio across every sliding window of `window` words.

    Falls back to plain TTR when the text is shorter than one window.
    """
    total = len(words)
    if total == 0:
        return 0.0
    if total <= window:
        return len(set(words)) / total

    ratios = [
        len(set(words[i:i + window])) / window
        for i in range(total - window + 1)
    ]
    return sum(ratios) / len(ratios)


def analyze_clarity(text: str, filler_rate: float) -> Dict[str, Any]:
    """Calculate a 0-100 clarity score from lexical variety minus fillers."""
    empty = {"lexical_diversity": 0.0, "mattr": 0.0, "clarity_score": 0}
    if not text or not text.strip():
        return empty

    words = tk.words(text)
    total_words = len(words)
    if total_words == 0:
        return empty

    lexical_diversity = round(len(set(words)) / total_words, 2)
    mattr = moving_average_ttr(words)

    normalised = min(1.0, mattr / DIVERSITY_TARGET) * 100.0
    score = normalised - max(0.0, float(filler_rate)) * FILLER_PENALTY_WEIGHT
    score = max(0.0, min(100.0, score))

    return {
        "lexical_diversity": lexical_diversity,
        "mattr": round(mattr, 3),
        "clarity_score": int(round(score)),
    }
