"""
SpeakTwin - Filler Word Detection Service
============================================
Detects common filler words/phrases in transcribed text and
computes frequency metrics.
"""

import re
from backend.utils.helpers import get_logger, FILLER_WORDS

logger = get_logger(__name__)


def detect_fillers(text: str) -> dict:
    """
    Scan transcribed text for filler words.

    Parameters
    ----------
    text : str
        The transcription from the STT engine.

    Returns
    -------
    dict with keys:
        total_fillers : int   – total filler occurrences
        filler_rate   : float – fillers / total words
        details       : dict  – per-filler breakdown  {word: count}
        total_words   : int
    """
    if not text or not text.strip():
        return {
            "total_fillers": 0,
            "filler_rate": 0.0,
            "details": {},
            "total_words": 0,
        }

    # Normalise text for matching
    normalised = text.lower().strip()
    words = normalised.split()
    total_words = len(words)

    details: dict[str, int] = {}
    total_fillers = 0

    # Check multi-word fillers first (e.g. "you know", "kind of")
    # then single-word fillers
    for filler in sorted(FILLER_WORDS, key=lambda f: -len(f.split())):
        if " " in filler:
            # Multi-word filler: use regex word-boundary matching
            pattern = r"\b" + re.escape(filler) + r"\b"
            matches = re.findall(pattern, normalised)
            count = len(matches)
        else:
            # Single-word filler
            count = sum(1 for w in words if w.strip(".,!?;:") == filler)

        if count > 0:
            details[filler] = count
            total_fillers += count

    filler_rate = round(total_fillers / total_words, 4) if total_words > 0 else 0.0

    return {
        "total_fillers": total_fillers,
        "filler_rate": filler_rate,
        "details": details,
        "total_words": total_words,
    }
