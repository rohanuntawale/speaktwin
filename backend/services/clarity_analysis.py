"""
SpeakTwin - Clarity Analysis Service
============================================
Computes language clarity metrics such as Lexical Diversity
to determine how rich and articulate the transcription is.
"""
import typing
from backend.utils.helpers import get_logger # type: ignore

logger = get_logger(__name__)

def analyze_clarity(text: str, filler_rate: float) -> typing.Dict[str, typing.Any]:
    """
    Calculate the Clarity Score of the transcription based on
    lexical diversity and penalty from fillers.
    """
    if not text or not text.strip():
        return {"lexical_diversity": 0.0, "clarity_score": 0}

    normalised = text.lower().strip()
    
    # Strip common punctuation for accurate word counting
    import re
    cleaned_text = re.sub(r'[^\w\s]', '', normalised)
    words = cleaned_text.split()
    
    total_words = len(words)
    
    if total_words == 0:
        return {"lexical_diversity": 0.0, "clarity_score": 0}

    unique_words = len(set(words))
    lexical_diversity = round(float(unique_words) / float(total_words), 2) # type: ignore
    
    # Generate a clarity score (0-100) based on diversity and fillers
    # Base clarity from lexical diversity (typically ~0.4 to 0.8 in natural speech depending on length)
    # We normalize it so 0.75+ is 100 for short snippets
    normalized_diversity = min(1.0, float(lexical_diversity) / 0.75) * 100.0
    
    clarity_score = normalized_diversity - (float(filler_rate) * 300.0)
    clarity_score = max(0.0, min(100.0, clarity_score))
    
    return {
        "lexical_diversity": lexical_diversity,
        "clarity_score": int(round(clarity_score))
    }
