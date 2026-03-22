"""
SpeakTwin - Keyword Detection Service
============================================
Scans transcribed text for target keywords to encourage positive 
reinforcement of impactful language.
"""
import re
import typing
from backend.utils.helpers import get_logger, TARGET_KEYWORDS # type: ignore

logger = get_logger(__name__)

def detect_keywords(text: str) -> typing.Dict[str, typing.Any]:
    """
    Find occurrences of predefined target keywords in the transcript.
    """
    if not text or not text.strip():
        return {"total_keywords": 0, "found_keywords": {}, "keywords_list": []}

    normalised = text.lower().strip()
    words = normalised.split()
    
    found_keywords: typing.Dict[str, int] = {}
    total_keywords: int = 0

    for keyword in sorted(TARGET_KEYWORDS, key=lambda k: -len(k.split())):
        count: int = 0
        if " " in keyword:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            matches = re.findall(pattern, normalised)
            count = len(matches)
        else:
            count = sum(1 for w in words if w.strip(".,!?;:") == keyword)

        if count > 0:
            found_keywords[keyword] = count
            total_keywords = total_keywords + count # type: ignore

    return {
        "total_keywords": total_keywords,
        "found_keywords": found_keywords,
        "keywords_list": list(found_keywords.keys())
    }
