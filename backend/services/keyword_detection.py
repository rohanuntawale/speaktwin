"""
SpeakTwin - Keyword Detection
==============================
Scans a transcript for target keywords, so impactful language gets
positive reinforcement rather than only mistakes being surfaced.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.utils import text as tk  # type: ignore
from backend.utils.helpers import (  # type: ignore
    get_logger,
    TARGET_KEYWORDS,
    MAX_KEYWORD_NGRAM,
)

logger = get_logger(__name__)

_KEYWORDS = frozenset(TARGET_KEYWORDS)


def detect_keywords(text: str) -> Dict[str, Any]:
    """Find occurrences of the configured target keywords in `text`."""
    empty = {"total_keywords": 0, "found_keywords": {}, "keywords_list": []}
    if not text or not text.strip():
        return empty

    tokens = tk.tokenize(text)
    found: Dict[str, int] = {}
    total = 0

    index = 0
    while index < len(tokens):
        if not tk.is_word(tokens[index]):
            index += 1
            continue

        # Longest match first: "artificial intelligence" should not also
        # register as two separate one-word hits.
        matched_size = 0
        for size in range(MAX_KEYWORD_NGRAM, 0, -1):
            phrase = tk.ngram_at(tokens, index, size)
            if phrase is not None and phrase in _KEYWORDS:
                found[phrase] = found.get(phrase, 0) + 1
                total += 1
                matched_size = size
                break

        index += matched_size if matched_size else 1

    return {
        "total_keywords": total,
        "found_keywords": found,
        "keywords_list": list(found.keys()),
    }
