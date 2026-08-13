"""
SpeakTwin - Filler Word Detection
==================================
Scans a transcript for filler words and computes frequency metrics.

Detection is context-aware. Hesitation sounds ("um", "uh") are always
fillers, but discourse markers have legitimate lexical uses and are only
counted when they sit in a filler position:

    "So the solution is..."   -> "so" is a filler (clause-initial)
    "...so we shipped it"     -> "so" is a conjunction, not counted
    "I like this approach"    -> "like" is a verb, not counted
    "it was like really fast" -> "like" is a filler
    "exactly right"           -> "right" is a filler (clause-final)
    "the right answer"        -> "right" is an adjective, not counted
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.utils import text as tk  # type: ignore
from backend.utils.helpers import (  # type: ignore
    get_logger,
    ALWAYS_FILLERS,
    CONTEXTUAL_FILLERS,
    FILLER_PHRASES,
    MAX_FILLER_NGRAM,
    LIKE_LEXICAL_PREV,
    LIKE_LEXICAL_NEXT,
)

logger = get_logger(__name__)

_ALWAYS = frozenset(ALWAYS_FILLERS)


def _is_filler_here(phrase: str, tokens: List[str], index: int) -> bool:
    """Decide whether `phrase` at `index` is being used as a filler."""
    if phrase in _ALWAYS:
        return True

    rule = CONTEXTUAL_FILLERS.get(phrase)

    if rule == "clause_initial":
        return tk.at_clause_start(tokens, index)

    if rule == "clause_final":
        return tk.at_clause_end(tokens, index)

    if rule == "like_rule":
        previous = tk.prev_word(tokens, index)
        if previous in LIKE_LEXICAL_PREV:
            return False          # "I like", "would like", "just like"
        following = tk.next_token(tokens, index)
        if following in LIKE_LEXICAL_NEXT:
            return False          # "like to", "like this"
        return True

    # Unknown rule - be conservative and do not penalise the speaker.
    return False


def detect_fillers(text: str) -> Dict[str, Any]:
    """
    Scan transcribed text for filler words.

    Returns
    -------
    dict with keys:
        total_fillers : int   - total filler occurrences
        filler_rate   : float - fillers / total words
        details       : dict  - per-filler breakdown {phrase: count}
        total_words   : int
    """
    empty = {"total_fillers": 0, "filler_rate": 0.0, "details": {}, "total_words": 0}
    if not text or not text.strip():
        return empty

    tokens = tk.tokenize(text)
    total_words = sum(1 for t in tokens if tk.is_word(t))
    if total_words == 0:
        return empty

    details: Dict[str, int] = {}
    total_fillers = 0

    index = 0
    while index < len(tokens):
        if not tk.is_word(tokens[index]):
            index += 1
            continue

        # Longest match first so "you know" beats a bare "know".
        matched_size = 0
        for size in range(MAX_FILLER_NGRAM, 0, -1):
            phrase = tk.ngram_at(tokens, index, size)
            if phrase is None or phrase not in FILLER_PHRASES:
                continue
            if _is_filler_here(phrase, tokens, index):
                details[phrase] = details.get(phrase, 0) + 1
                total_fillers += 1
                matched_size = size
            else:
                # Recognised phrase used lexically - skip it without
                # counting, but do not re-test its shorter prefixes.
                matched_size = size
            break

        index += matched_size if matched_size else 1

    return {
        "total_fillers": total_fillers,
        "filler_rate": round(total_fillers / total_words, 4),
        "details": details,
        "total_words": total_words,
    }
