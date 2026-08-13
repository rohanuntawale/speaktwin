"""
SpeakTwin - Text Tokenisation
==============================
One tokeniser shared by filler detection, keyword detection, and clarity
analysis so all three agree on what counts as a word.

Punctuation is preserved as its own token because filler rules need to know
where clauses begin and end ("so" at the start of a clause is a filler;
"so" in the middle usually is not).
"""

from __future__ import annotations

import re
from typing import List

# Words may contain internal apostrophes ("don't") and hyphens ("uh-huh").
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*|[.?!,;:]")

PUNCTUATION = frozenset({".", "?", "!", ",", ";", ":"})
CLAUSE_BREAKS = PUNCTUATION  # any of these opens a new clause


def tokenize(text: str) -> List[str]:
    """Lower-case token list including punctuation marks."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def words(text: str) -> List[str]:
    """Lower-case word tokens only, punctuation stripped."""
    return [t for t in tokenize(text) if t not in PUNCTUATION]


def word_count(text: str) -> int:
    """Number of word tokens in `text`."""
    return len(words(text))


def is_word(token: str) -> bool:
    return token not in PUNCTUATION


def ngram_at(tokens: List[str], index: int, size: int) -> str | None:
    """
    Join `size` consecutive tokens starting at `index` into a phrase.

    Returns None if the span runs past the end or contains punctuation,
    which keeps phrases from spanning a clause boundary.
    """
    end = index + size
    if end > len(tokens):
        return None
    span = tokens[index:end]
    if any(t in PUNCTUATION for t in span):
        return None
    return " ".join(span)


def prev_word(tokens: List[str], index: int) -> str | None:
    """Nearest word token before `index`, skipping punctuation."""
    for i in range(index - 1, -1, -1):
        if tokens[i] not in PUNCTUATION:
            return tokens[i]
    return None


def next_token(tokens: List[str], index: int) -> str | None:
    """The very next token (punctuation included) after `index`."""
    return tokens[index + 1] if index + 1 < len(tokens) else None


def at_clause_start(tokens: List[str], index: int) -> bool:
    """True when `index` is the first word of the text or of a new clause."""
    if index == 0:
        return True
    return tokens[index - 1] in CLAUSE_BREAKS


def at_clause_end(tokens: List[str], index: int) -> bool:
    """True when the token at `index` is the last word of a clause."""
    nxt = next_token(tokens, index)
    return nxt is None or nxt in CLAUSE_BREAKS
