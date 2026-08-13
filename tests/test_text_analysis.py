"""Filler detection, keyword detection, and clarity scoring."""

from __future__ import annotations

import pytest

from backend.services.clarity_analysis import analyze_clarity, moving_average_ttr
from backend.services.filler_detection import detect_fillers
from backend.services.keyword_detection import detect_keywords
from backend.utils import text as tk


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------
def test_tokenizer_keeps_punctuation_separate():
    assert tk.tokenize("So, we shipped it.") == ["so", ",", "we", "shipped", "it", "."]


def test_tokenizer_keeps_contractions_and_hyphens_intact():
    assert tk.words("don't uh-huh") == ["don't", "uh-huh"]


def test_ngram_does_not_span_punctuation():
    tokens = tk.tokenize("you, know")
    assert tk.ngram_at(tokens, 0, 3) is None


# ---------------------------------------------------------------------------
# Fillers - always-fillers
# ---------------------------------------------------------------------------
def test_hesitation_sounds_always_count():
    result = detect_fillers("um I think uh this works")
    assert result["total_fillers"] == 2
    assert result["details"] == {"um": 1, "uh": 1}


def test_multiword_filler_beats_its_prefix():
    result = detect_fillers("it was, you know, fine")
    assert result["details"] == {"you know": 1}
    assert result["total_fillers"] == 1


def test_empty_text_is_safe():
    result = detect_fillers("")
    assert result == {"total_fillers": 0, "filler_rate": 0.0,
                      "details": {}, "total_words": 0}


def test_filler_rate_uses_word_count_not_token_count():
    result = detect_fillers("um, ok.")
    assert result["total_words"] == 2      # punctuation excluded
    assert result["filler_rate"] == 0.5


# ---------------------------------------------------------------------------
# Fillers - context-dependent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("So the solution is ready", 1),      # clause-initial -> filler
    ("We shipped it so the team relaxed", 0),  # conjunction -> not a filler
    ("Well, that went badly", 1),         # clause-initial -> filler
    ("It went well today", 0),            # adverb -> not a filler
])
def test_clause_initial_rule(text, expected):
    assert detect_fillers(text)["total_fillers"] == expected


@pytest.mark.parametrize("text,expected", [
    ("That is exactly right", 1),         # clause-final -> filler
    ("Nice work, right?", 1),             # clause-final -> filler
    ("The right answer matters", 0),      # adjective -> not a filler
])
def test_clause_final_rule(text, expected):
    assert detect_fillers(text)["total_fillers"] == expected


@pytest.mark.parametrize("text,expected", [
    ("it was like really fast", 1),       # discourse marker -> filler
    ("I like this approach", 0),          # verb -> not a filler
    ("We would like to ship", 0),         # verb phrase -> not a filler
    ("it works just like that", 0),       # comparative -> not a filler
    ("something like to do", 0),          # followed by "to" -> not a filler
])
def test_like_rule(text, expected):
    assert detect_fillers(text)["total_fillers"] == expected


def test_context_rules_combine_in_one_pass():
    text = "So um it was like really good, you know, right?"
    result = detect_fillers(text)
    assert result["details"] == {
        "so": 1, "um": 1, "like": 1, "you know": 1, "right": 1,
    }


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------
def test_keyword_detection_counts_occurrences():
    result = detect_keywords("Our AI strategy drives growth and growth")
    assert result["found_keywords"] == {"ai": 1, "strategy": 1, "growth": 2}
    assert result["total_keywords"] == 4


def test_multiword_keyword_not_double_counted():
    result = detect_keywords("artificial intelligence is the vision")
    assert result["found_keywords"] == {"artificial intelligence": 1, "vision": 1}
    assert result["total_keywords"] == 2


def test_keyword_matching_ignores_substrings():
    # "AIn't" tokenises to "ain't", which must not register as "ai"
    assert detect_keywords("ain't nobody")["total_keywords"] == 0


def test_keyword_empty_text():
    assert detect_keywords("   ")["total_keywords"] == 0


# ---------------------------------------------------------------------------
# Clarity
# ---------------------------------------------------------------------------
def test_mattr_falls_back_to_ttr_for_short_text():
    assert moving_average_ttr(["a", "b", "c", "d"]) == 1.0
    assert moving_average_ttr(["a", "a", "b", "b"]) == 0.5


def test_mattr_is_stable_as_text_grows():
    """The whole point of MATTR: length must not drive the score."""
    vocabulary = [f"w{i}" for i in range(20)]
    short = vocabulary * 3    # 60 words
    long = vocabulary * 12    # 240 words
    assert moving_average_ttr(short) == pytest.approx(moving_average_ttr(long), abs=0.02)


def test_clarity_penalises_fillers():
    text = "we deliver innovative measurable outcomes for every partner team"
    clean = analyze_clarity(text, 0.0)["clarity_score"]
    filled = analyze_clarity(text, 0.15)["clarity_score"]
    assert filled < clean


def test_clarity_score_stays_in_range():
    assert analyze_clarity("word word word", 5.0)["clarity_score"] == 0
    assert 0 <= analyze_clarity("a diverse varied rich lexicon", 0.0)["clarity_score"] <= 100


def test_clarity_empty_text():
    assert analyze_clarity("", 0.0)["clarity_score"] == 0
