"""
SpeakTwin - LLM Coaching Insight (via OpenRouter)
==================================================
Turns the transcript plus its metrics into one punchy coaching line.

This call sits on the critical path of every analysed chunk, so it is
defended on three fronts:
  * a hard timeout and a single retry, so a slow provider cannot stall
    the request indefinitely
  * a per-session minimum interval, so a 2.5s chunk cadence does not
    translate into ~24 paid calls per minute per speaker
  * a transcript length cap, so prompt size stays bounded

Every failure path returns None and the caller falls back to rule-based
feedback.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import get_logger  # type: ignore

logger = get_logger(__name__)

_client: Any = None
_client_lock = threading.Lock()

# session_id -> monotonic timestamp of the last call
_last_call: Dict[str, float] = {}
_throttle_lock = threading.Lock()

MAX_INSIGHT_WORDS = 12
_GLOBAL_KEY = "__global__"


def _get_client():
    """Lazily build the OpenRouter client (OpenAI-compatible endpoint)."""
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if not settings.llm_enabled:
        return None

    with _client_lock:
        if _client is not None:
            return _client
        try:
            from openai import OpenAI  # type: ignore

            _client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
                timeout=settings.llm_timeout_seconds,
                max_retries=1,
            )
        except Exception as exc:
            logger.warning("Could not initialise the OpenRouter client: %s", exc)
            _client = None

    return _client


def is_enabled() -> bool:
    """Whether LLM coaching is configured and usable."""
    return get_settings().llm_enabled


def should_call(session_id: Optional[str] = None) -> bool:
    """
    Rate-gate the LLM per session.

    Returns True at most once every `LLM_MIN_INTERVAL_SECONDS`, and reserves
    the slot as a side effect so two concurrent chunks cannot both pass.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return False

    interval = settings.llm_min_interval_seconds
    if interval <= 0:
        return True

    key = session_id or _GLOBAL_KEY
    now = time.monotonic()

    with _throttle_lock:
        previous = _last_call.get(key)
        if previous is not None and (now - previous) < interval:
            return False
        _last_call[key] = now

        # Keep the throttle map from growing without bound.
        if len(_last_call) > 1000:
            cutoff = now - max(interval * 10, 600)
            for stale in [k for k, ts in _last_call.items() if ts < cutoff]:
                _last_call.pop(stale, None)

    return True


def forget_session(session_id: str) -> None:
    """Drop a session's throttle entry when the session ends."""
    with _throttle_lock:
        _last_call.pop(session_id, None)


def _build_prompt(transcript: str, metrics: Dict[str, Any]) -> str:
    settings = get_settings()
    clipped = transcript.strip()
    if len(clipped) > settings.llm_max_transcript_chars:
        clipped = clipped[-settings.llm_max_transcript_chars:]

    return (
        "You are an expert public speaking communication coach AI named SpeakTwin.\n"
        "Analyze this speech and its metrics:\n"
        f'Transcript: "{clipped}"\n'
        f"WPM: {metrics.get('wpm', 0)} | "
        f"Pitch Var: {float(metrics.get('pitch_std', 0) or 0):.1f} | "
        f"Fillers: {metrics.get('total_fillers', 0)}\n\n"
        "Provide ONLY ONE ULTRA-CONCISE phrase (MAX 8 WORDS) of coaching. "
        "Be punchy and direct.\n"
        "Examples:\n"
        '- "Great pace! Watch the filler words."\n'
        '- "Good vocal variety, stay expressive!"\n'
        '- "Slow down and pause more."\n'
    )


def _tidy(raw: str) -> Optional[str]:
    """Take the first line, drop quoting, and enforce the length cap."""
    text = (raw or "").strip().splitlines()
    if not text:
        return None

    insight = text[0].strip().strip('"').strip("'").lstrip("-").strip()
    if not insight:
        return None

    words = insight.split()
    if len(words) > MAX_INSIGHT_WORDS:
        insight = " ".join(words[:MAX_INSIGHT_WORDS]).rstrip(",;:") + "..."
    return insight


def generate_llm_insight(transcript: str, metrics: Dict[str, Any],
                         session_id: Optional[str] = None) -> Optional[str]:
    """
    Generate one actionable coaching line, or None if unavailable.

    The caller is expected to gate on `should_call()` first; this function
    re-checks configuration but not the throttle, so a caller that has
    already reserved a slot is not double-charged.
    """
    if not transcript or not transcript.strip():
        return None

    client = _get_client()
    if client is None:
        return None

    settings = get_settings()
    try:
        response = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[{"role": "user", "content": _build_prompt(transcript, metrics)}],
            max_tokens=40,
            temperature=0.7,
        )
    except Exception as exc:
        logger.warning("OpenRouter insight failed (session=%s): %s",
                       session_id or "-", exc)
        return None

    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        content = choices[0].message.content
        return _tidy(content) if content else None
    except Exception as exc:
        logger.warning("Unexpected OpenRouter response shape: %s", exc)
        return None
