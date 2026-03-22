"""
SpeakTwin - LLM Feedback Engine (OpenRouter)
=============================================
Uses OpenRouter (e.g., openai/gpt-5.4) to generate highly personalized,
concise coaching tips based on the user's transcript and acoustic metrics.
"""

import os
from openai import OpenAI # type: ignore
from backend.utils.helpers import get_logger # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

logger = get_logger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.4")

client = None
if OPENROUTER_API_KEY:
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client for OpenRouter: {e}")

from typing import Optional

def generate_llm_insight(transcript: str, metrics: dict) -> Optional[str]:
    """
    Calls OpenRouter LLM to generate a single, highly actionable coach insight.
    Returns None if the API fails or is not configured.
    """
    if client is None or not transcript.strip():
        return None
        
    prompt = f"""You are an expert public speaking communication coach AI named SpeakTwin.
Analyze this speech and its metrics:
Transcript: "{transcript}"
WPM: {metrics.get('wpm', 0)} | Pitch Var: {metrics.get('pitch_std', 0):.1f} | Fillers: {metrics.get('total_fillers', 0)}

Provide ONLY ONE ULTRA-CONCISE phrase (MAX 8 WORDS) of coaching. Be punchy and direct.
Examples: 
- "Great pace! Watch the filler words."
- "Good vocal variety, stay expressive!"
- "Slow down and pause more."

"""
    try:
        # type: ignore (Pylance/Pyre might map None to client somehow if uninitialized)
        response = client.chat.completions.create( # type: ignore
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.7
        )
        insight = response.choices[0].message.content.strip().replace('"', '')
        return insight
    except Exception as e:
        logger.error(f"OpenRouter LLM feedback generation failed: {e}")
        return None
