"""
llm.py – Shared Google Gemini client.

A single `genai.Client` is created lazily and reused across the whole app instead
of being reconstructed in every agent, tool, and helper. Use `get_client()`
everywhere a Gemini client is needed.
"""
from __future__ import annotations

import functools

from google import genai

from rag import config


@functools.lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Return the process-wide singleton Gemini client."""
    return genai.Client(api_key=config.GEMINI_API_KEY)
