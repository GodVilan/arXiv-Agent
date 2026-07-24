"""
generator.py – Gemini answer generator with rate limiting.
"""
import logging
import threading
import time

from google import genai
from google.genai import types

from rag import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research assistant with access to recent arXiv ML papers (cs.LG, 2026).
Answer based on the provided context passages. Always cite the source papers by name.
If context is partially relevant, extract what is useful. Be precise and academic.
"""


class _RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm
        self._lock     = threading.Lock()
        self._last     = 0.0

    def wait(self) -> None:
        with self._lock:
            gap = self._interval - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


class Generator:
    def __init__(self) -> None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self._client  = genai.Client(api_key=config.GEMINI_API_KEY)
        self._limiter = _RateLimiter(config.GEMINI_RPM)
        self._cfg     = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.GEMINI_TEMPERATURE,
            max_output_tokens=config.GEMINI_MAX_TOKENS,
        )
        log.info("Gemini ready: %s", config.GEMINI_MODEL)

    def generate(self, query: str, context: str, retries: int = 3) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        for attempt in range(1, retries + 1):
            self._limiter.wait()
            try:
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=self._cfg,
                )
                return response.text.strip()
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "quota" in err:
                    wait = 30 * attempt
                    log.warning("Rate limit — waiting %ds", wait)
                    time.sleep(wait)
                else:
                    log.error("Gemini error: %s", exc)
                    return f"[Generation error: {exc}]"
        return "[Generation error: max retries exceeded]"

    def generate_raw(self, prompt: str, system: str | None = None, retries: int = 3) -> str:
        """Generate with a custom system prompt (used by lit review synthesiser)."""
        cfg = types.GenerateContentConfig(
            system_instruction=system or SYSTEM_PROMPT,
            temperature=config.GEMINI_TEMPERATURE,
            max_output_tokens=config.GEMINI_MAX_TOKENS,
        )
        for attempt in range(1, retries + 1):
            self._limiter.wait()
            try:
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=cfg,
                )
                return response.text.strip()
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "quota" in err:
                    time.sleep(30 * attempt)
                else:
                    return f"[Error: {exc}]"
        return "[Error: max retries]"
