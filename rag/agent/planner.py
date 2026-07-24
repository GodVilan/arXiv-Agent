"""
planner.py – Query decomposition for complex multi-hop questions.
"""
from __future__ import annotations
import json
import logging
import re
from google import genai
from google.genai import types
from rag import config

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a research query analyst. Decide if a question needs multiple searches or just one.

Respond ONLY with valid JSON — no preamble, no markdown fences:

Simple (one search): {"type": "simple", "sub_questions": []}
Complex (2-4 searches): {"type": "complex", "sub_questions": ["q1", "q2", ...]}

Rules:
- "compare X and Y" / "X vs Y" → complex
- Literature review / survey → complex
- Two distinct topics joined by "and" → complex
- Single topic factual question → simple
- Max 4 sub-questions
"""


class QueryPlanner:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._cfg    = types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=0.0,
            max_output_tokens=256,
        )

    def plan(self, query: str) -> list[str]:
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=f"Question: {query}",
                config=self._cfg,
            )
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
            parsed = json.loads(raw)
            if parsed.get("type") == "complex":
                subs = parsed.get("sub_questions", [])
                if subs:
                    log.info("Planner: %d sub-questions", len(subs))
                    return subs[:config.AGENT_MAX_SUBQUESTIONS]
        except Exception as exc:
            log.warning("Planner failed (%s) — simple fallback", exc)
        return [query]
