"""
planner.py – Query decomposition for the arXiv agent.

Simple queries  →  passed straight to the ReAct agent.
Complex queries →  decomposed into sub-questions, each retrieved independently,
                   then synthesised into a single coherent answer.

Decomposition is triggered by an LLM call so the decision is semantic,
not rule-based. The cost is one extra Gemini call (~200 tokens input/output)
on questions that actually need it.
"""

from __future__ import annotations

import json
import logging
import re

from google import genai
from google.genai import types

from rag import config

log = logging.getLogger(__name__)


# ── Prompt ─────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a research query analyst. Given a user question, decide whether it is:
  (A) Simple — can be answered with a single search query
  (B) Complex — requires multiple searches to gather enough information

For complex questions, decompose into 2-4 focused sub-questions.

Respond with ONLY valid JSON in one of these two formats:

Simple:
{"type": "simple", "sub_questions": []}

Complex:
{"type": "complex", "sub_questions": ["sub-question 1", "sub-question 2", ...]}

Rules:
- Comparison questions ("compare X and Y", "X vs Y") → always complex
- Questions with "and" joining two different topics → usually complex
- Literature review / survey requests → complex
- Single-topic factual questions → simple
- Never exceed 4 sub-questions
"""


# ── Planner ────────────────────────────────────────────────────────────────────

class QueryPlanner:
    """
    Decides whether a query needs decomposition and returns sub-questions.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._cfg    = types.GenerateContentConfig(
            system_instruction=_PLANNER_SYSTEM,
            temperature=0.0,
            max_output_tokens=300,
        )

    def plan(self, query: str) -> list[str]:
        """
        Returns a list of search queries:
          - [query] if simple (no decomposition)
          - [sub_q1, sub_q2, ...] if complex
        """
        try:
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=f"User question: {query}",
                config=self._cfg,
            )
            raw = response.text.strip()

            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)

            if parsed.get("type") == "complex":
                sub_questions = parsed.get("sub_questions", [])
                if sub_questions:
                    log.info(
                        "Planner: complex query → %d sub-questions", len(sub_questions)
                    )
                    return sub_questions[: config.AGENT_MAX_SUBQUESTIONS]

            log.info("Planner: simple query — no decomposition")
            return [query]

        except (json.JSONDecodeError, Exception) as exc:
            log.warning("Planner failed (%s) — treating as simple query", exc)
            return [query]

    def is_complex(self, query: str) -> bool:
        """Quick check without full decomposition (uses heuristics)."""
        indicators = [
            "compare", "contrast", "difference between", "versus", " vs ",
            "relationship between", "how do .* and .* differ",
            "literature review", "survey", "overview of recent",
        ]
        q_lower = query.lower()
        return any(re.search(pat, q_lower) for pat in indicators)
