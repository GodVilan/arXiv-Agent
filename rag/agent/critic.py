"""
critic.py – Self-critique and reflection for the arXiv agent.

After the ReAct loop produces an answer, the Critic evaluates it on three axes:
  1. Grounded     — is every claim traceable to a retrieved passage?
  2. Complete     — does it address all parts of the question?
  3. Specific     — does it name actual papers/methods rather than speaking vaguely?

If the answer fails any check, the Critic returns:
  - a verdict (pass | retry)
  - a list of specific gaps to fill
  - an improved scratchpad hint for the retry

This is the "reflection" step that turns a single-pass RAG system into a
self-improving agent. One extra Gemini call; only triggered when needed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

from rag import config

log = logging.getLogger(__name__)


# ── Prompt ─────────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a rigorous research assistant evaluator.

Given:
  - The user's original question
  - The agent's proposed answer
  - The context passages that were retrieved

Evaluate the answer on three criteria:
  1. grounded   — every factual claim is supported by the retrieved context
  2. complete   — all parts of the question are addressed
  3. specific   — actual paper titles / method names are cited, not vague statements

Respond ONLY with valid JSON:
{
  "verdict": "pass" | "retry",
  "grounded": true | false,
  "complete": true | false,
  "specific": true | false,
  "gaps": ["gap 1", "gap 2"],           // empty list if verdict is pass
  "search_hints": ["search hint 1"]     // additional queries to fix gaps
}

Be strict. If any important part of the question is unanswered, return "retry".
"""


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class CritiqueResult:
    verdict: str            # "pass" | "retry"
    grounded: bool
    complete: bool
    specific: bool
    gaps: list[str]
    search_hints: list[str]

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


# ── Critic ─────────────────────────────────────────────────────────────────────

class AnswerCritic:
    """
    Reviews a proposed answer and returns a CritiqueResult.

    Usage:
        critique = critic.evaluate(question, answer, context)
        if not critique.passed:
            # use critique.search_hints to gather more context
            ...
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._cfg    = types.GenerateContentConfig(
            system_instruction=_CRITIC_SYSTEM,
            temperature=0.0,
            max_output_tokens=400,
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> CritiqueResult:
        """
        Evaluate the answer and return a CritiqueResult.
        Falls back to a passing result on any API error to avoid blocking the agent.
        """
        prompt = (
            f"Question: {question}\n\n"
            f"Proposed answer:\n{answer}\n\n"
            f"Retrieved context (first 1500 chars):\n{context[:1500]}"
        )

        try:
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=self._cfg,
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)
            result = CritiqueResult(
                verdict      = parsed.get("verdict", "pass"),
                grounded     = parsed.get("grounded", True),
                complete     = parsed.get("complete", True),
                specific     = parsed.get("specific", True),
                gaps         = parsed.get("gaps", []),
                search_hints = parsed.get("search_hints", []),
            )
            log.info(
                "Critic verdict: %s | grounded=%s complete=%s specific=%s",
                result.verdict, result.grounded, result.complete, result.specific,
            )
            return result

        except Exception as exc:
            log.warning("Critic failed (%s) — defaulting to pass", exc)
            return CritiqueResult(
                verdict="pass",
                grounded=True,
                complete=True,
                specific=True,
                gaps=[],
                search_hints=[],
            )

    def format_feedback(self, critique: CritiqueResult) -> str:
        """Human-readable critique summary for display in the UI."""
        if critique.passed:
            return "✅ Answer passed self-critique review."

        issues = []
        if not critique.grounded:
            issues.append("⚠ Some claims may not be grounded in retrieved context")
        if not critique.complete:
            issues.append("⚠ Question not fully answered")
        if not critique.specific:
            issues.append("⚠ Answer is too vague — missing specific paper citations")

        lines = ["🔄 Critique flagged issues — refining answer:"] + issues
        if critique.gaps:
            lines.append("Gaps: " + "; ".join(critique.gaps))
        return "\n".join(lines)
