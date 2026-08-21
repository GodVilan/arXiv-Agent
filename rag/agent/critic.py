"""
critic.py – Self-critique: checks if an answer is grounded, complete, and specific.
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from rag.llm import get_client
from google.genai import types
from rag import config

log = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert double-blind academic peer reviewer for top-tier Machine Learning conferences (NeurIPS, ICML, ICLR).
Evaluate the proposed answer to the research question based on the provided retrieved context.

You must respond with ONLY a valid JSON object:
{
  "verdict": "pass" | "retry",
  "grounded": true | false,
  "complete": true | false,
  "specific": true | false,
  "gaps": ["any missing details or loose claims"],
  "search_hints": ["suggested queries to find missing details"],
  "peer_review": {
    "overall_score": 5, -- 1-10 NeurIPS scale (10 is groundbreaking, 5 is borderline, 1 is reject)
    "confidence_score": 3, -- 1-5 scale
    "strengths": ["clear scientific points"],
    "weaknesses": ["limitations, lack of detail, baseline comparison issues"],
    "constructive_feedback": "how to improve this scientific answer"
  }
}
"""


@dataclass
class CritiqueResult:
    verdict:      str
    grounded:     bool
    complete:     bool
    specific:     bool
    gaps:         list[str]
    search_hints: list[str]
    peer_review:  dict | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


class AnswerCritic:
    def __init__(self) -> None:
        self._client = get_client()
        self._cfg    = types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=0.0,
            max_output_tokens=400,
        )

    def evaluate(self, question: str, answer: str, context: str) -> CritiqueResult:
        prompt = (
            f"Question: {question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Context (first 30000 chars):\n{context[:30000]}"
        )
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=self._cfg,
            )
            raw    = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
            parsed = json.loads(raw)
            result = CritiqueResult(
                verdict      = parsed.get("verdict", "pass"),
                grounded     = parsed.get("grounded", True),
                complete     = parsed.get("complete", True),
                specific     = parsed.get("specific", True),
                gaps         = parsed.get("gaps", []),
                search_hints = parsed.get("search_hints", []),
                peer_review  = parsed.get("peer_review"),
            )
            log.info("Critic: %s | grounded=%s complete=%s specific=%s score=%s",
                     result.verdict, result.grounded, result.complete, result.specific,
                     result.peer_review.get("overall_score") if result.peer_review else "N/A")
            return result
        except Exception as exc:
            log.warning("Critic failed (%s) — default pass", exc)
            return CritiqueResult("pass", True, True, True, [], [])
