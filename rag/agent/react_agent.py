"""
react_agent.py – ReAct (Reason + Act) agent for arXiv ML research Q&A.

Loop per query:
  1. Planner  — decompose complex questions into sub-questions
  2. ReAct    — iterative Thought → Action → Observation until confident
  3. Critic   — self-critique; if gaps found, run one corrective ReAct pass
  4. Return   — final answer + full scratchpad (for UI transparency)

The agent uses Gemini as its reasoning backbone. Each step is one API call
that returns a JSON {thought, action, action_input} object.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from rag import config
from rag.agent.tools import Tool, build_tool_registry, format_tool_descriptions
from rag.agent.memory import ConversationMemory, ResearchMemory
from rag.agent.planner import QueryPlanner
from rag.agent.critic import AnswerCritic, CritiqueResult

log = logging.getLogger(__name__)


# ── Scratchpad entry ───────────────────────────────────────────────────────────

@dataclass
class Step:
    thought: str
    action: str
    action_input: str
    observation: str = ""
    is_final: bool = False


# ── Agent response ─────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    answer: str
    scratchpad: list[Step]
    sources: list[dict]             # [{title, score}]
    critique: CritiqueResult | None = None
    sub_questions: list[str] = field(default_factory=list)
    total_steps: int = 0
    latency_ms: float = 0.0


# ── System prompt ──────────────────────────────────────────────────────────────

_REACT_SYSTEM = """\
You are a research agent with access to 150 recent arXiv ML papers (cs.LG, early 2026).
You reason step-by-step using the ReAct framework: Thought → Action → Observation.

Available tools:
{tool_descriptions}

{conversation_context}
{research_notes}

At each step respond with ONLY valid JSON in exactly this format:
{{
  "thought": "your reasoning about what to do next",
  "action": "tool_name",
  "action_input": "input to the tool"
}}

To give the final answer:
{{
  "thought": "I now have enough information to give a complete, well-cited answer.",
  "action": "finish",
  "action_input": "your full answer, citing papers as [Source: Paper Title]"
}}

Rules:
- Always retrieve before answering — never answer from prior knowledge alone.
- Cite specific paper titles in your final answer.
- For broad questions, search 2-3 different angles.
- If a search returns nothing relevant, try different keywords.
- You have at most {max_steps} steps before you must call finish.
- Current sub-question being answered: {current_question}
"""


# ── ReAct agent ────────────────────────────────────────────────────────────────

class ReActAgent:
    """
    The main agent. Instantiate once after retrievers are loaded,
    then call .run(query) for each user question.
    """

    def __init__(
        self,
        retriever,
        bm25_retriever,
        conv_memory: ConversationMemory | None = None,
        research_memory: ResearchMemory | None = None,
    ) -> None:
        self._tools     = build_tool_registry(retriever, bm25_retriever)
        self._memory    = conv_memory    or ConversationMemory()
        self._research  = research_memory or ResearchMemory()
        self._planner   = QueryPlanner()
        self._critic    = AnswerCritic()
        self._client    = genai.Client(api_key=config.GEMINI_API_KEY)
        self._retriever = retriever      # kept for source extraction

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, query: str) -> AgentResponse:
        t0 = time.monotonic()
        self._memory.add_user(query)

        # 1. Plan
        sub_questions = self._planner.plan(query)
        is_complex    = len(sub_questions) > 1

        # 2. Gather context across all sub-questions
        all_steps: list[Step] = []
        all_context: list[str] = []

        for sub_q in sub_questions:
            steps, context = self._react_loop(
                question=sub_q,
                original_query=query,
            )
            all_steps.extend(steps)
            if context:
                all_context.append(f"[Sub-question: {sub_q}]\n{context}")

        combined_context = "\n\n".join(all_context)

        # 3. For complex queries, synthesise all sub-answers into one final answer
        if is_complex and all_context:
            final_answer = self._synthesise(query, combined_context)
        else:
            # Simple query: the last "finish" step has the answer
            finish_steps = [s for s in all_steps if s.is_final]
            final_answer = (
                finish_steps[-1].action_input if finish_steps
                else "I could not find relevant information in the corpus."
            )

        # 4. Critic
        critique = self._critic.evaluate(query, final_answer, combined_context)

        # 5. If critic says retry, run one corrective pass
        if not critique.passed and critique.search_hints:
            log.info("Critic requested retry with hints: %s", critique.search_hints)
            for hint in critique.search_hints[:2]:
                extra_steps, extra_ctx = self._react_loop(
                    question=hint,
                    original_query=query,
                    max_steps=3,
                )
                all_steps.extend(extra_steps)
                if extra_ctx:
                    combined_context += f"\n\n[Corrective search: {hint}]\n{extra_ctx}"

            final_answer = self._synthesise(query, combined_context)

        # 6. Update memory
        self._memory.add_assistant(final_answer)

        # 7. Extract sources from scratchpad observations
        sources = self._extract_sources(all_steps)

        latency = round((time.monotonic() - t0) * 1000, 1)
        return AgentResponse(
            answer        = final_answer,
            scratchpad    = all_steps,
            sources       = sources,
            critique      = critique,
            sub_questions = sub_questions if is_complex else [],
            total_steps   = len(all_steps),
            latency_ms    = latency,
        )

    # ── ReAct loop ────────────────────────────────────────────────────────────

    def _react_loop(
        self,
        question: str,
        original_query: str,
        max_steps: int | None = None,
    ) -> tuple[list[Step], str]:
        """
        Run the ReAct loop for one (sub-)question.
        Returns (steps, accumulated_observation_context).
        """
        max_steps = max_steps or config.AGENT_MAX_STEPS
        scratchpad: list[Step] = []
        observations: list[str] = []

        for step_num in range(max_steps):
            # Build the prompt with current scratchpad
            system_prompt = self._build_system_prompt(question)
            user_prompt   = self._build_user_prompt(
                original_query, question, scratchpad
            )

            # Ask the model what to do next
            raw_json = self._call_model(system_prompt, user_prompt)
            parsed   = self._parse_step(raw_json)

            if parsed is None:
                log.warning("Step %d: could not parse model output — stopping", step_num)
                break

            thought, action, action_input = (
                parsed["thought"],
                parsed["action"],
                parsed["action_input"],
            )

            # ── finish ────────────────────────────────────────────────────────
            if action == "finish":
                step = Step(
                    thought      = thought,
                    action       = action,
                    action_input = action_input,
                    observation  = "",
                    is_final     = True,
                )
                scratchpad.append(step)
                observations.append(action_input)
                log.info("Step %d: FINISH", step_num + 1)
                break

            # ── tool call ─────────────────────────────────────────────────────
            tool = self._tools.get(action)
            if tool is None:
                observation = f"[Unknown tool: {action}. Available: {list(self._tools)}]"
            else:
                log.info("Step %d: %s(%r)", step_num + 1, action, action_input[:60])
                try:
                    observation = tool.fn(action_input)
                except Exception as exc:
                    observation = f"[Tool error: {exc}]"

            step = Step(
                thought      = thought,
                action       = action,
                action_input = action_input,
                observation  = observation,
            )
            scratchpad.append(step)
            observations.append(observation)

            # Brief pause to respect rate limits
            time.sleep(0.3)

        return scratchpad, "\n\n".join(observations)

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesise(self, original_query: str, combined_context: str) -> str:
        """
        Final synthesis pass: given all gathered context, produce one cohesive answer.
        Used for complex multi-hop queries.
        """
        prompt = (
            f"Based on the following research context, answer the question comprehensively.\n\n"
            f"Question: {original_query}\n\n"
            f"Context:\n{combined_context[:3000]}\n\n"
            f"Write a detailed, well-structured answer. "
            f"Cite specific papers as [Source: Paper Title]. "
            f"Cover all aspects of the question."
        )
        try:
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=config.GEMINI_MAX_TOKENS,
                ),
            )
            return response.text.strip()
        except Exception as exc:
            log.error("Synthesis failed: %s", exc)
            return combined_context[:1000]   # fallback

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_system_prompt(self, current_question: str) -> str:
        conv_ctx = self._memory.format_for_prompt()
        notes    = self._research.format_for_prompt()
        return _REACT_SYSTEM.format(
            tool_descriptions    = format_tool_descriptions(self._tools),
            conversation_context = (
                f"\n**Recent conversation:**\n{conv_ctx}" if conv_ctx else ""
            ),
            research_notes       = (f"\n{notes}" if notes else ""),
            max_steps            = config.AGENT_MAX_STEPS,
            current_question     = current_question,
        )

    def _build_user_prompt(
        self,
        original_query: str,
        current_question: str,
        scratchpad: list[Step],
    ) -> str:
        lines = [f"Original question: {original_query}"]
        if current_question != original_query:
            lines.append(f"Current sub-question: {current_question}")

        if scratchpad:
            lines.append("\nScratchpad so far:")
            for i, step in enumerate(scratchpad, 1):
                lines.append(f"\nStep {i}:")
                lines.append(f"  Thought: {step.thought}")
                lines.append(f"  Action: {step.action}")
                lines.append(f"  Action Input: {step.action_input}")
                if step.observation:
                    obs_preview = step.observation[:600]
                    lines.append(f"  Observation: {obs_preview}{'…' if len(step.observation) > 600 else ''}")

        lines.append("\nWhat is your next step? Respond with JSON only.")
        return "\n".join(lines)

    # ── Model call ────────────────────────────────────────────────────────────

    def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    max_output_tokens=800,
                ),
            )
            return response.text.strip()
        except Exception as exc:
            log.error("Model call failed: %s", exc)
            return '{"thought": "Error occurred", "action": "finish", "action_input": "[Agent error]"}'

    # ── JSON parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_step(raw: str) -> dict | None:
        try:
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if all(k in parsed for k in ("thought", "action", "action_input")):
                return parsed
        except json.JSONDecodeError:
            # Try to extract JSON block from mixed output
            match = re.search(r'\{[^{}]*"thought"[^{}]*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        log.warning("Could not parse step JSON from: %s", raw[:200])
        return None

    # ── Source extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_sources(steps: list[Step]) -> list[dict]:
        """
        Parse paper titles and scores from tool observations.
        Returns deduplicated [{title, score}] sorted by score.
        """
        import re as _re
        seen: dict[str, float] = {}

        for step in steps:
            if not step.observation:
                continue
            # Match lines like: [1] **Title** (score: 0.874)
            for match in _re.finditer(
                r'\*\*(.+?)\*\*\s*\((?:score|bm25_score):\s*([\d.]+)\)',
                step.observation,
            ):
                title = match.group(1).strip()
                score = float(match.group(2))
                if title not in seen or score > seen[title]:
                    seen[title] = score

        return [
            {"title": t, "score": round(s, 3)}
            for t, s in sorted(seen.items(), key=lambda x: -x[1])
        ]
