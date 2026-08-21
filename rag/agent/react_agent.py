"""
react_agent.py – ReAct loop with two key fixes:
  1. Scope guard  — redirects non-ML questions immediately
  2. Loop guard   — detects repeated searches and forces progress
"""
from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from google.genai import types
from rag import config
from rag.llm import get_client
from rag.sources.source_router import SourceConfig
from rag.agent.tools import build_tool_registry, format_tool_descriptions
from rag.agent.memory import ConversationMemory, ResearchMemory
from rag.agent.planner import QueryPlanner
from rag.agent.critic import AnswerCritic, CritiqueResult

log = logging.getLogger(__name__)

# ── Scope check ────────────────────────────────────────────────────────────────

_SCOPE_SYSTEM = """\
You are a scope classifier for an ML research assistant.
Decide if a question is related to machine learning, AI, deep learning, data science,
or academic research papers.

Respond with ONLY one word: YES or NO.

YES examples:
- "What is LoRA?"
- "Explain transformers"
- "Compare RLHF approaches"
- "What papers exist on continual learning?"

NO examples:
- "What is 1+1?"
- "Who is the president?"
- "What's the weather?"
- "Write me a poem"
"""

_OUT_OF_SCOPE_REPLY = (
    "I'm specialised in machine learning and AI research. "
    "I can answer questions about ML methods, models, papers, and techniques — "
    "for example: *'What is LoRA?'*, *'Compare RLHF approaches'*, or "
    "*'Generate a literature review on continual learning'*. "
    "What would you like to know about ML research?"
)

# ── ReAct system prompt ────────────────────────────────────────────────────────

_REACT_SYSTEM = """\
You are a research agent with access to 150 recent ML papers from arXiv and user-uploaded documents.
You reason step-by-step: Thought → Action → Observation.

Available tools:
{tool_descriptions}

{conversation_context}
{research_notes}

{loop_hint}

Respond ONLY with valid JSON at each step:
{{
  "thought": "your reasoning",
  "action": "tool_name",
  "action_input": "input to the tool"
}}

To give the final answer:
{{
  "thought": "I now have enough information.",
  "action": "finish",
  "action_input": "your complete answer citing papers as [Source: Title]"
}}

Rules:
- Always retrieve before answering — never answer from prior knowledge alone.
- Cite specific paper titles in your final answer.
- NEVER repeat the same search query you already tried.
- If two searches return the same top result, switch tools (use fetch_arxiv or keyword_search).
- If after 4 searches you still lack relevant results, use fetch_arxiv to get fresh papers.
- You have at most {max_steps} steps. Current question: {current_question}
"""


@dataclass
class Step:
    thought:      str
    action:       str
    action_input: str
    observation:  str  = ""
    is_final:     bool = False


@dataclass
class AgentResponse:
    answer:        str
    scratchpad:    list[Step]
    sources:       list[dict]
    critique:      CritiqueResult | None = None
    sub_questions: list[str] = field(default_factory=list)
    total_steps:   int   = 0
    latency_ms:    float = 0.0
    out_of_scope:  bool  = False


@dataclass
class GatherResult:
    """Evidence gathered by the ReAct loop before the final answer is written.

    Separating gathering (slow, multi-step) from answer synthesis (streamable) lets
    the UI show reasoning progress, then stream the answer token-by-token.
    """
    combined:      str
    steps:         list[Step]
    sub_questions: list[str]
    is_complex:    bool
    started_at:    float
    out_of_scope:  bool = False
    oos_reply:     str  = ""


class ReActAgent:
    def __init__(self, router, bm25, session_index,
                 conv_memory: ConversationMemory | None = None,
                 research_memory: ResearchMemory | None = None) -> None:
        self._router   = router
        self._allowed_paper_ids = None
        self._source_config = None
        self._use_arxiv = True

        self._tools    = build_tool_registry(
            router, bm25, session_index,
            get_allowed_paper_ids=lambda: self._allowed_paper_ids,
            get_source_config=lambda: self._source_config,
            get_use_arxiv=lambda: self._use_arxiv
        )
        self._memory   = conv_memory    or ConversationMemory()
        self._research = research_memory or ResearchMemory()
        self._planner  = QueryPlanner()
        self._critic   = AnswerCritic()
        self._client   = get_client()
        # Populated by stream_synthesis() once the streamed answer completes.
        self.last_response: AgentResponse | None = None

    # ── Public entry point ─────────────────────────────────────────────────────

    def gather(
        self, query: str,
        allowed_paper_ids: set[str] | None = None,
        source_cfg: SourceConfig | None = None,
        use_arxiv: bool = True,
        custom_instructions: str | None = None,
        step_callback: Callable | None = None
    ) -> GatherResult:
        """Phase 1: scope-check, plan, and run the ReAct loop to gather evidence.

        Returns the combined context and reasoning steps without writing the final
        answer, so the caller can then either synthesise (run) or stream it.
        """
        self._allowed_paper_ids = allowed_paper_ids
        self._source_config = source_cfg
        self._use_arxiv = use_arxiv
        self._custom_instructions = custom_instructions

        t0 = time.monotonic()
        self._memory.add_user(query)

        # 1. Scope check — fast, before anything else
        if not self._is_in_scope(query):
            return GatherResult(
                combined="", steps=[], sub_questions=[], is_complex=False,
                started_at=t0, out_of_scope=True,
                oos_reply=self._generate_out_of_scope_reply(query),
            )

        # 2. Plan
        sub_questions = self._planner.plan(query)
        is_complex    = len(sub_questions) > 1

        all_steps:   list[Step] = []
        all_context: list[str]  = []

        for sub_q in sub_questions:
            steps, context = self._react_loop(sub_q, query, step_callback=step_callback)
            all_steps.extend(steps)
            if context:
                all_context.append(
                    f"[Sub-question: {sub_q}]\n{context}" if is_complex else context
                )

        return GatherResult(
            combined="\n\n".join(all_context), steps=all_steps,
            sub_questions=sub_questions, is_complex=is_complex, started_at=t0,
        )

    def run(
        self, query: str,
        allowed_paper_ids: set[str] | None = None,
        source_cfg: SourceConfig | None = None,
        use_arxiv: bool = True,
        custom_instructions: str | None = None,
        step_callback: Callable | None = None
    ) -> AgentResponse:
        g = self.gather(query, allowed_paper_ids, source_cfg, use_arxiv,
                        custom_instructions, step_callback)

        # 1. Out of scope — short-circuit
        if g.out_of_scope:
            self._memory.add_assistant(g.oos_reply)
            return AgentResponse(
                answer       = g.oos_reply,
                scratchpad   = [],
                sources      = [],
                out_of_scope = True,
                latency_ms   = round((time.monotonic() - g.started_at) * 1000, 1),
            )

        all_steps    = g.steps
        combined     = g.combined
        sub_questions = g.sub_questions
        is_complex   = g.is_complex

        # 3. Synthesise
        if is_complex and combined:
            final_answer = self._synthesise(query, combined)
        else:
            finish_steps = [s for s in all_steps if s.is_final]
            if finish_steps:
                final_answer = finish_steps[-1].action_input
            elif combined.strip():
                # If ReAct loop timed out/exceeded max steps but gathered context, synthesise anyway!
                final_answer = self._synthesise(query, combined)
            else:
                final_answer = "I could not find relevant information in the corpus for this query."

        # 4. Critic
        critique = self._critic.evaluate(query, final_answer, combined)

        if not critique.passed and critique.search_hints:
            for hint in critique.search_hints[:2]:
                extra_steps, extra_ctx = self._react_loop(hint, query, max_steps=3, step_callback=step_callback)
                all_steps.extend(extra_steps)
                if extra_ctx:
                    combined += f"\n\n[Corrective: {hint}]\n{extra_ctx}"
            final_answer = self._synthesise(query, combined)

        # 5. Format & wrap citations with paper_ids in HTML spans
        papers_lookup = self._build_papers_lookup()
        final_answer = self._wrap_citations(final_answer, papers_lookup)

        self._memory.add_assistant(final_answer)
        sources = self._extract_sources(all_steps)
        latency = round((time.monotonic() - g.started_at) * 1000, 1)

        return AgentResponse(
            answer        = final_answer,
            scratchpad    = all_steps,
            sources       = sources,
            critique      = critique,
            sub_questions = sub_questions if is_complex else [],
            total_steps   = len(all_steps),
            latency_ms    = latency,
        )

    # ── Streaming answer (Phase 2) ─────────────────────────────────────────────

    def stream_synthesis(self, query: str, gathered: GatherResult):
        """Phase 2 generator: yield the final answer token-by-token for st.write_stream.

        The complete, citation-wrapped AgentResponse (with steps, sources, critique)
        is stored on self.last_response once the stream is exhausted.

        Note: unlike run(), the streaming path does not perform the critic's corrective
        re-search loop — the answer is already being shown as it is written. The critique
        verdict is still computed and surfaced.
        """
        if gathered.out_of_scope:
            self._memory.add_assistant(gathered.oos_reply)
            self.last_response = AgentResponse(
                answer=gathered.oos_reply, scratchpad=[], sources=[], out_of_scope=True,
                latency_ms=round((time.monotonic() - gathered.started_at) * 1000, 1),
            )
            yield gathered.oos_reply
            return

        if not gathered.combined.strip():
            fallback = "I could not find relevant information in the corpus for this query."
            self._memory.add_assistant(fallback)
            self.last_response = AgentResponse(
                answer=fallback, scratchpad=gathered.steps,
                sources=self._extract_sources(gathered.steps),
                sub_questions=gathered.sub_questions if gathered.is_complex else [],
                total_steps=len(gathered.steps),
                latency_ms=round((time.monotonic() - gathered.started_at) * 1000, 1),
            )
            yield fallback
            return

        buffer: list[str] = []
        for chunk in self._synthesise_stream(query, gathered.combined):
            buffer.append(chunk)
            yield chunk

        raw_answer = "".join(buffer).strip() or \
            "I could not find relevant information in the corpus for this query."

        critique = self._critic.evaluate(query, raw_answer, gathered.combined)
        final_answer = self._wrap_citations(raw_answer, self._build_papers_lookup())

        self._memory.add_assistant(final_answer)
        self.last_response = AgentResponse(
            answer        = final_answer,
            scratchpad    = gathered.steps,
            sources       = self._extract_sources(gathered.steps),
            critique      = critique,
            sub_questions = gathered.sub_questions if gathered.is_complex else [],
            total_steps   = len(gathered.steps),
            latency_ms    = round((time.monotonic() - gathered.started_at) * 1000, 1),
        )

    # ── Scope check ────────────────────────────────────────────────────────────

    # Terms that make a query unambiguously in-scope. Presence of any of these skips
    # the LLM scope classifier entirely, removing one Gemini round-trip from the hot path.
    _IN_SCOPE_TERMS = frozenset({
        "machine learning", "deep learning", "neural network", "transformer",
        "llm", "language model", "embedding", "fine-tun", "pretrain", "rlhf",
        "lora", "diffusion", "gradient", "attention", "arxiv", "paper", "dataset",
        "benchmark", "reinforcement learning", "supervised", "unsupervised",
        "convolution", "gan", "bert", "gpt", "backprop", "overfit", "regulariz",
        "hyperparameter", "quantization", "distillation", "retrieval", "rag",
        "catastrophic forgetting", "continual learning", "few-shot", "zero-shot",
    })

    def _is_in_scope(self, query: str) -> bool:
        """Is this an ML/AI research question? Keyword fast-path, then LLM fallback."""
        q = query.lower()
        if any(term in q for term in self._IN_SCOPE_TERMS):
            return True
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=f"Question: {query}",
                config=types.GenerateContentConfig(
                    system_instruction=_SCOPE_SYSTEM,
                    temperature=0.0,
                    max_output_tokens=5,
                ),
            )
            answer = resp.text.strip().upper()
            in_scope = answer.startswith("Y")
            if not in_scope:
                log.info("Out of scope: %s", query)
            return in_scope
        except Exception as exc:
            log.warning("Scope check failed (%s) — defaulting to in-scope", exc)
            return True   # fail open

    def _generate_out_of_scope_reply(self, query: str) -> str:
        """
        Dynamically generates a helpful, polite out-of-scope response
        suggesting 2-3 machine learning research topics or questions
        related to what the user asked.
        """
        system_instruction = (
            "You are a helpful and polite ML research assistant. The user has asked an out-of-scope or non-ML question.\n"
            "Respond politely explaining that you are specialized in machine learning and AI research.\n"
            "Then, suggest 2-3 interesting, related machine learning or AI research questions or concepts "
            "that the user might want to explore instead, showing how they can connect their interest to ML.\n"
            "Format the suggested questions in bullet points with emoji icons.\n"
            "Keep the reply concise, polite, and under 150 words."
        )
        prompt = f"User's question: '{query}'"
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=300,
                ),
            )
            return resp.text.strip()
        except Exception as e:
            log.warning("Dynamic scope reply failed: %s", e)
            return _OUT_OF_SCOPE_REPLY  # Fallback to static reply

    # ── ReAct loop ─────────────────────────────────────────────────────────────

    def _react_loop(self, question: str, original_query: str,
                    max_steps: int | None = None,
                    step_callback: Callable | None = None) -> tuple[list[Step], str]:
        max_steps     = max_steps or config.AGENT_MAX_STEPS
        scratchpad:   list[Step] = []
        observations: list[str]  = []

        # Loop-guard state
        used_queries:    list[str] = []   # normalised queries already tried
        top_result_hits: dict[str, int] = {}  # top result title → times seen
        consecutive_same = 0

        for step_num in range(max_steps):
            loop_hint = self._build_loop_hint(used_queries, top_result_hits,
                                              consecutive_same, step_num)
            system = self._build_system(question, loop_hint)
            user   = self._build_user(original_query, question, scratchpad)
            raw    = self._call_model(system, user)
            parsed = self._parse(raw)

            if parsed is None:
                break

            thought, action, action_input = (
                parsed["thought"], parsed["action"], parsed["action_input"]
            )

            # Trigger step callback for real-time progress stream
            if step_callback:
                try:
                    step_callback(step_num, thought, action, action_input)
                except Exception as cb_err:
                    log.warning("Step callback failed: %s", cb_err)

            # ── finish ────────────────────────────────────────────────────────
            if action == "finish":
                scratchpad.append(Step(thought, action, action_input, "", True))
                observations.append(action_input)
                break

            # ── loop guard: block duplicate search queries ─────────────────
            norm_query = self._normalise_query(action_input)
            if action in ("search_corpus", "keyword_search") and norm_query in used_queries:
                log.info("Loop guard: blocking duplicate query '%s'", action_input[:50])
                # Inject a forced observation telling the agent to change approach
                observation = (
                    f"[LOOP GUARD] You already searched for '{action_input}'. "
                    "Try a different query, use fetch_arxiv for fresh papers, "
                    "or call finish if you have enough context."
                )
                step = Step(thought, action, action_input, observation)
                scratchpad.append(step)
                observations.append(observation)
                consecutive_same += 1
                # Force finish if stuck badly
                if consecutive_same >= 3:
                    log.info("Loop guard: forcing finish after 3 blocked queries")
                    break
                continue

            # ── execute tool ───────────────────────────────────────────────
            tool = self._tools.get(action)
            if tool is None:
                observation = f"[Unknown tool: {action}. Available: {list(self._tools)}]"
            else:
                log.info("Step %d: %s(%r…)", step_num + 1, action, action_input[:50])
                try:
                    observation = tool.fn(action_input)
                except Exception as exc:
                    observation = f"[Tool error: {exc}]"

            # Track for loop guard
            if action in ("search_corpus", "keyword_search"):
                used_queries.append(norm_query)
                top_title = self._extract_top_result_title(observation)
                if top_title:
                    top_result_hits[top_title] = top_result_hits.get(top_title, 0) + 1
                    if top_result_hits[top_title] > 1:
                        consecutive_same += 1
                    else:
                        consecutive_same = 0
                else:
                    consecutive_same = 0

            step = Step(thought, action, action_input, observation)
            scratchpad.append(step)
            observations.append(observation)

        return scratchpad, "\n\n".join(observations)

    # ── Loop guard helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _normalise_query(q: str) -> str:
        """Lowercase, strip punctuation, sort words — catches near-duplicates."""
        words = sorted(re.sub(r"[^\w\s]", "", q.lower()).split())
        return " ".join(words)

    @staticmethod
    def _extract_top_result_title(observation: str) -> str | None:
        """Pull the title of the first result from a tool observation."""
        m = re.search(r'\[1\] \*\*(.+?)\*\*', observation)
        return m.group(1).strip() if m else None

    @staticmethod
    def _build_loop_hint(used_queries: list[str], top_hits: dict[str, int],
                         consecutive_same: int, step_num: int) -> str:
        hints = []

        if used_queries:
            hints.append(
                f"Queries already tried (DO NOT repeat): "
                + ", ".join(f'"{q}"' for q in used_queries[-4:])
            )

        repeated = [t for t, c in top_hits.items() if c >= 2]
        if repeated:
            hints.append(
                f"These results keep appearing — the corpus may not cover this well. "
                f"Try fetch_arxiv or a completely different angle: "
                + ", ".join(f'"{t[:40]}"' for t in repeated[:2])
            )

        if consecutive_same >= 2:
            hints.append(
                "⚠ You are looping. Switch to fetch_arxiv to get fresh papers, "
                "or call finish with what you have."
            )

        if step_num >= config.AGENT_MAX_STEPS - 2:
            hints.append(
                f"⚠ Only {config.AGENT_MAX_STEPS - step_num} steps remaining. "
                "Call finish soon."
            )

        return "\n".join(hints) if hints else ""

    # ── Synthesis ──────────────────────────────────────────────────────────────

    @staticmethod
    def _synthesis_prompt(query: str, context: str) -> str:
        return (
            f"Based on the research context below, write a comprehensive answer.\n\n"
            f"Question: {query}\n\nContext:\n{context[:3500]}\n\n"
            f"Write a detailed, well-structured answer. "
            f"Cite papers as [Source: Paper Title]."
        )

    def _synthesise(self, query: str, context: str) -> str:
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=self._synthesis_prompt(query, context),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=config.GEMINI_MAX_TOKENS,
                ),
            )
            return resp.text.strip()
        except Exception as exc:
            log.error("Synthesis failed: %s", exc)
            return context[:1000]

    def _synthesise_stream(self, query: str, context: str):
        """Yield synthesis text chunks. Falls back to a single blocking call if the
        streaming API yields nothing (immediate error or empty response)."""
        produced = False
        try:
            for chunk in self._client.models.generate_content_stream(
                model=config.GEMINI_MODEL,
                contents=self._synthesis_prompt(query, context),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=config.GEMINI_MAX_TOKENS,
                ),
            ):
                text = getattr(chunk, "text", None)
                if text:
                    produced = True
                    yield text
        except Exception as exc:
            log.error("Streaming synthesis failed: %s", exc)
        if not produced:
            yield self._synthesise(query, context)

    # ── Prompt builders ────────────────────────────────────────────────────────

    def _build_system(self, current_question: str, loop_hint: str = "") -> str:
        conv  = self._memory.format_for_prompt()
        notes = self._research.format_for_prompt()
        
        base_system = _REACT_SYSTEM
        if getattr(self, "_custom_instructions", None):
            base_system += f"\n\n**Custom Workspace Instructions (strictly obey):**\n{self._custom_instructions}"
            
        return base_system.format(
            tool_descriptions    = format_tool_descriptions(self._tools),
            conversation_context = f"\n**Recent conversation:**\n{conv}" if conv else "",
            research_notes       = f"\n{notes}" if notes else "",
            loop_hint            = f"\n**⚠ Progress tracker:**\n{loop_hint}" if loop_hint else "",
            max_steps            = config.AGENT_MAX_STEPS,
            current_question     = current_question,
        )

    def _build_user(self, original: str, current: str, scratchpad: list[Step]) -> str:
        lines = [f"Original question: {original}"]
        if current != original:
            lines.append(f"Current sub-question: {current}")
        if scratchpad:
            lines.append("\nScratchpad so far:")
            for i, s in enumerate(scratchpad, 1):
                lines.append(f"\nStep {i}:")
                lines.append(f"  Thought: {s.thought}")
                lines.append(f"  Action: {s.action}({s.action_input[:80]})")
                if s.observation:
                    lines.append(f"  Observation: {s.observation[:500]}…")
        lines.append("\nNext step? JSON only.")
        return "\n".join(lines)

    # ── Model call ─────────────────────────────────────────────────────────────

    def _call_model(self, system: str, user: str) -> str:
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    max_output_tokens=600,
                ),
            )
            return resp.text.strip()
        except Exception as exc:
            log.error("Model call failed: %s", exc)
            return '{"thought":"error","action":"finish","action_input":"[Agent error — please try again]"}'

    # ── JSON parsing ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str) -> dict | None:
        try:
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            parsed = json.loads(raw)
            if all(k in parsed for k in ("thought", "action", "action_input")):
                return parsed
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*"thought"[^{}]*\}', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return None

    # ── Source extraction ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_sources(steps: list[Step]) -> list[dict]:
        seen: dict[str, float] = {}
        for step in steps:
            if not step.observation:
                continue
            for m in re.finditer(
                r'\*\*(.+?)\*\*.*?\((?:score|bm25):\s*([\d.]+)\)', step.observation
            ):
                title = m.group(1).strip()
                score = float(m.group(2))
                if title not in seen or score > seen[title]:
                    seen[title] = score
        return [{"title": t, "score": round(s, 3)}
                for t, s in sorted(seen.items(), key=lambda x: -x[1])]

    # ── Follow-up, Citation & Paper Lookup Upgrades ───────────────────────────

    def generate_follow_ups(self, query: str, response: str, sources: list) -> list[str]:
        """Generates 3-4 contextual follow-up suggestions using Gemini."""
        system_instruction = (
            "You are a helpful ML research assistant. The user has just asked a question, and the assistant responded.\n"
            "Generate 3-4 short, highly specific, and interesting follow-up questions (under 12 words each) "
            "that the user might want to ask next to deepen their research.\n"
            "Focus on specific parameters, method comparisons, future directions, or limits mentioned in the response.\n"
            "Format the output as a valid JSON list of strings, for example: [\"question 1\", \"question 2\", \"question 3\"]\n"
            "Respond with ONLY the raw JSON list of strings, no markdown codeblocks, no extra explanation."
        )
        sources_str = ", ".join(s.get("title", "") for s in sources[:3])
        prompt = (
            f"Original Query: '{query}'\n"
            f"Assistant Response: '{response[:1000]}...'\n"
            f"Sources Cited: {sources_str}\n"
            f"Generate follow-up suggestions:"
        )
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=150,
                ),
            )
            raw = resp.text.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
            questions = json.loads(raw)
            if isinstance(questions, list):
                return [q.strip() for q in questions[:4] if q.strip()]
        except Exception as e:
            log.warning("Failed to generate follow-ups: %s", e)
        return [
            "Can you explain the methodology details?",
            "What are the experimental baselines compared?",
            "Are there any open questions or future directions?"
        ]

    def _build_papers_lookup(self) -> dict[str, str]:
        """Builds a mapping from lowercase paper titles to paper_ids."""
        lookup = {}
        try:
            meta_path = config.DATA_DIR / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    corpus_papers = json.load(f)
                    for p in corpus_papers:
                        if "title" in p and "paper_id" in p:
                            lookup[p["title"].lower().strip()] = p["paper_id"]
        except Exception as e:
            log.warning("Failed to load corpus metadata for citation mapping: %s", e)

        try:
            if self._router and self._router._session:
                session_papers = self._router._session.list_papers()
                for p in session_papers:
                    if "title" in p and "paper_id" in p:
                        lookup[p["title"].lower().strip()] = p["paper_id"]
        except Exception as e:
            log.warning("Failed to load session papers for citation mapping: %s", e)

        return lookup

    def _wrap_citations(self, text: str, lookup: dict[str, str]) -> str:
        """
        Scans response text for patterns like [Source: Title] or [Source: Paper Title],
        resolves Title against the lookup mapping, and wraps it in a citation span:
        <span class="citation" data-paper-id="PAPER_ID">[Source: Title]</span>
        """
        def repl(match):
            citation_text = match.group(0)
            title_text = match.group(1).strip()
            match_key = title_text.lower().strip()
            
            paper_id = lookup.get(match_key)
            if not paper_id:
                clean_title = re.sub(r"[^\w\s]", "", match_key)
                for t, pid in lookup.items():
                    clean_t = re.sub(r"[^\w\s]", "", t)
                    if clean_title in clean_t or clean_t in clean_title:
                        paper_id = pid
                        break
                        
            if paper_id:
                return f'<span class="citation" data-paper-id="{paper_id}">{citation_text}</span>'
            return citation_text

        return re.sub(r"\[Source:\s*(.+?)\]", repl, text)