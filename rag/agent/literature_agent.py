"""
literature_agent.py – Orchestrates a full literature review from topic to document.

Workflow:
  1. Decompose topic into N themes
  2. For each theme: deep multi-source research via ReAct loop
  3. Synthesise academic prose per section
  4. Self-critique coverage gaps
  5. Assemble final document with formatted references
"""
from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from rag.llm import get_client
from google.genai import types

from rag import config
from rag.agent.citation_formatter import CitationFormatter, PaperMeta
from rag.processing.chunker import Chunk
from rag.sources.source_router import SourceConfig

log = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Theme:
    title:        str
    search_query: str
    description:  str = ""


@dataclass
class ThemeSection:
    theme:   Theme
    content: str
    sources: list[dict] = field(default_factory=list)


@dataclass
class LiteratureReview:
    topic:          str
    citation_format: str
    introduction:   str
    sections:       list[ThemeSection]
    gaps:           str
    future_work:    str
    conclusion:     str
    references:     list[PaperMeta]
    total_papers:   int  = 0
    latency_ms:     float = 0.0

    def to_markdown(self) -> str:
        fmt   = CitationFormatter(self.citation_format)
        lines = []

        lines.append(f"# Literature Review: {self.topic}\n")
        lines.append(f"*Citation format: {self.citation_format}*\n")
        lines.append("---\n")

        lines.append("## 1. Introduction\n")
        lines.append(self.introduction)
        lines.append("")

        for i, sec in enumerate(self.sections, 2):
            lines.append(f"\n## {i}. {sec.theme.title}\n")
            lines.append(sec.content)
            lines.append("")

        next_num = len(self.sections) + 2
        lines.append(f"\n## {next_num}. Research Gaps and Open Problems\n")
        lines.append(self.gaps)
        lines.append("")

        lines.append(f"\n## {next_num + 1}. Future Directions\n")
        lines.append(self.future_work)
        lines.append("")

        lines.append(f"\n## {next_num + 2}. Conclusion\n")
        lines.append(self.conclusion)
        lines.append("")

        if self.references:
            lines.append(f"\n## References\n")
            lines.append(fmt.format_list(self.references))

        return "\n".join(lines)


# ── Theme decomposition ────────────────────────────────────────────────────────

_THEME_SYSTEM = """\
You are an academic research planner. Given a research topic, identify the key themes
for a comprehensive literature review.

Respond ONLY with valid JSON:
{
  "themes": [
    {"title": "Theme Title", "search_query": "search terms", "description": "what this covers"},
    ...
  ]
}

Rules:
- Between 3 and 6 themes
- Each theme should be distinct and non-overlapping
- search_query should be specific (2-6 keywords)
- Themes should progress logically: foundations → methods → applications → evaluation
"""

_SECTION_SYSTEM = """\
You are an academic writer producing a literature review section.
Write 3-5 paragraphs of scholarly prose based on the retrieved papers.
- Use formal academic language
- Cite papers inline as (Author et al., Year) or [Source: Title] where Year is available
- Compare and contrast approaches where possible
- Note methodological similarities and differences
- End with a brief synthesis of what this body of work shows
Do NOT use bullet points. Write flowing academic paragraphs only.
"""

_GAP_SYSTEM = """\
You are a senior researcher identifying gaps in a literature review.
Based on the topic and content covered, identify:
1. What has NOT been studied adequately
2. Contradictions or conflicting findings
3. Methodological limitations in existing work
4. Under-explored populations, domains, or settings
Write 2-3 paragraphs of academic prose.
"""

_FUTURE_SYSTEM = """\
Based on the research gaps identified and the current state of the field,
suggest 2-3 concrete future research directions. Be specific about:
- What should be studied
- What methods or approaches to use
- Why this matters for the field
Write 2-3 paragraphs of academic prose.
"""

_CONCLUSION_SYSTEM = """\
Write a concise conclusion (1-2 paragraphs) for the literature review.
Summarise the key findings across all themes, the state of the field,
and the most important open challenges. Academic prose only.
"""

_INTRO_SYSTEM = """\
Write an introduction (2-3 paragraphs) for a literature review on the given topic.
Cover: why this topic matters, scope of the review, what themes will be covered,
and how the review is organised. Academic prose only.
"""


class LiteratureAgent:
    """
    Produces a full structured literature review from a research topic.
    Uses the source router for retrieval across all available paper sources.
    """

    def __init__(self, router, bm25, session_index) -> None:
        self._router  = router
        self._bm25    = bm25
        self._session = session_index
        self._client  = get_client()
        self._formatter = None
        self._cited_papers: dict[str, PaperMeta] = {}
        
        # Load corpus metadata for resolving publication details (like year)
        self._corpus_meta = {}
        meta_path = config.DATA_DIR / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    for p in json.load(f):
                        if "paper_id" in p:
                            self._corpus_meta[p["paper_id"]] = p
            except Exception as exc:
                log.warning("Could not load corpus metadata in LiteratureAgent: %s", exc)

    def run(
        self,
        topic:           str,
        citation_format: str  = "APA",
        n_themes:        int  = 4,
        use_live_arxiv:  bool = False,
        allowed_paper_ids: set[str] | None = None,
        source_cfg: SourceConfig | None = None,
        progress_cb:     Callable[[str], None] | None = None,
    ) -> LiteratureReview:
        t0 = time.monotonic()
        self._cited_papers = {}
        self._formatter    = CitationFormatter(citation_format)

        def _progress(msg: str):
            log.info(msg)
            if progress_cb:
                progress_cb(msg)

        # 1. Decompose into themes
        _progress("🗺 Decomposing topic into themes…")
        themes = self._decompose_topic(topic, n_themes)
        _progress(f"✓ Identified {len(themes)} themes: {[t.title for t in themes]}")

        # 2. Research each theme
        sections: list[ThemeSection] = []
        for i, theme in enumerate(themes, 1):
            _progress(f"🔍 Researching theme {i}/{len(themes)}: {theme.title}…")
            section = self._research_theme(
                theme, topic, use_live_arxiv,
                allowed_paper_ids=allowed_paper_ids,
                source_cfg=source_cfg
            )
            sections.append(section)
            _progress(f"✓ Theme {i} done ({len(section.sources)} sources)")

        # 3. Collect all context for meta-sections
        all_context = "\n\n".join(
            f"[Theme: {s.theme.title}]\n{s.content[:600]}" for s in sections
        )

        # 4. Write meta-sections
        _progress("✍ Writing introduction…")
        introduction = self._write_section(
            prompt=f"Topic: {topic}\nThemes covered: {[t.title for t in themes]}\n\nWrite the introduction.",
            system=_INTRO_SYSTEM,
        )

        _progress("🔎 Identifying research gaps…")
        gaps = self._write_section(
            prompt=f"Topic: {topic}\n\nContent covered:\n{all_context[:2000]}\n\nIdentify research gaps.",
            system=_GAP_SYSTEM,
        )

        _progress("🚀 Writing future directions…")
        future = self._write_section(
            prompt=f"Topic: {topic}\n\nGaps identified:\n{gaps}\n\nSuggest future research directions.",
            system=_FUTURE_SYSTEM,
        )

        _progress("📝 Writing conclusion…")
        conclusion = self._write_section(
            prompt=f"Topic: {topic}\n\nKey findings:\n{all_context[:2000]}\n\nWrite the conclusion.",
            system=_CONCLUSION_SYSTEM,
        )

        # 5. Assemble references
        references = list(self._cited_papers.values())
        latency    = round((time.monotonic() - t0) * 1000, 1)

        _progress(f"✅ Literature review complete! {len(references)} papers cited.")

        return LiteratureReview(
            topic           = topic,
            citation_format = citation_format,
            introduction    = introduction,
            sections        = sections,
            gaps            = gaps,
            future_work     = future,
            conclusion      = conclusion,
            references      = references,
            total_papers    = len(references),
            latency_ms      = latency,
        )

    # ── Theme decomposition ────────────────────────────────────────────────────

    def _decompose_topic(self, topic: str, n_themes: int) -> list[Theme]:
        cfg = types.GenerateContentConfig(
            system_instruction=_THEME_SYSTEM,
            temperature=0.0,
            max_output_tokens=600,
        )
        prompt = f"Research topic: {topic}\nTarget number of themes: {n_themes}"
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL, contents=prompt, config=cfg,
            )
            raw    = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
            parsed = json.loads(raw)
            themes = []
            for t in parsed.get("themes", [])[:config.LIT_REVIEW_MAX_THEMES]:
                themes.append(Theme(
                    title        = t.get("title", ""),
                    search_query = t.get("search_query", t.get("title", "")),
                    description  = t.get("description", ""),
                ))
            if themes:
                return themes
        except Exception as exc:
            log.warning("Theme decomposition failed: %s", exc)

        # Fallback: generic themes
        return [
            Theme("Foundations and Background",      f"{topic} foundations background"),
            Theme("Methods and Approaches",          f"{topic} methods techniques algorithms"),
            Theme("Empirical Results and Benchmarks",f"{topic} evaluation benchmarks results"),
            Theme("Applications and Case Studies",   f"{topic} applications real world"),
        ]

    # ── Theme research ─────────────────────────────────────────────────────────

    def _research_theme(
        self, theme: Theme, topic: str, use_live_arxiv: bool,
        allowed_paper_ids: set[str] | None = None,
        source_cfg: SourceConfig | None = None
    ) -> ThemeSection:
        # Multi-angle retrieval
        queries = [
            theme.search_query,
            f"{topic} {theme.search_query}",
            f"{theme.title} {topic}",
        ]

        all_results: list[tuple[Chunk, float]] = []
        seen_ids: set[str] = set()

        for q in queries:
            results = self._router.search(q, top_k=5, source_cfg=source_cfg, allowed_paper_ids=allowed_paper_ids)
            for chunk, score in results:
                if chunk.chunk_id not in seen_ids:
                    all_results.append((chunk, score))
                    seen_ids.add(chunk.chunk_id)

        # Optionally fetch from live arXiv
        if use_live_arxiv and len(all_results) < 6:
            from rag.sources.arxiv_fetcher import search_arxiv, fetch_paper_chunks
            arxiv_results = search_arxiv(theme.search_query, max_results=2)
            for meta in arxiv_results:
                pid = meta["paper_id"]
                if not self._session.has_paper(pid):
                    chunks = fetch_paper_chunks(meta)
                    if chunks:
                        self._session.add_chunks(chunks, meta)
                        # Re-search now that new chunks are available
                        new_results = self._router.search(theme.search_query, top_k=3, source_cfg=source_cfg, allowed_paper_ids=allowed_paper_ids)
                        for chunk, score in new_results:
                            if chunk.chunk_id not in seen_ids:
                                all_results.append((chunk, score))
                                seen_ids.add(chunk.chunk_id)

        # Sort by score, keep top-8
        all_results.sort(key=lambda x: -x[1])
        top_results = all_results[:8]

        # Track cited papers
        sources = []
        for chunk, score in top_results:
            sources.append({"title": chunk.title, "score": round(score, 3),
                            "source": chunk.source})
            # Register for reference list
            if chunk.paper_id not in self._cited_papers:
                # Try to resolve paper year, authors, url
                year = "2025"
                authors = chunk.authors if hasattr(chunk, "authors") and chunk.authors else []
                url = ""
                doi = ""
                venue = "arXiv"
                citation_count = 0
                
                # Check corpus metadata first
                if chunk.paper_id in self._corpus_meta:
                    meta = self._corpus_meta[chunk.paper_id]
                    year = meta.get("published", "")[:4] or "2025"
                    authors = meta.get("authors", authors)
                    url = meta.get("url", "")
                # Check session index next
                elif self._session.has_paper(chunk.paper_id):
                    meta = self._session._paper_meta[chunk.paper_id]
                    year = meta.get("published", "")[:4] or meta.get("year", "2025")
                    authors = meta.get("authors", authors)
                    url = meta.get("url", "")
                
                # Live Semantic Scholar Enrichment
                try:
                    from rag.sources.arxiv_fetcher import enrich_with_semantic_scholar
                    ss_meta = enrich_with_semantic_scholar(chunk.paper_id)
                    if ss_meta.get("year"):
                        year = ss_meta["year"]
                    doi = ss_meta.get("doi", "")
                    venue = ss_meta.get("venue", "") or "arXiv"
                    citation_count = ss_meta.get("citation_count", 0)
                except Exception as ss_err:
                    log.warning("SS enrichment failed in LiteratureAgent: %s", ss_err)
                
                self._cited_papers[chunk.paper_id] = PaperMeta(
                    paper_id = chunk.paper_id,
                    title    = chunk.title,
                    authors  = authors,
                    year     = year,
                    url      = url,
                    source   = chunk.source,
                    doi      = doi,
                    venue    = venue,
                    citation_count = citation_count,
                )

        # Build context for section writing
        context = "\n\n".join(
            f"[Paper: {c.title}]\n{c.text[:500]}…"
            for c, _ in top_results
        )

        # Write section
        content = self._write_section(
            prompt=(
                f"Research topic: {topic}\n"
                f"Section theme: {theme.title}\n"
                f"Theme description: {theme.description}\n\n"
                f"Retrieved passages:\n{context}\n\n"
                f"Write the literature review section for this theme."
            ),
            system=_SECTION_SYSTEM,
        )

        return ThemeSection(theme=theme, content=content, sources=sources)

    # ── Section writer ─────────────────────────────────────────────────────────

    def _write_section(self, prompt: str, system: str) -> str:
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,
            max_output_tokens=config.GEMINI_MAX_TOKENS,
        )
        try:
            resp = self._client.models.generate_content(
                model=config.GEMINI_MODEL, contents=prompt, config=cfg,
            )
            return resp.text.strip()
        except Exception as exc:
            log.error("Section write failed: %s", exc)
            return f"[Section generation error: {exc}]"

    def regenerate_section(
        self,
        review:          LiteratureReview,
        section_idx:     int,
        use_live_arxiv:  bool = False,
        allowed_paper_ids: set[str] | None = None,
        source_cfg:      SourceConfig | None = None,
        progress_cb:     Callable[[str], None] | None = None,
    ) -> LiteratureReview:
        """
        Regenerates a single specific theme section, re-running research and synthesis,
        and updates the reference list accordingly.
        """
        t0 = time.monotonic()
        if section_idx < 0 or section_idx >= len(review.sections):
            raise ValueError(f"Invalid section index: {section_idx}")

        def _progress(msg: str):
            log.info(msg)
            if progress_cb:
                progress_cb(msg)

        self._formatter = CitationFormatter(review.citation_format)
        
        # 1. Rebuild self._cited_papers from all OTHER sections to avoid losing them
        self._cited_papers = {}
        for idx, sec in enumerate(review.sections):
            if idx == section_idx:
                continue
            for src in sec.sources:
                paper_id = None
                title = src.get("title", "")
                
                # Check corpus meta
                for pid, meta in self._corpus_meta.items():
                    if meta.get("title", "") == title:
                        paper_id = pid
                        break
                
                # Check session index
                if not paper_id and self._session:
                    for paper in self._session.list_papers():
                        if paper.get("title", "") == title:
                            paper_id = paper.get("paper_id")
                            break
                            
                if paper_id:
                    year = "2025"
                    authors = []
                    url = ""
                    doi = ""
                    venue = "arXiv"
                    citation_count = 0
                    source = src.get("source", "corpus")
                    
                    if paper_id in self._corpus_meta:
                        meta = self._corpus_meta[paper_id]
                        year = meta.get("published", "")[:4] or "2025"
                        authors = meta.get("authors", [])
                        url = meta.get("url", "")
                    elif self._session and self._session.has_paper(paper_id):
                        meta = self._session._paper_meta[paper_id]
                        year = meta.get("published", "")[:4] or meta.get("year", "2025")
                        authors = meta.get("authors", [])
                        url = meta.get("url", "")
                    
                    # Live Semantic Scholar Enrichment
                    try:
                        from rag.sources.arxiv_fetcher import enrich_with_semantic_scholar
                        ss_meta = enrich_with_semantic_scholar(paper_id)
                        if ss_meta.get("year"):
                            year = ss_meta["year"]
                        doi = ss_meta.get("doi", "")
                        venue = ss_meta.get("venue", "") or "arXiv"
                        citation_count = ss_meta.get("citation_count", 0)
                    except Exception as ss_err:
                        log.warning("SS enrichment failed in regenerate_section: %s", ss_err)
                        
                    self._cited_papers[paper_id] = PaperMeta(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        year=year,
                        url=url,
                        source=source,
                        doi=doi,
                        venue=venue,
                        citation_count=citation_count,
                    )

        # 2. Re-research and synthesize the targeted theme
        target_section = review.sections[section_idx]
        theme = target_section.theme
        _progress(f"🔄 Regenerating theme section: {theme.title}…")
        
        new_section = self._research_theme(
            theme, review.topic, use_live_arxiv,
            allowed_paper_ids=allowed_paper_ids,
            source_cfg=source_cfg
        )
        review.sections[section_idx] = new_section
        
        # 3. Update the global reference list
        review.references = list(self._cited_papers.values())
        review.total_papers = len(review.references)
        review.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        
        _progress("✓ Section regeneration complete!")
        return review
