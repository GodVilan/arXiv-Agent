"""
tools.py – Tool registry for the ReAct agent.
All tools are source-aware: they query corpus, session (live arXiv + uploads), or both.
"""
from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable
from rag import config
from rag.sources.arxiv_fetcher import search_arxiv, fetch_paper_chunks
from rag.sources.source_router import SourceConfig

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name:          str
    description:   str
    fn:            Callable | None
    example_input: str = ""


def _search_corpus(query: str, router, top_k: int = config.DEFAULT_TOP_K,
                   source_cfg: SourceConfig | None = None,
                   allowed_paper_ids: set[str] | None = None) -> str:
    results = router.search(query, top_k=top_k, source_cfg=source_cfg, allowed_paper_ids=allowed_paper_ids)
    if not results:
        return "No relevant passages found."
    lines = []
    for i, (chunk, score) in enumerate(results, 1):
        src = f"[{chunk.source.upper()}]"
        lines.append(
            f"[{i}] **{chunk.title}** {src} (score: {score:.3f})\n{chunk.text[:400]}…"
        )
    return "\n\n".join(lines)


def _keyword_search(query: str, bm25, top_k: int = config.DEFAULT_TOP_K,
                    allowed_paper_ids: set[str] | None = None) -> str:
    results = bm25.retrieve(query, top_k=top_k, allowed_paper_ids=allowed_paper_ids)
    if not results:
        return "No keyword matches found."
    lines = [
        f"[{i}] **{c.title}** (bm25: {s:.3f})\n{c.text[:400]}…"
        for i, (c, s) in enumerate(results, 1)
    ]
    return "\n\n".join(lines)


def _fetch_arxiv(query: str, session_index, chunk_size: int = config.DEFAULT_CHUNK, is_enabled: bool = True) -> str:
    """Search arXiv, fetch PDF, add to session index, return abstract summary."""
    if not is_enabled:
        return "Live arXiv search & fetch is disabled for this conversation thread. You can enable it in the conversation settings."

    results = search_arxiv(query, max_results=3)
    if not results:
        return "No papers found on arXiv for that query."

    output_lines = []
    for meta in results[:2]:
        title = meta["title"]
        pid   = meta["paper_id"]

        if session_index.has_paper(pid):
            output_lines.append(f"✓ Already indexed: **{title}**\nAbstract: {meta['abstract'][:400]}…")
            continue

        # Fetch full PDF and add to session
        chunks = fetch_paper_chunks(meta, chunk_size=chunk_size)
        if chunks:
            added = session_index.add_chunks(chunks, meta)
            output_lines.append(
                f"✓ Fetched & indexed: **{title}** ({added} chunks)\n"
                f"Authors: {', '.join(meta['authors'][:3])}\n"
                f"Abstract: {meta['abstract'][:400]}…"
            )
        else:
            output_lines.append(
                f"⚠ Could not download PDF for **{title}**\n"
                f"Abstract only: {meta['abstract'][:400]}…"
            )
        time.sleep(0.5)

    return "\n\n---\n\n".join(output_lines)


def _summarize_paper(query: str, router, top_k: int = 8, allowed_paper_ids: set[str] | None = None) -> str:
    results = router.search(query, top_k=top_k, allowed_paper_ids=allowed_paper_ids)
    if not results:
        return f"Could not find '{query}' in any source."
    from collections import Counter
    paper_counts = Counter(c.paper_id for c, _ in results)
    top_pid      = paper_counts.most_common(1)[0][0]
    paper_chunks = [(c, s) for c, s in results if c.paper_id == top_pid]
    title = paper_chunks[0][0].title
    body  = "\n\n".join(c.text for c, _ in paper_chunks[:5])
    return f"**Paper:** {title}\n**ID:** {top_pid}\n\n{body[:1800]}…"


def _compare_papers(input_str: str, router, top_k: int = 6, allowed_paper_ids: set[str] | None = None) -> str:
    parts = [p.strip() for p in input_str.split("|")]
    if len(parts) < 2:
        return "Format: 'topic_A | topic_B | aspect'"
    topic_a = parts[0]
    topic_b = parts[1]
    aspect  = parts[2] if len(parts) > 2 else "comparison"
    res_a   = router.search(f"{topic_a} {aspect}", top_k=top_k // 2, allowed_paper_ids=allowed_paper_ids)
    res_b   = router.search(f"{topic_b} {aspect}", top_k=top_k // 2, allowed_paper_ids=allowed_paper_ids)

    def _fmt(res, label):
        if not res:
            return f"**{label}:** No passages found."
        return f"**{label}:**\n" + "\n\n".join(
            f"[{c.title}]\n{c.text[:350]}…" for c, _ in res
        )

    return f"{_fmt(res_a, topic_a)}\n\n{'─'*50}\n\n{_fmt(res_b, topic_b)}"


def _trace_bibliography(query: str, router, allowed_paper_ids: set[str] | None = None) -> str:
    """Search for paper references/bibliography and extract cited paper titles/authors using Gemini."""
    results = router.search(query, top_k=8, allowed_paper_ids=allowed_paper_ids)
    if not results:
        return f"Could not find paper '{query}' in the index."
    
    from collections import Counter
    paper_counts = Counter(c.paper_id for c, _ in results)
    top_pid = paper_counts.most_common(1)[0][0]
    
    # Target reference/bibliography chunks
    ref_results = router.search(f"{top_pid} references bibliography bibliography references", top_k=6, allowed_paper_ids=allowed_paper_ids)
    bib_text = "\n\n".join(c.text for c, _ in ref_results if c.paper_id == top_pid)
    if not bib_text.strip():
        bib_text = "\n\n".join(c.text for c, _ in results if c.paper_id == top_pid)
        
    from google.genai import types
    from rag.llm import get_client
    client = get_client()
    
    system_instruction = (
        "You are a bibliography extractor. Extract up to 5 key papers cited in this paper text.\n"
        "For each extracted paper, provide the Title and Authors in a clean structured list."
    )
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=f"Extract key cited papers from this text:\n\n{bib_text[:3000]}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                max_output_tokens=400,
            ),
        )
        return f"**Bibliography for paper {top_pid}:**\n\n{resp.text.strip()}"
    except Exception as e:
        return f"Failed to extract bibliography: {e}"


def build_tool_registry(
    router, bm25, session_index,
    get_allowed_paper_ids=None,
    get_source_config=None,
    get_use_arxiv=None
) -> dict[str, Tool]:
    return {
        "search_corpus": Tool(
            name="search_corpus",
            description=(
                "Semantic search across ALL available sources: the 150-paper corpus, "
                "any live-fetched arXiv papers, and user-uploaded PDFs. "
                "Best for broad topic questions. Input: natural language query."
            ),
            fn=lambda q: _search_corpus(
                q, router,
                source_cfg=get_source_config() if get_source_config else None,
                allowed_paper_ids=get_allowed_paper_ids() if get_allowed_paper_ids else None
            ),
            example_input="continual learning catastrophic forgetting",
        ),
        "keyword_search": Tool(
            name="keyword_search",
            description=(
                "Sparse BM25 keyword search on the corpus. "
                "Best for exact method names, author names, paper IDs. "
                "Input: keywords or exact phrase."
            ),
            fn=lambda q: _keyword_search(
                q, bm25,
                allowed_paper_ids=get_allowed_paper_ids() if get_allowed_paper_ids else None
            ),
            example_input="LoRA low-rank adaptation Hu 2021",
        ),
        "fetch_arxiv": Tool(
            name="fetch_arxiv",
            description=(
                "Search arXiv live, download the full PDF, and add it to the search index. "
                "Use when the local corpus lacks papers on a specific topic or by a specific author. "
                "Input: a topic query or arXiv ID (e.g. '2106.09685')."
            ),
            fn=lambda q: _fetch_arxiv(
                q, session_index,
                is_enabled=get_use_arxiv() if get_use_arxiv else True
            ),
            example_input="RLHF reward model training 2024",
        ),
        "summarize_paper": Tool(
            name="summarize_paper",
            description=(
                "Retrieve and summarise a specific paper from any source. "
                "Input: the paper title or paper_id."
            ),
            fn=lambda q: _summarize_paper(
                q, router,
                allowed_paper_ids=get_allowed_paper_ids() if get_allowed_paper_ids else None
            ),
            example_input="Attention Is All You Need",
        ),
        "compare_papers": Tool(
            name="compare_papers",
            description=(
                "Side-by-side retrieval for two approaches or papers. "
                "Input format: 'topic_A | topic_B | aspect'. "
                "Example: 'LoRA | prefix tuning | parameter efficiency'"
            ),
            fn=lambda q: _compare_papers(
                q, router,
                allowed_paper_ids=get_allowed_paper_ids() if get_allowed_paper_ids else None
            ),
            example_input="LoRA | adapter layers | inference overhead",
        ),
        "trace_bibliography": Tool(
            name="trace_bibliography",
            description=(
                "Extract bibliography citations/references from an indexed paper's reference section "
                "to find parent papers and trace scientific origins. "
                "Input: paper title, paper_id, or search query."
            ),
            fn=lambda q: _trace_bibliography(
                q, router,
                allowed_paper_ids=get_allowed_paper_ids() if get_allowed_paper_ids else None
            ),
            example_input="Attention Is All You Need",
        ),
        "finish": Tool(
            name="finish",
            description=(
                "Emit the final answer. Use only when you have enough information. "
                "Input: your complete, well-cited answer."
            ),
            fn=None,
            example_input="Based on [Source: Paper Title], ...",
        ),
    }


def format_tool_descriptions(tools: dict[str, Tool]) -> str:
    return "\n\n".join(
        f"• **{t.name}**: {t.description}"
        + (f'\n  Example: "{t.example_input}"' if t.example_input else "")
        for t in tools.values()
    )
