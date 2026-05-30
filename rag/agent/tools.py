"""
tools.py – Tool registry for the ReAct agent.

Each tool wraps a callable with a name, description, and schema.
The agent selects tools by reasoning about their descriptions.

Available tools
───────────────
search_corpus     Semantic search over the BGE/FAISS index (primary)
keyword_search    BM25 sparse search — best for exact terms / author names
fetch_arxiv       Pull a paper live from arXiv by ID or natural-language query
summarize_paper   Structured summary of one paper already in the corpus
compare_papers    Side-by-side comparison of two papers on a given aspect
finish            Emit the final answer and stop the loop
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Any

import arxiv   # pip install arxiv

from rag import config

log = logging.getLogger(__name__)


# ── Tool data-class ────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str          # shown verbatim to the model
    fn: Callable | None       # None for the special "finish" tool
    example_input: str = ""   # few-shot hint for the model


# ── Concrete tool implementations ──────────────────────────────────────────────

def _search_corpus(query: str, retriever, top_k: int = config.DEFAULT_TOP_K) -> str:
    """
    Semantic search over the BGE/FAISS index.
    Returns a formatted block of retrieved passages.
    """
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return "No relevant passages found in the corpus."

    lines = [f"[{i+1}] **{chunk.title}** (score: {score:.3f})\n{chunk.text[:400]}…"
             for i, (chunk, score) in enumerate(results)]
    return "\n\n".join(lines)


def _keyword_search(query: str, bm25_retriever, top_k: int = config.DEFAULT_TOP_K) -> str:
    """
    BM25 sparse search — best for exact method names, paper titles, author names.
    Returns a formatted block of retrieved passages.
    """
    results = bm25_retriever.retrieve(query, top_k=top_k)
    if not results:
        return "No keyword matches found in the corpus."

    lines = [f"[{i+1}] **{chunk.title}** (bm25_score: {score:.3f})\n{chunk.text[:400]}…"
             for i, (chunk, score) in enumerate(results)]
    return "\n\n".join(lines)


def _fetch_arxiv_paper(query: str, max_results: int = 3) -> str:
    """
    Pull papers from the live arXiv API by ID (e.g. '2106.09685')
    or natural-language query. Returns title + abstract + first page.
    """
    try:
        # Detect arXiv ID pattern
        import re
        id_pattern = re.compile(r"\b\d{4}\.\d{4,5}\b")
        match = id_pattern.search(query)

        client = arxiv.Client()

        if match:
            arxiv_id = match.group()
            search = arxiv.Search(id_list=[arxiv_id])
        else:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

        results = []
        for paper in client.results(search):
            results.append(
                f"**{paper.title}** ({paper.published.strftime('%Y-%m-%d')})\n"
                f"Authors: {', '.join(a.name for a in paper.authors[:3])}\n"
                f"Abstract: {paper.summary[:600]}…\n"
                f"URL: {paper.entry_id}"
            )
            time.sleep(0.5)   # be polite to the API

        return "\n\n---\n\n".join(results) if results else "No papers found on arXiv."

    except Exception as exc:
        log.error("arXiv fetch error: %s", exc)
        return f"[arXiv fetch failed: {exc}]"


def _summarize_paper(paper_title_or_id: str, retriever, top_k: int = 8) -> str:
    """
    Retrieve multiple chunks from one paper and return a structured summary
    covering: contribution, method, results, limitations.
    """
    results = retriever.retrieve(paper_title_or_id, top_k=top_k)
    if not results:
        return f"Could not find '{paper_title_or_id}' in the corpus."

    # Group by paper — take the paper that appears most
    from collections import Counter
    paper_counts = Counter(chunk.paper_id for chunk, _ in results)
    top_paper_id = paper_counts.most_common(1)[0][0]

    paper_chunks = [
        (chunk, score) for chunk, score in results
        if chunk.paper_id == top_paper_id
    ]

    title = paper_chunks[0][0].title
    body  = "\n\n".join(chunk.text for chunk, _ in paper_chunks[:5])

    return (
        f"**Paper:** {title}\n"
        f"**ID:** {top_paper_id}\n\n"
        f"**Corpus content (top chunks):**\n{body[:1800]}…"
    )


def _compare_papers(input_str: str, retriever, top_k: int = 6) -> str:
    """
    Input format: 'paper1 | paper2 | aspect'
    e.g. 'LoRA | prefix tuning | parameter efficiency'

    Retrieves chunks for each and returns side-by-side context.
    """
    parts = [p.strip() for p in input_str.split("|")]
    if len(parts) < 2:
        return (
            "Invalid input format. Use: 'paper_or_topic1 | paper_or_topic2 | aspect'"
        )

    topic_a  = parts[0]
    topic_b  = parts[1]
    aspect   = parts[2] if len(parts) > 2 else "general comparison"

    results_a = retriever.retrieve(f"{topic_a} {aspect}", top_k=top_k // 2)
    results_b = retriever.retrieve(f"{topic_b} {aspect}", top_k=top_k // 2)

    def _fmt(results, label):
        if not results:
            return f"**{label}:** No relevant passages found."
        chunks = "\n\n".join(
            f"[{chunk.title}]\n{chunk.text[:350]}…"
            for chunk, _ in results
        )
        return f"**{label}:**\n{chunks}"

    return f"{_fmt(results_a, topic_a)}\n\n{'─'*60}\n\n{_fmt(results_b, topic_b)}"


# ── Tool factory ───────────────────────────────────────────────────────────────

def build_tool_registry(retriever, bm25_retriever) -> dict[str, Tool]:
    """
    Bind retrievers to tool implementations and return the registry.
    Call once after the retrievers are loaded.
    """
    return {
        "search_corpus": Tool(
            name="search_corpus",
            description=(
                "Semantic search over 150 arXiv ML papers using BGE embeddings. "
                "Best for broad topic questions, concepts, and methods. "
                "Input: a natural language search query."
            ),
            fn=lambda q: _search_corpus(q, retriever),
            example_input="LoRA low-rank adaptation fine-tuning efficiency",
        ),
        "keyword_search": Tool(
            name="keyword_search",
            description=(
                "Sparse BM25 keyword search. Best for exact method names, "
                "paper titles, author names, or very specific technical terms. "
                "Input: keywords or an exact phrase."
            ),
            fn=lambda q: _keyword_search(q, bm25_retriever),
            example_input="BAAI BGE embedding retrieval 2024",
        ),
        "fetch_arxiv": Tool(
            name="fetch_arxiv",
            description=(
                "Fetch a paper directly from arXiv by ID (e.g. '2106.09685') "
                "or by a natural-language search query. Use this when you suspect "
                "a relevant paper is NOT in the local corpus. "
                "Input: an arXiv ID or search query string."
            ),
            fn=_fetch_arxiv_paper,
            example_input="2106.09685",
        ),
        "summarize_paper": Tool(
            name="summarize_paper",
            description=(
                "Retrieve a structured summary of a specific paper in the corpus. "
                "Input: the paper title or paper_id. "
                "Returns: contribution, method, and key results from that paper."
            ),
            fn=lambda q: _summarize_paper(q, retriever),
            example_input="Attention Is All You Need",
        ),
        "compare_papers": Tool(
            name="compare_papers",
            description=(
                "Side-by-side retrieval for comparing two papers or approaches. "
                "Input format: 'topic_or_paper_A | topic_or_paper_B | aspect_to_compare'. "
                "Example: 'LoRA | prefix tuning | parameter efficiency'"
            ),
            fn=lambda q: _compare_papers(q, retriever),
            example_input="LoRA | prefix tuning | parameter efficiency",
        ),
        "finish": Tool(
            name="finish",
            description=(
                "Provide the final answer to the user's question. "
                "Only use this when you have gathered sufficient information. "
                "Input: your complete, well-cited answer."
            ),
            fn=None,   # handled specially by the agent
            example_input="Based on [Source: Paper Title], ...",
        ),
    }


def format_tool_descriptions(tools: dict[str, Tool]) -> str:
    """Format all tool descriptions for inclusion in the system prompt."""
    lines = []
    for tool in tools.values():
        lines.append(
            f"• **{tool.name}**: {tool.description}"
            + (f"\n  Example input: \"{tool.example_input}\"" if tool.example_input else "")
        )
    return "\n\n".join(lines)
