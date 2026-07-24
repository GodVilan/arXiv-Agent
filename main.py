"""
main.py – CLI for the arXiv Agent.

Usage:
    python main.py                          # interactive Q&A
    python main.py --query "What is LoRA?" # single query
    python main.py --review "Continual Learning" --format APA --themes 4
    python main.py --list                   # list corpus papers
    python main.py --notes                  # show saved research notes
    python main.py --verbose                # show full scratchpad
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from rag import config
from rag.processing.chunker import process_papers, save_chunks, load_chunks
from rag.retrieval.dense import Retriever
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.embeddings import EmbeddingModel
from rag.sources.session_index import SessionIndex
from rag.sources.source_router import SourceRouter
from rag.agent import (
    ReActAgent, LiteratureAgent,
    ConversationMemory, ResearchMemory,
)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║  arXiv Agent — ML Research Assistant                            ║
║  ReAct · Literature Review · BGE · Multi-Source · Gemini 2.5   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def parse_args():
    p = argparse.ArgumentParser(description="arXiv Agent CLI")
    p.add_argument("--query",   type=str,  default=None, help="Single Q&A query")
    p.add_argument("--review",  type=str,  default=None, help="Generate a literature review on this topic")
    p.add_argument("--format",  type=str,  default="APA", choices=config.CITATION_FORMATS, help="Citation format")
    p.add_argument("--themes",  type=int,  default=4,    help="Number of themes for literature review")
    p.add_argument("--live",    action="store_true",     help="Fetch live arXiv papers during review")
    p.add_argument("--list",    action="store_true",     help="List corpus papers")
    p.add_argument("--notes",   action="store_true",     help="Show research notes")
    p.add_argument("--verbose", action="store_true",     help="Show full agent scratchpad")
    p.add_argument("--output",  type=str,  default=None, help="Save literature review to file")
    return p.parse_args()


def load_corpus() -> list:
    cache = config.DATA_DIR / f"chunks_{config.DEFAULT_CHUNK}.json"
    if cache.exists():
        print("  Loading chunks from cache…")
        return load_chunks(cache)
    meta = config.DATA_DIR / "metadata.json"
    if not meta.exists():
        print("  No corpus found. Download papers first:")
        print("  python -c \"from rag.data.collector import download_papers; download_papers()\"")
        sys.exit(1)
    with open(meta) as f:
        papers = json.load(f)
    print(f"  Processing {len(papers)} papers…")
    chunks = process_papers(papers)
    save_chunks(chunks, cache)
    return chunks


def list_papers():
    meta = config.DATA_DIR / "metadata.json"
    if not meta.exists():
        print("No corpus. Download papers first.")
        return
    with open(meta) as f:
        papers = json.load(f)
    print(f"\n📚  {len(papers)} papers:\n")
    for i, p in enumerate(papers, 1):
        print(f"  {i:3d}. {p['title']}")
        print(f"       {p['paper_id']}  |  {p['published'][:10]}")


def build_system(chunks):
    print(f"\nBuilding BGE retriever…")
    dense = Retriever.build(
        chunks=chunks,
        chunk_size=config.DEFAULT_CHUNK,
        index_dir=config.RESULTS_DIR / "indices",
    )
    print("  Building BM25 index…")
    bm25 = BM25Retriever(chunks)
    print("  Initialising session index…")
    emb_model   = EmbeddingModel()
    session_idx = SessionIndex(emb_model)
    router      = SourceRouter(dense, session_idx)
    print("  Ready.\n")
    return dense, bm25, session_idx, router


STEP_COLORS = {
    "search_corpus":   "\033[94m",
    "keyword_search":  "\033[96m",
    "fetch_arxiv":     "\033[93m",
    "summarize_paper": "\033[95m",
    "compare_papers":  "\033[95m",
    "finish":          "\033[92m",
}
RESET = "\033[0m"


def print_step(i, step, verbose):
    color = STEP_COLORS.get(step.action, "\033[97m")
    print(f"\n  Step {i}")
    print(f"  💭 {step.thought[:120]}")
    print(f"  {color}⚡ {step.action}{RESET}({step.action_input[:80]}{'…' if len(step.action_input)>80 else ''})")
    if verbose and step.observation:
        print(f"  📋 {step.observation[:400]}{'…' if len(step.observation)>400 else ''}")


def run_query(agent, query, verbose):
    print(f"\n❓  {query}")
    print("─" * 70)
    response = agent.run(query)
    for i, step in enumerate(response.scratchpad, 1):
        print_step(i, step, verbose)
    if response.sub_questions:
        print(f"\n  🗺 Decomposed: {response.sub_questions}")
    if response.critique:
        verdict = "✅ passed" if response.critique.passed else "🔄 refined"
        print(f"\n  🔍 Self-critique: {verdict}")
    print(f"\n\n💬  Answer\n{'─'*70}")
    print(response.answer)
    if response.sources:
        print(f"\n📄  Sources:")
        for i, s in enumerate(response.sources[:6], 1):
            print(f"   [{i}] {s['title'][:70]}  ({s['score']:.3f})")
    print(f"\n⏱  {response.total_steps} steps · {response.latency_ms:.0f}ms")
    print("─" * 70)


def run_review(lit_agent, topic, citation_format, n_themes, use_live, output_path, verbose):
    print(f"\n📖  Generating literature review: '{topic}'")
    print(f"     Format: {citation_format} | Themes: {n_themes} | Live arXiv: {use_live}")
    print("─" * 70)

    def _progress(msg):
        print(f"  {msg}")

    review = lit_agent.run(
        topic           = topic,
        citation_format = citation_format,
        n_themes        = n_themes,
        use_live_arxiv  = use_live,
        progress_cb     = _progress,
    )

    print(f"\n✅  Done! {review.total_papers} papers · {review.latency_ms/1000:.1f}s")

    md = review.to_markdown()

    if output_path:
        Path(output_path).write_text(md, encoding="utf-8")
        print(f"📄  Saved to: {output_path}")
    else:
        out_path = f"lit_review_{topic[:30].replace(' ','_')}.md"
        Path(out_path).write_text(md, encoding="utf-8")
        print(f"📄  Saved to: {out_path}")


def main():
    args = parse_args()

    if args.list:
        list_papers()
        return

    if args.notes:
        mem = ResearchMemory()
        notes = mem.all_notes()
        print(f"\n📌  {len(notes)} notes:\n")
        for n in notes:
            print(f"  {n.key}: {n.content[:100]}")
        return

    print(BANNER)
    print("Loading corpus…")
    chunks = load_corpus()
    print(f"  {len(chunks):,} chunks ready.")

    dense, bm25, session_idx, router = build_system(chunks)

    conv_mem = ConversationMemory()
    res_mem  = ResearchMemory()
    agent    = ReActAgent(router, bm25, session_idx, conv_mem, res_mem)
    lit_agent = LiteratureAgent(router, bm25, session_idx)

    # Literature review mode
    if args.review:
        run_review(lit_agent, args.review, args.format, args.themes,
                   args.live, args.output, args.verbose)
        return

    # Single query mode
    if args.query:
        run_query(agent, args.query, args.verbose)
        return

    # Interactive Q&A
    print("Ask anything about ML research. Commands: 'review <topic>' | 'notes' | 'clear' | 'quit'\n")
    print("─" * 70)

    while True:
        try:
            query = input("\n❓  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if query.lower() == "clear":
            conv_mem.clear()
            print("  Conversation memory cleared.")
            continue
        if query.lower() == "notes":
            for n in res_mem.all_notes():
                print(f"  {n.key}: {n.content[:80]}")
            continue
        if query.lower().startswith("review "):
            topic = query[7:].strip()
            run_review(lit_agent, topic, args.format, args.themes,
                       args.live, args.output, args.verbose)
            continue

        run_query(agent, query, args.verbose)


if __name__ == "__main__":
    main()
