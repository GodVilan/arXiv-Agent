"""
main.py – CLI for the arXiv ReAct Agent.

Usage:
    python main.py                         # interactive loop (BGE, top_k=5)
    python main.py --query "What is LoRA?" # single query and exit
    python main.py --list                  # list all papers in corpus
    python main.py --notes                 # show saved research notes
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
from rag.agent import ReActAgent, ConversationMemory, ResearchMemory

logging.basicConfig(level=logging.WARNING)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   arXiv ReAct Agent — ML Research Assistant                 ║
║   BGE · Multi-hop · Self-critique · Conversation Memory     ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args():
    p = argparse.ArgumentParser(description="arXiv ReAct Agent CLI")
    p.add_argument("--query",  type=str, default=None, help="Single query (non-interactive)")
    p.add_argument("--list",   action="store_true",    help="List all papers in corpus")
    p.add_argument("--notes",  action="store_true",    help="Show saved research notes")
    p.add_argument("--top_k",  type=int, default=config.DEFAULT_TOP_K)
    p.add_argument("--verbose", action="store_true",   help="Show full scratchpad")
    return p.parse_args()


def load_corpus() -> list:
    cache = config.DATA_DIR / f"chunks_{config.DEFAULT_CHUNK}.json"
    if cache.exists():
        print(f"  Loading chunks from cache…")
        return load_chunks(cache)
    meta = config.DATA_DIR / "metadata.json"
    if not meta.exists():
        print("  No corpus found. Run the data collection script first.")
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
        print("No papers. Run data collection first.")
        return
    with open(meta) as f:
        papers = json.load(f)
    print(f"\n📚  {len(papers)} papers in corpus:\n")
    for i, p in enumerate(papers, 1):
        print(f"  {i:3d}. {p['title']}")
        print(f"       {p['paper_id']}  |  {p['published'][:10]}")


def print_step(i: int, step, verbose: bool = False):
    action_fmt = {
        "search_corpus":  "\033[94m",   # blue
        "keyword_search": "\033[96m",   # cyan
        "fetch_arxiv":    "\033[93m",   # yellow
        "summarize_paper":"\033[95m",   # magenta
        "compare_papers": "\033[95m",
        "finish":         "\033[92m",   # green
    }
    reset = "\033[0m"
    color = action_fmt.get(step.action, "\033[97m")

    print(f"\n  Step {i}")
    print(f"  💭 {step.thought[:120]}")
    print(f"  {color}⚡ {step.action}{reset}({step.action_input[:80]}{'…' if len(step.action_input)>80 else ''})")
    if verbose and step.observation:
        print(f"  📋 {step.observation[:300]}{'…' if len(step.observation)>300 else ''}")


def run_query(agent: ReActAgent, query: str, verbose: bool = False):
    print(f"\n❓  {query}")
    print("─" * 65)
    print("🤖  Agent thinking…\n")

    response = agent.run(query)

    # Show scratchpad
    for i, step in enumerate(response.scratchpad, 1):
        print_step(i, step, verbose)

    # Sub-question info
    if response.sub_questions:
        print(f"\n  🗺  Decomposed into {len(response.sub_questions)} sub-questions:")
        for q in response.sub_questions:
            print(f"      • {q}")

    # Critique
    if response.critique:
        verdict = "✅ passed" if response.critique.passed else "🔄 refined"
        print(f"\n  🔍 Self-critique: {verdict}")
        if response.critique.gaps:
            for gap in response.critique.gaps:
                print(f"     Gap: {gap}")

    # Answer
    print(f"\n\n💬  Answer\n{'─'*65}")
    print(response.answer)

    # Sources
    if response.sources:
        print(f"\n📄  Sources ({len(response.sources)}):")
        for i, s in enumerate(response.sources[:6], 1):
            print(f"   [{i}] {s['title'][:72]}  (score: {s['score']:.3f})")

    print(f"\n⏱  {response.total_steps} steps · {response.latency_ms:.0f}ms")
    print("─" * 65)


def main():
    args = parse_args()

    if args.list:
        list_papers()
        return

    if args.notes:
        mem = ResearchMemory()
        notes = mem.all_notes()
        if notes:
            print(f"\n📌  {len(notes)} research notes:\n")
            for note in notes:
                src = f"  [{note.source}]" if note.source else ""
                print(f"  {note.key}{src}:\n  {note.content}\n")
        else:
            print("No research notes saved yet.")
        return

    print(BANNER)
    print(f"Model      : {config.MODEL_KEY}  ({config.EMBEDDING_MODEL})")
    print(f"Chunk size : {config.DEFAULT_CHUNK} tokens")
    print(f"Top-K      : {args.top_k}")
    print()

    print("Loading corpus…")
    chunks = load_corpus()
    print(f"  {len(chunks):,} chunks ready.\n")

    print(f"Building BGE retriever…")
    dense = Retriever.build(
        model_key=config.MODEL_KEY,
        chunks=chunks,
        chunk_size=config.DEFAULT_CHUNK,
        index_dir=config.RESULTS_DIR / "indices",
    )
    print("  BGE retriever ready.")
    print("  Building BM25 fallback…")
    bm25 = BM25Retriever(chunks)

    conv_mem = ConversationMemory()
    res_mem  = ResearchMemory()
    agent    = ReActAgent(dense, bm25, conv_mem, res_mem)
    print("  Agent ready.\n")

    # Single-query mode
    if args.query:
        run_query(agent, args.query, verbose=args.verbose)
        return

    # Interactive mode
    print("Ask anything about ML research. Type 'quit' to exit.")
    print("Commands: 'notes' — show saved notes | 'clear' — reset memory\n")
    print("─" * 65)

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
        if query.lower() == "notes":
            notes = res_mem.all_notes()
            for n in notes:
                print(f"  {n.key}: {n.content[:80]}")
            continue
        if query.lower() == "clear":
            conv_mem.clear()
            print("  Conversation memory cleared.")
            continue

        run_query(agent, query, verbose=args.verbose)


if __name__ == "__main__":
    main()
