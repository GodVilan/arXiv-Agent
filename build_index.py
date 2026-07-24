"""
build_index.py – One-time setup: chunks → BGE embeddings → FAISS index.

Run this once after downloading papers. Takes ~15-20 min on CPU, ~5 min on MPS/GPU.
Progress is saved at each step so you can resume if interrupted.

Usage:
    python build_index.py
"""
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    from rag import config
    from rag.processing.chunker import process_papers, save_chunks, load_chunks
    from rag.retrieval.dense import Retriever
    from rag.retrieval.bm25 import BM25Retriever

    print("\n" + "═" * 60)
    print("  arXiv Agent — Index Builder")
    print("═" * 60)

    # ── Check metadata ─────────────────────────────────────────────
    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        print("\n❌  data/metadata.json not found.")
        print("   Run the collector first:")
        print('   python -c "from rag.data.collector import download_papers; download_papers()"')
        return

    with open(meta_path) as f:
        papers = json.load(f)

    # Only include papers whose PDF actually exists
    valid = [p for p in papers if Path(p.get("pdf_path", "")).exists()]
    missing = len(papers) - len(valid)
    print(f"\n📚  Found {len(valid)} papers with PDFs", end="")
    if missing:
        print(f"  ({missing} skipped — PDF not found)", end="")
    print()

    if not valid:
        print("❌  No valid PDFs found. Check your data/ folder.")
        return

    # ── Step 1: Chunking ───────────────────────────────────────────
    chunk_size  = config.DEFAULT_CHUNK   # 512
    chunks_path = config.DATA_DIR / f"chunks_{chunk_size}.json"

    if chunks_path.exists():
        print(f"\n✅  Step 1/3 — Chunks already exist ({chunks_path.name}), loading…")
        chunks = load_chunks(chunks_path)
        print(f"   {len(chunks):,} chunks loaded.")
    else:
        print(f"\n🔧  Step 1/3 — Chunking {len(valid)} papers (chunk_size={chunk_size})…")
        t0     = time.time()
        chunks = process_papers(valid, chunk_size=chunk_size)
        save_chunks(chunks, chunks_path)
        elapsed = round(time.time() - t0, 1)
        print(f"   ✅  {len(chunks):,} chunks saved to {chunks_path.name}  ({elapsed}s)")

    # ── Step 2: BGE embeddings + FAISS index ───────────────────────
    index_dir  = config.RESULTS_DIR / "indices"
    index_file = index_dir / f"BGE_cs{chunk_size}.faiss"

    if index_file.exists():
        print(f"\n✅  Step 2/3 — BGE index already exists ({index_file.name}), skipping.")
        print("   (Delete results/indices/BGE_cs512.faiss to force rebuild)")
    else:
        print(f"\n🔧  Step 2/3 — Building BGE embeddings + FAISS index…")
        print(f"   Device : {config.DEVICE}")
        print(f"   Model  : {config.EMBEDDING_MODEL}")
        print(f"   Chunks : {len(chunks):,}")
        print(f"   This takes ~15-20 min on CPU, ~5 min on MPS/GPU.\n")
        t0 = time.time()
        Retriever.build(
            chunks     = chunks,
            chunk_size = chunk_size,
            index_dir  = index_dir,
        )
        elapsed = round((time.time() - t0) / 60, 1)
        print(f"\n   ✅  BGE index saved to {index_dir}  ({elapsed} min)")

    # ── Step 3: BM25 smoke-test ────────────────────────────────────
    print(f"\n🔧  Step 3/3 — Smoke-testing BM25 retriever…")
    bm25    = BM25Retriever(chunks)
    results = bm25.retrieve("transformer attention mechanism", top_k=3)
    print(f"   ✅  BM25 ready. Sample result: '{results[0][0].title[:55]}…'" if results else "   ✅  BM25 ready.")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅  Setup complete! Run the app with:")
    print("      streamlit run app.py")
    print("  Or use the CLI:")
    print("      python main.py")
    print("      python main.py --query 'What is LoRA?'")
    print("      python main.py --review 'Continual Learning' --format APA")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()