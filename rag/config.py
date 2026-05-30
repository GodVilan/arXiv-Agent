"""
config.py – Central configuration for the arXiv Agent.

Benchmark-driven decisions:
  - BGE is the single embedding model (MRR@5=0.990, AR=0.912 — best overall).
  - BM25 is retained as a silent hybrid fallback for keyword-heavy queries.
  - Chunk size 512 balances precision and recall.
"""

import os
from pathlib import Path
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = BASE_DIR / "embeddings_cache"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── arXiv collection ───────────────────────────────────────────────────────────
ARXIV_CATEGORY = "cs.LG"
NUM_PAPERS     = 150
ARXIV_SORT_BY  = "submittedDate"

# ── Chunking ───────────────────────────────────────────────────────────────────
# Benchmark winner: 512 tokens with 64 overlap
DEFAULT_CHUNK  = 512
CHUNK_OVERLAP  = 64
CHUNK_SIZES    = [256, 512, 1024]   # kept for evaluation scripts

# ── Embedding — BGE only ───────────────────────────────────────────────────────
# Decision: benchmarked MiniLM / MPNet / BGE; BGE won on every generation metric.
MODEL_KEY      = "BGE"
EMBEDDING_MODEL = "BAAI/bge-large-en"   # 1024-dim

# Kept for backward-compat with run_experiments.py
EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet":  "sentence-transformers/all-mpnet-base-v2",
    "BGE":    "BAAI/bge-large-en",
}

# ── FAISS ──────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K    = 5
TOP_K_VALUES     = [3, 5, 10]
FAISS_INDEX_TYPE = "FlatIP"

# ── Generation — Gemini ────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-2.5-flash-lite"
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_TOKENS  = 2048
GEMINI_RPM         = 60    # paid tier; set to 12 on free tier

# ── Agent settings ─────────────────────────────────────────────────────────────
AGENT_MAX_STEPS         = 8     # max ReAct iterations per query
AGENT_MAX_SUBQUESTIONS  = 4     # max sub-questions from planner
MEMORY_WINDOW           = 6     # conversation turns to keep in context
RESEARCH_NOTES_PATH     = DATA_DIR / "research_notes.json"

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_K_VALUES            = [1, 3, 5, 10]
RETRIEVAL_EVAL_SAMPLES   = 100
GENERATION_EVAL_SAMPLES  = 30

# ── Secrets ────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def _best_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()
