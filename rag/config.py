"""
config.py – Central configuration for the arXiv Agent.
BGE is the single embedding model (benchmark winner).
"""
import os
from pathlib import Path
import torch
from dotenv import load_dotenv

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
CACHE_DIR   = BASE_DIR / "embeddings_cache"
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = BASE_DIR / "uploads"

for d in [DATA_DIR, CACHE_DIR, RESULTS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── arXiv ──────────────────────────────────────────────────────────────────────
ARXIV_CATEGORY = "cs.LG"
NUM_PAPERS     = 150
ARXIV_SORT_BY  = "submittedDate"

# ── Chunking ───────────────────────────────────────────────────────────────────
DEFAULT_CHUNK  = 512
CHUNK_OVERLAP  = 64
CHUNK_SIZES    = [256, 512, 1024]

# ── Embedding (BGE only — benchmark winner) ────────────────────────────────────
MODEL_KEY       = "BGE"
EMBEDDING_MODEL = "BAAI/bge-large-en"
EMBEDDING_DIM   = 1024

# ── FAISS ──────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K    = 5
FAISS_INDEX_TYPE = "FlatIP"

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-2.5-flash-lite"
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_TOKENS  = 4096
GEMINI_RPM         = 60

# ── Agent ──────────────────────────────────────────────────────────────────────
AGENT_MAX_STEPS        = 8
AGENT_MAX_SUBQUESTIONS = 4
MEMORY_WINDOW          = 6
RESEARCH_NOTES_PATH    = DATA_DIR / "research_notes.json"

# ── Literature Review ──────────────────────────────────────────────────────────
LIT_REVIEW_MAX_THEMES  = 6
LIT_REVIEW_MIN_THEMES  = 3
CITATION_FORMATS       = ["APA", "MLA", "Chicago", "IEEE", "Vancouver"]

# ── Secrets ────────────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "") or os.getenv("HF_Token", "")

if HF_TOKEN and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HF_TOKEN

def _best_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _best_device()
