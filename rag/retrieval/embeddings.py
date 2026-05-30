"""
embeddings.py – Wrapper around sentence-transformers for all three models.

Features:
  • Lazy loading (model downloaded on first call)
  • L2 normalisation (enables cosine similarity via inner product in FAISS)
  • Batch encoding with progress bar
  • Disk caching per (model_key, chunk_size)

Usage:
    from embeddings import EmbeddingModel
    model = EmbeddingModel("MiniLM")
    vecs  = model.encode(["hello world", "machine learning"])
"""

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag import config

log = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer that adds caching and
    automatic L2-normalisation.
    """

    def __init__(
        self,
        model_key: str,                      # one of config.EMBEDDING_MODELS keys
        batch_size: int = 64,
        device: str | None = None,
        cache_dir: Path = config.CACHE_DIR,
    ) -> None:
        if model_key not in config.EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Choose from {list(config.EMBEDDING_MODELS)}"
            )

        self.model_key  = model_key
        self.model_name = config.EMBEDDING_MODELS[model_key]
        self.batch_size = batch_size
        self.cache_dir  = cache_dir
        self.device     = device or config.DEVICE
        self.batch_size = batch_size if self.device != "mps" else min(batch_size, 32)

        log.info("Loading embedding model '%s' (%s) on %s …",
                 model_key, self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dim    = self._model.get_sentence_embedding_dimension()
        log.info("  → embedding dim = %d", self.dim)

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        normalise: bool = True,
        show_progress: bool = True,
        cache_key: str | None = None,
    ) -> np.ndarray:
        """
        Encode *texts* into dense float32 embeddings (shape: N × dim).

        If *cache_key* is provided, embeddings are saved to / loaded from disk
        so repeated runs skip the expensive encode step.
        """
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                log.info("Cache hit for key '%s' – skipping encode.", cache_key)
                return cached

        log.info("Encoding %d texts with %s …", len(texts), self.model_key)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalise,  # handles L2 norm in-library
        ).astype(np.float32)

        if normalise:
            # Double-check: ensure unit norm (some models may not honour the flag)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

        if cache_key:
            self._save_cache(cache_key, embeddings)

        return embeddings

    def encode_query(self, query: str, normalise: bool = True) -> np.ndarray:
        """Encode a single query string → shape (1, dim)."""
        # BGE models benefit from an instruction prefix for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=normalise,
        ).astype(np.float32)
        if normalise:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            vec = vec / np.maximum(norms, 1e-10)
        return vec

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        safe_key = hashlib.md5(f"{self.model_key}_{key}".encode()).hexdigest()
        return self.cache_dir / f"{self.model_key}_{safe_key}.pkl"

    def _load_cache(self, key: str) -> np.ndarray | None:
        p = self._cache_path(key)
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, arr: np.ndarray) -> None:
        p = self._cache_path(key)
        with open(p, "wb") as f:
            pickle.dump(arr, f)
        log.info("Cached embeddings to %s", p)


# ──────────────────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = ["What is attention mechanism?", "How does BERT work?"]
    for key in config.EMBEDDING_MODELS:
        m = EmbeddingModel(key)
        v = m.encode(sample, show_progress=False)
        print(f"{key}: shape={v.shape}, norm={np.linalg.norm(v[0]):.4f}")
