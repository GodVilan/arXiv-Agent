"""
vector_store.py – FAISS-backed vector database for document chunks.

Supports:
  • IndexFlatIP  (exact cosine search after L2 normalisation)
  • IndexIVFFlat (approximate, faster for large corpora)
  • Save / load to disk
  • Metadata lookup by FAISS integer ID

Usage:
    store = VectorStore(dim=384)
    store.add(embeddings, chunks)
    results = store.search(query_vec, top_k=5)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import faiss

from rag import config
from rag.processing.chunker import Chunk

log = logging.getLogger(__name__)


class VectorStore:
    """
    Wraps a FAISS index and maps integer IDs → Chunk metadata.

    All embeddings are assumed to be L2-normalised (unit vectors)
    so inner-product search is equivalent to cosine similarity.
    """

    def __init__(self, dim: int, index_type: str = config.FAISS_INDEX_TYPE) -> None:
        self.dim        = dim
        self.index_type = index_type
        self._index     = self._build_index(dim, index_type)
        self._id_map: list[Chunk] = []   # position i ↔ FAISS id i

    # ── Index construction ────────────────────────────────────────────────────

    @staticmethod
    def _build_index(dim: int, index_type: str) -> faiss.Index:
        if index_type == "FlatIP":
            return faiss.IndexFlatIP(dim)
        elif index_type == "FlatL2":
            return faiss.IndexFlatL2(dim)
        elif index_type.startswith("IVF"):
            # IVFFlat with 100 clusters – good for >50k vectors
            nlist = 100
            quantiser = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            return index
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    # ── Add vectors ───────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """
        Add *embeddings* (shape N×dim, float32) with associated *chunks*.
        Embeddings must be L2-normalised.
        """
        assert embeddings.shape[0] == len(chunks), "Mismatch: embeddings vs chunks count"
        assert embeddings.shape[1] == self.dim,    "Mismatch: embedding dimension"

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # IVF indexes need training before first add
        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            log.info("Training IVF index on %d vectors …", len(embeddings))
            self._index.train(embeddings)

        self._index.add(embeddings)
        self._id_map.extend(chunks)
        log.info("VectorStore: %d vectors total", self._index.ntotal)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,          # shape (1, dim)
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[tuple[Chunk, float]]:
        """
        Return (chunk, score) pairs sorted by descending similarity.
        Score is cosine similarity ∈ [-1, 1].
        """
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:          # FAISS returns -1 for empty slots
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def search_batch(
        self,
        query_vecs: np.ndarray,    # shape (Q, dim)
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[list[tuple[Chunk, float]]]:
        """Batch version of search for evaluation."""
        query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vecs, k)

        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for idx, score in zip(q_indices, q_scores):
                if idx != -1:
                    results.append((self._id_map[idx], float(score)))
            all_results.append(results)
        return all_results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path, name: str = "index") -> None:
        """Save FAISS index + metadata to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / f"{name}.faiss"))
        with open(directory / f"{name}_meta.pkl", "wb") as f:
            pickle.dump(self._id_map, f)
        log.info("Saved VectorStore to %s/%s.*", directory, name)

    @classmethod
    def load(cls, directory: Path, name: str = "index") -> "VectorStore":
        """Load a previously saved VectorStore from *directory*."""
        directory = Path(directory)
        index = faiss.read_index(str(directory / f"{name}.faiss"))
        with open(directory / f"{name}_meta.pkl", "rb") as f:
            id_map: list[Chunk] = pickle.load(f)

        dim = index.d
        store = cls(dim=dim)
        store._index  = index
        store._id_map = id_map
        log.info("Loaded VectorStore: %d vectors, dim=%d", index.ntotal, dim)
        return store

    @property
    def size(self) -> int:
        return self._index.ntotal


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import numpy as np

    dim = 8
    store = VectorStore(dim=dim)

    # Dummy chunks + random unit-norm embeddings
    dummy_chunks = [
        Chunk(f"p_{i}_0000", f"p_{i}", f"Paper {i}", f"Text about topic {i}.", 5, 0)
        for i in range(10)
    ]
    vecs = np.random.randn(10, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    store.add(vecs, dummy_chunks)

    q = np.random.randn(1, dim).astype(np.float32)
    q /= np.linalg.norm(q)
    results = store.search(q, top_k=3)
    for chunk, score in results:
        print(f"  {chunk.chunk_id}  score={score:.4f}  text='{chunk.text}'")
