"""
vector_store.py – FAISS-backed vector store.
"""
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from rag import config
from rag.processing.chunker import Chunk

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dim: int = config.EMBEDDING_DIM) -> None:
        self.dim    = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        assert embeddings.shape[0] == len(chunks)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._chunks.extend(chunks)
        log.debug("VectorStore: %d vectors total", self._index.ntotal)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = config.DEFAULT_TOP_K,
        allowed_paper_ids: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._index.ntotal == 0:
            return []
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        
        # If filtering is active, retrieve more candidates first to ensure we get enough matches
        search_k = min(self._index.ntotal, 200 if allowed_paper_ids is not None else top_k)
        scores, indices = self._index.search(query_vec, search_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                chunk = self._chunks[idx]
                if allowed_paper_ids is None or chunk.paper_id in allowed_paper_ids:
                    results.append((chunk, float(score)))
                    if len(results) >= top_k:
                        break
        return results

    def save(self, directory: Path, name: str = "index") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / f"{name}.faiss"))
        with open(directory / f"{name}_meta.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    @classmethod
    def load(cls, directory: Path, name: str = "index") -> "VectorStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / f"{name}.faiss"))
        with open(directory / f"{name}_meta.pkl", "rb") as f:
            chunks: list[Chunk] = pickle.load(f)
        store = cls(dim=index.d)
        store._index  = index
        store._chunks = chunks
        log.info("Loaded %d vectors (dim=%d)", index.ntotal, index.d)
        return store

    @property
    def size(self) -> int:
        return self._index.ntotal
