"""
dense.py – Semantic retrieval pipeline combining EmbeddingModel + VectorStore.

Usage:
    retriever = Retriever.build(model_key="MiniLM", chunks=chunks)
    results   = retriever.retrieve("What is self-attention?", top_k=5)
"""

import logging
from pathlib import Path

import numpy as np

from rag import config
from rag.processing.chunker import Chunk, load_chunks
from rag.retrieval.embeddings import EmbeddingModel
from rag.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)


class Retriever:
    """
    End-to-end retriever: query string → ranked list of (Chunk, score).
    """

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store    = vector_store

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        model_key: str,
        chunks: list[Chunk],
        chunk_size: int = config.DEFAULT_CHUNK,
        index_dir: Path | None = None,
        force_rebuild: bool = False,
    ) -> "Retriever":
        """
        Build (or load from disk) a Retriever for the given *model_key*.

        If *index_dir* is provided and a saved index exists there, it is
        loaded instead of re-encoding all chunks (saves time on reruns).
        """
        emb_model = EmbeddingModel(model_key)
        index_name = f"{model_key}_cs{chunk_size}"

        if index_dir:
            index_dir = Path(index_dir)
            index_file = index_dir / f"{index_name}.faiss"
            if index_file.exists() and not force_rebuild:
                log.info("Loading existing index '%s' from %s …", index_name, index_dir)
                store = VectorStore.load(index_dir, name=index_name)
                return cls(emb_model, store)

        # ── Encode all chunks ──────────────────────────────────────────────
        texts = [c.text for c in chunks]
        cache_key = f"{model_key}_cs{chunk_size}_n{len(chunks)}"
        embeddings = emb_model.encode(texts, cache_key=cache_key)

        store = VectorStore(dim=emb_model.dim)
        store.add(embeddings, chunks)

        if index_dir:
            store.save(index_dir, name=index_name)

        return cls(emb_model, store)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[tuple[Chunk, float]]:
        """
        Retrieve the *top_k* most relevant chunks for *query*.
        Returns a list of (Chunk, cosine_score) pairs.
        """
        q_vec = self.embedding_model.encode_query(query)
        return self.vector_store.search(q_vec, top_k=top_k)

    def retrieve_texts(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[str]:
        """Convenience method: return just the text strings."""
        return [chunk.text for chunk, _ in self.retrieve(query, top_k)]

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[list[tuple[Chunk, float]]]:
        """Encode multiple queries in one pass (faster for evaluation)."""
        q_vecs = np.vstack([
            self.embedding_model.encode_query(q) for q in queries
        ])
        return self.vector_store.search_batch(q_vecs, top_k=top_k)

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_context(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        max_tokens: int = 1500,
    ) -> str:
        """
        Build a formatted context string from retrieved chunks, suitable
        for passing to a language model as grounding context.
        """
        results = self.retrieve(query, top_k=top_k)
        context_parts = []
        total_tokens  = 0

        for i, (chunk, score) in enumerate(results, start=1):
            header = f"[Source {i}] {chunk.title} (score: {score:.3f})"
            body   = chunk.text.strip()
            part   = f"{header}\n{body}"
            part_tokens = len(part.split())

            if total_tokens + part_tokens > max_tokens:
                break

            context_parts.append(part)
            total_tokens += part_tokens

        return "\n\n".join(context_parts)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import json

    meta_path = config.DATA_DIR / "metadata.json"
    chunks_path = config.DATA_DIR / "chunks_512.json"

    if not chunks_path.exists():
        print("Run chunker.py first.")
    else:
        chunks = load_chunks(chunks_path)

        retriever = Retriever.build(
            model_key="MiniLM",
            chunks=chunks,
            index_dir=config.RESULTS_DIR,
        )

        query = "What is the transformer self-attention mechanism?"
        results = retriever.retrieve(query, top_k=3)
        print(f"\nQuery: {query}\n")
        for chunk, score in results:
            print(f"  [{score:.4f}] {chunk.title}")
            print(f"  {chunk.text[:200]}…\n")
