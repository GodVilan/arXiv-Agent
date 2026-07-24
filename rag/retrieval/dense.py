"""
dense.py – Dense BGE/FAISS retriever.
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
    def __init__(self, emb_model: EmbeddingModel, store: VectorStore) -> None:
        self.emb_model = emb_model
        self.store     = store

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        chunk_size: int = config.DEFAULT_CHUNK,
        index_dir: Path | None = None,
        force_rebuild: bool = False,
    ) -> "Retriever":
        emb_model  = EmbeddingModel()
        index_name = f"BGE_cs{chunk_size}"

        if index_dir:
            index_dir  = Path(index_dir)
            index_file = index_dir / f"{index_name}.faiss"
            if index_file.exists() and not force_rebuild:
                log.info("Loading index from %s", index_dir)
                store = VectorStore.load(index_dir, name=index_name)
                return cls(emb_model, store)

        texts     = [c.text for c in chunks]
        cache_key = f"cs{chunk_size}_n{len(chunks)}"
        embeddings = emb_model.encode(texts, cache_key=cache_key, show_progress=True)

        store = VectorStore(dim=emb_model.dim)
        store.add(embeddings, chunks)

        if index_dir:
            store.save(index_dir, name=index_name)

        return cls(emb_model, store)

    def retrieve(
        self, query: str, top_k: int = config.DEFAULT_TOP_K, allowed_paper_ids: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        q_vec = self.emb_model.encode_query(query)
        return self.store.search(q_vec, top_k=top_k, allowed_paper_ids=allowed_paper_ids)

    def format_context(
        self, query: str, top_k: int = config.DEFAULT_TOP_K, max_tokens: int = 2000,
    ) -> str:
        results = self.retrieve(query, top_k=top_k)
        parts, total = [], 0
        for i, (chunk, score) in enumerate(results, 1):
            part = f"[Source {i}] {chunk.title} (score: {score:.3f})\n{chunk.text.strip()}"
            tokens = len(part.split())
            if total + tokens > max_tokens:
                break
            parts.append(part)
            total += tokens
        return "\n\n".join(parts)
