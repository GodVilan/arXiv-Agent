"""
bm25.py – BM25 sparse retrieval (keyword fallback).
"""
import logging
import time

from rank_bm25 import BM25Okapi

from rag import config
from rag.processing.chunker import Chunk

log = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        t0 = time.time()
        self._chunks = chunks
        tokenized    = [c.text.lower().split() for c in chunks]
        self._bm25   = BM25Okapi(tokenized)
        self.build_time = round(time.time() - t0, 3)
        log.info("BM25 built in %.3fs (%d docs)", self.build_time, len(chunks))

    def retrieve(
        self, query: str, top_k: int = config.DEFAULT_TOP_K, allowed_paper_ids: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        tokens  = query.lower().split()
        scores  = self._bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        for i in indices:
            if scores[i] <= 0:
                continue
            chunk = self._chunks[i]
            if allowed_paper_ids is None or chunk.paper_id in allowed_paper_ids:
                results.append((chunk, float(scores[i])))
                if len(results) >= top_k:
                    break
        return results

    def format_context(
        self, query: str, top_k: int = config.DEFAULT_TOP_K, max_tokens: int = 2000,
    ) -> str:
        results = self.retrieve(query, top_k=top_k)
        parts, total = [], 0
        for i, (chunk, score) in enumerate(results, 1):
            part = f"[Source {i}] {chunk.title} (bm25: {score:.3f})\n{chunk.text.strip()}"
            tokens = len(part.split())
            if total + tokens > max_tokens:
                break
            parts.append(part)
            total += tokens
        return "\n\n".join(parts)
