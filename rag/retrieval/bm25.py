"""
bm25.py – Sparse BM25 baseline retriever.

Wraps rank-bm25 to match the exact same interface as Retriever,
so it drops into run_experiments.py with zero other changes.
"""

import logging
import math
import time

from rank_bm25 import BM25Okapi

from rag import config
from rag.processing.chunker import Chunk

log = logging.getLogger(__name__)


class BM25Retriever:
    """
    Sparse BM25 retriever with the same .retrieve() interface as Retriever.
    Used as the baseline to compare against dense embedding models.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        log.info("Building BM25 index over %d chunks …", len(chunks))
        t0 = time.time()

        # Tokenise: lowercase + whitespace split (matches our naive token count)
        tokenised = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenised)

        self.build_time = round(time.time() - t0, 2)
        log.info("BM25 index built in %.2fs", self.build_time)

    def retrieve(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[tuple[Chunk, float]]:
        """
        Returns (Chunk, normalised_score) pairs — same signature as Retriever.retrieve().
        Scores are normalised to [0, 1] so they're comparable to cosine similarities.
        """
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Normalise scores to [0, 1] using log1p scaling
        max_score = scores[top_indices[0]] if top_indices else 1.0
        if max_score == 0:
            max_score = 1.0

        results = []
        for idx in top_indices:
            raw   = float(scores[idx])
            norm  = math.log1p(raw) / math.log1p(max_score) if max_score > 0 else 0.0
            results.append((self.chunks[idx], round(norm, 4)))

        return results

    def retrieve_texts(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> list[str]:
        return [chunk.text for chunk, _ in self.retrieve(query, top_k)]

    def format_context(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        max_tokens: int = 1500,
    ) -> str:
        results      = self.retrieve(query, top_k=top_k)
        context_parts = []
        total_tokens  = 0
        for i, (chunk, score) in enumerate(results, start=1):
            part        = f"[Source {i}] {chunk.title} (score: {score:.3f})\n{chunk.text.strip()}"
            part_tokens = len(part.split())
            if total_tokens + part_tokens > max_tokens:
                break
            context_parts.append(part)
            total_tokens += part_tokens
        return "\n\n".join(context_parts)