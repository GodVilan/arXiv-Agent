"""
source_router.py – Unified search across corpus, live arXiv, and user uploads.

Merges results from the static corpus retriever and the dynamic session index,
deduplicates by paper_id, and returns a ranked combined list.
"""
import logging
from dataclasses import dataclass

from rag import config
from rag.processing.chunker import Chunk

log = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    use_corpus:  bool = True
    use_session: bool = True   # live arXiv fetched + user uploads


class SourceRouter:
    """
    Routes search queries to one or more sources and merges the results.
    """

    def __init__(self, corpus_retriever, session_index) -> None:
        self._corpus  = corpus_retriever
        self._session = session_index

    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        source_cfg: SourceConfig | None = None,
        allowed_paper_ids: set[str] | None = None,
        section_type: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        cfg = source_cfg or SourceConfig()
        all_results: list[tuple[Chunk, float]] = []

        # Retrieve a larger pool if filtering by section to ensure we don't starve results
        fetch_k = top_k * 4 if section_type else top_k

        if cfg.use_corpus and self._corpus is not None:
            corpus_results = self._corpus.retrieve(query, top_k=fetch_k, allowed_paper_ids=allowed_paper_ids)
            all_results.extend(corpus_results)

        if cfg.use_session and self._session is not None and not self._session.is_empty():
            session_results = self._session.search(query, top_k=fetch_k, allowed_paper_ids=allowed_paper_ids)
            all_results.extend(session_results)

        # Deduplicate: keep best score per chunk_id
        seen: dict[str, tuple[Chunk, float]] = {}
        for chunk, score in all_results:
            if chunk.chunk_id not in seen or score > seen[chunk.chunk_id][1]:
                seen[chunk.chunk_id] = (chunk, score)

        merged = sorted(seen.values(), key=lambda x: -x[1])

        # Filter by section_type if specified
        if section_type:
            merged = [item for item in merged if getattr(item[0], "section_type", "general") == section_type]

        return merged[:top_k]

    def format_context(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        source_cfg: SourceConfig | None = None,
        max_tokens: int = 2000,
    ) -> str:
        results = self.search(query, top_k=top_k, source_cfg=source_cfg)
        parts, total = [], 0
        for i, (chunk, score) in enumerate(results, 1):
            src_tag = f"[{chunk.source.upper()}]"
            part    = f"[Source {i}] {src_tag} {chunk.title} (score: {score:.3f})\n{chunk.text.strip()}"
            tokens  = len(part.split())
            if total + tokens > max_tokens:
                break
            parts.append(part)
            total += tokens
        return "\n\n".join(parts)
