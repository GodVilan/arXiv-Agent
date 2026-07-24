"""
session_index.py – Dynamic in-memory FAISS index for the current session.

Holds papers from live arXiv fetches + user uploads.
Lives in Streamlit session_state so it persists across reruns.
"""
import logging
import sqlite3
import json
from pathlib import Path

import numpy as np

from rag.processing.chunker import Chunk
from rag.retrieval.embeddings import EmbeddingModel
from rag.retrieval.vector_store import VectorStore
from rag import config

log = logging.getLogger(__name__)


class SessionIndex:
    """
    A dynamic, SQLite-backed persistent FAISS index populated at runtime from any paper source.
    Grows as users upload or fetch papers, and survives app restarts.
    """

    def __init__(self, emb_model: EmbeddingModel) -> None:
        self._emb             = emb_model
        self._added_ids:  set[str]  = set()
        self._paper_meta: dict[str, dict] = {}
        
        # SQLite storage setup
        self._db_path = config.DATA_DIR / "session_papers.db"
        self._init_sqlite()
        
        # FAISS directory and file name
        self._index_dir = config.RESULTS_DIR / "indices"
        self._index_name = "session_index"
        
        # Load existing index if it exists on disk
        index_file = self._index_dir / f"{self._index_name}.faiss"
        if index_file.exists():
            try:
                self._store = VectorStore.load(self._index_dir, name=self._index_name)
                for chunk in self._store._chunks:
                    self._added_ids.add(chunk.chunk_id)
                log.info("Loaded persistent FAISS session index with %d chunks", self._store.size)
            except Exception as e:
                log.error("Failed to load persistent FAISS index: %s. Re-creating.", e)
                self._store = VectorStore(dim=emb_model.dim)
        else:
            self._store = VectorStore(dim=emb_model.dim)
            
        # Load SQLite paper metadata
        self._load_sqlite_metadata()

    def _init_sqlite(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    abstract TEXT,
                    published TEXT,
                    url TEXT,
                    pdf_url TEXT,
                    source TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            log.error("Failed to initialize SQLite for SessionIndex: %s", e)

    def _load_sqlite_metadata(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT paper_id, title, authors, abstract, published, url, pdf_url, source FROM papers")
            rows = cursor.fetchall()
            for row in rows:
                pid = row[0]
                self._paper_meta[pid] = {
                    "paper_id": pid,
                    "title": row[1],
                    "authors": json.loads(row[2]) if row[2] else [],
                    "abstract": row[3],
                    "published": row[4],
                    "url": row[5],
                    "pdf_url": row[6],
                    "source": row[7]
                }
            conn.close()
            log.info("Loaded %d paper metadata rows from SQLite store", len(self._paper_meta))
        except Exception as e:
            log.error("Failed to load SQLite paper metadata: %s", e)

    def _save_paper_sqlite(self, metadata: dict) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            pid = metadata.get("paper_id", metadata.get("title", "unknown"))
            authors_json = json.dumps(metadata.get("authors", []))
            cursor.execute("""
                INSERT OR REPLACE INTO papers (paper_id, title, authors, abstract, published, url, pdf_url, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pid,
                metadata.get("title", "Untitled"),
                authors_json,
                metadata.get("abstract", ""),
                metadata.get("published", ""),
                metadata.get("url", ""),
                metadata.get("pdf_url", ""),
                metadata.get("source", "upload")
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error("Failed to save paper to SQLite: %s", e)

    # ── Add papers ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], metadata: dict) -> int:
        """
        Embed and add chunks to the index.
        Skips chunks whose chunk_id already exists.
        Persists chunks to FAISS on disk and paper metadata to SQLite.
        Returns number of chunks actually added.
        """
        new_chunks = [c for c in chunks if c.chunk_id not in self._added_ids]
        if not new_chunks:
            log.info("All chunks for '%s' already indexed.", metadata.get("title", "?"))
            return 0

        texts      = [c.text for c in new_chunks]
        embeddings = self._emb.encode(texts, show_progress=False)
        self._store.add(embeddings, new_chunks)

        for c in new_chunks:
            self._added_ids.add(c.chunk_id)

        pid = metadata.get("paper_id", metadata.get("title", "unknown"))
        metadata["paper_id"] = pid
        self._paper_meta[pid] = metadata
        
        # Save to SQLite
        self._save_paper_sqlite(metadata)
        
        # Save FAISS index
        try:
            self._store.save(self._index_dir, name=self._index_name)
            log.info("Saved FAISS session index on disk with %d chunks total", self._store.size)
        except Exception as e:
            log.error("Failed to save FAISS session index: %s", e)

        log.info("Added %d chunks for '%s'", len(new_chunks), metadata.get("title", pid))
        return len(new_chunks)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int = config.DEFAULT_TOP_K, allowed_paper_ids: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._store.size == 0:
            return []
        q_vec = self._emb.encode_query(query)
        return self._store.search(q_vec, top_k=top_k, allowed_paper_ids=allowed_paper_ids)

    # ── Info ───────────────────────────────────────────────────────────────────

    @property
    def paper_count(self) -> int:
        return len(self._paper_meta)

    @property
    def chunk_count(self) -> int:
        return self._store.size

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self._paper_meta

    def list_papers(self) -> list[dict]:
        return list(self._paper_meta.values())

    def is_empty(self) -> bool:
        return self._store.size == 0
