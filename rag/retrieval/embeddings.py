"""
embeddings.py – BGE embedding model wrapper with disk caching.
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
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        batch_size: int = 32,
        cache_dir: Path = config.CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self.cache_dir  = cache_dir
        self.device     = config.DEVICE
        self.batch_size = batch_size

        hf_token = config.HF_TOKEN or None
        log.info("Loading %s on %s …", model_name, self.device)
        self._model = SentenceTransformer(model_name, device=self.device, token=hf_token)
        self.dim    = self._model.get_sentence_embedding_dimension()
        log.info("Embedding dim = %d", self.dim)

    def encode(
        self,
        texts: list[str],
        normalise: bool = True,
        show_progress: bool = False,
        cache_key: str | None = None,
    ) -> np.ndarray:
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                log.info("Cache hit: %s", cache_key)
                return cached

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalise,
        ).astype(np.float32)

        if normalise:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

        if cache_key:
            self._save_cache(cache_key, embeddings)

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        # BGE instruction prefix for queries
        q = f"Represent this sentence for searching relevant passages: {query}"
        vec = self._model.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.maximum(norms, 1e-10)

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(f"BGE_{key}".encode()).hexdigest()
        return self.cache_dir / f"BGE_{h}.pkl"

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
