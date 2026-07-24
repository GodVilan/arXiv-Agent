import unittest
import numpy as np
from rag.processing.chunker import Chunk
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.vector_store import VectorStore
from rag.sources.source_router import SourceRouter, SourceConfig

class MockCorpusRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k=5, allowed_paper_ids=None):
        # Filter chunks if allowed_paper_ids is provided
        filtered = self.chunks
        if allowed_paper_ids is not None:
            filtered = [c for c in filtered if c.paper_id in allowed_paper_ids]
        return [(c, 0.9 - 0.1 * i) for i, c in enumerate(filtered[:top_k])]

class MockSessionIndex:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5, allowed_paper_ids=None):
        filtered = self.chunks
        if allowed_paper_ids is not None:
            filtered = [c for c in filtered if c.paper_id in allowed_paper_ids]
        return [(c, 0.95 - 0.1 * i) for i, c in enumerate(filtered[:top_k])]

    def is_empty(self):
        return len(self.chunks) == 0

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk(
                chunk_id="paper1_0000",
                paper_id="paper1",
                title="Transformer Attention",
                authors=["Author A"],
                text="The attention mechanism is very powerful.",
                token_count=6,
                chunk_index=0,
                source="corpus"
            ),
            Chunk(
                chunk_id="paper2_0000",
                paper_id="paper2",
                title="LoRA low rank",
                authors=["Author B"],
                text="Low-Rank adaptation makes model training efficient.",
                token_count=6,
                chunk_index=0,
                source="arxiv"
            ),
            Chunk(
                chunk_id="paper3_0000",
                paper_id="paper3",
                title="Catastrophic Forgetting",
                authors=["Author C"],
                text="Continual learning models suffer from catastrophic forgetting.",
                token_count=7,
                chunk_index=0,
                source="corpus"
            )
        ]

    def test_bm25_retrieval(self):
        retriever = BM25Retriever(self.chunks)
        # Search for exact term
        results = retriever.retrieve("attention", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].chunk_id, "paper1_0000")

    def test_vector_store_add_search(self):
        store = VectorStore(dim=4)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float32)
        store.add(embeddings, self.chunks)
        
        # Exact match query
        query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query_vec, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].chunk_id, "paper1_0000")
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_source_router_merging(self):
        corpus = MockCorpusRetriever([self.chunks[0]])
        session = MockSessionIndex([self.chunks[1]])
        router = SourceRouter(corpus, session)
        
        # Test merging both sources
        results = router.search("test query", top_k=5)
        self.assertEqual(len(results), 2)
        
        # Should be sorted by score descending: session has score 0.95, corpus 0.9
        self.assertEqual(results[0][0].chunk_id, "paper2_0000") # session
        self.assertEqual(results[1][0].chunk_id, "paper1_0000") # corpus

    def test_scoping_and_source_toggles(self):
        # 1. Test BM25 filtering
        bm25 = BM25Retriever(self.chunks)
        res_bm25 = bm25.retrieve("attention mechanism adaptation", top_k=5, allowed_paper_ids={"paper1"})
        self.assertTrue(all(c.paper_id == "paper1" for c, _ in res_bm25))

        # 2. Test VectorStore filtering
        store = VectorStore(dim=4)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float32)
        store.add(embeddings, self.chunks)
        query_vec = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        res_vec = store.search(query_vec, top_k=5, allowed_paper_ids={"paper2"})
        self.assertTrue(all(c.paper_id == "paper2" for c, _ in res_vec))

        # 3. Test SourceRouter scoping
        corpus = MockCorpusRetriever([self.chunks[0], self.chunks[2]])
        session = MockSessionIndex([self.chunks[1]])
        router = SourceRouter(corpus, session)
        
        results = router.search("test query", top_k=5, allowed_paper_ids={"paper1", "paper2"})
        paper_ids = [c.paper_id for c, _ in results]
        self.assertIn("paper1", paper_ids)
        self.assertIn("paper2", paper_ids)
        self.assertNotIn("paper3", paper_ids)

if __name__ == "__main__":
    unittest.main()
