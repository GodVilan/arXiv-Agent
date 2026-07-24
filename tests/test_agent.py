import unittest
from rag.agent.react_agent import ReActAgent

class TestAgent(unittest.TestCase):
    def test_query_normalization(self):
        # Query normalisation should strip punctuation, lowercase, and sort words
        q1 = "attention mechanism transformer!"
        q2 = "Transformer, attention mechanism?"
        self.assertEqual(
            ReActAgent._normalise_query(q1),
            ReActAgent._normalise_query(q2)
        )

    def test_loop_hint_builder(self):
        # Should build appropriate hints for loops and remaining steps
        used_queries = ["continual learning", "catastrophic forgetting"]
        top_hits = {"L2P: Learning to Prompt": 2} # repeated result
        
        hint = ReActAgent._build_loop_hint(
            used_queries=used_queries,
            top_hits=top_hits,
            consecutive_same=2,
            step_num=6
        )
        
        self.assertIn("continual learning", hint)
        self.assertIn("These results keep appearing", hint)
        self.assertIn("You are looping", hint)
        self.assertIn("steps remaining", hint)

    def test_extract_top_result_title(self):
        # Should extract first title from formatted tool output
        obs = "[1] **Attention Is All You Need** [CORPUS] (score: 0.95)\nThis is abstract text..."
        title = ReActAgent._extract_top_result_title(obs)
        self.assertEqual(title, "Attention Is All You Need")

if __name__ == "__main__":
    unittest.main()
