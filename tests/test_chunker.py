import unittest
from pathlib import Path
from rag.processing.chunker import clean_text, recursive_chunk, _token_count

class TestChunker(unittest.TestCase):
    def test_clean_text_hyphenation(self):
        # Clean text should merge line breaks with hyphens
        raw = "continual-\nlearning architectures"
        self.assertEqual(clean_text(raw), "continuallearning architectures")

    def test_clean_text_whitespace(self):
        # Clean text should merge consecutive spaces and linebreaks
        raw = "This   is   a \n\n\n test  ."
        self.assertEqual(clean_text(raw), "This is a \n\n test .")

    def test_clean_text_page_numbers(self):
        # Clean text should strip standalone page numbers
        raw = "Line 1\n  42  \nLine 2"
        self.assertEqual(clean_text(raw), "Line 1\n\nLine 2")

    def test_token_count(self):
        text = "This is a simple sentence with exactly eight words."
        self.assertEqual(_token_count(text), 9) # splitting by whitespace

    def test_recursive_chunk_small_text(self):
        # Text smaller than chunk size should remain as one chunk
        text = "Short text under the threshold."
        chunks = recursive_chunk(text, chunk_size=50, overlap=5)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_recursive_chunk_splitting(self):
        # Test splitting based on punctuation/spaces
        text = "Paragraph one with some text. Paragraph two with more text. Paragraph three with yet more."
        # Use very small chunk size to trigger splits
        chunks = recursive_chunk(text, chunk_size=8, overlap=2)
        self.assertTrue(len(chunks) > 1)
        # Verify the chunks cover the entire text
        rejoined = " ".join(chunks)
        self.assertIn("Paragraph one", rejoined)
        self.assertIn("Paragraph three", rejoined)

if __name__ == "__main__":
    unittest.main()
