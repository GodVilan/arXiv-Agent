import unittest
from rag.agent.citation_formatter import CitationFormatter, PaperMeta

class TestCitation(unittest.TestCase):
    def setUp(self):
        self.paper = PaperMeta(
            paper_id="2106.09685",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            authors=["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"],
            year="2021",
            url="https://arxiv.org/abs/2106.09685",
            source="arxiv",
            doi="10.48550/arXiv.2106.09685",
            venue="arXiv preprint",
            citation_count=1200
        )

    def test_apa_formatting(self):
        formatter = CitationFormatter("APA")
        cite = formatter._apa(self.paper, 1)
        # Expected components in APA: Authors init, (Year). Title. *Venue*. DOI
        self.assertIn("[1] Hu, E.J.", cite)
        self.assertIn("Y.", cite)
        self.assertIn("(2021).", cite)
        self.assertIn("LoRA: Low-Rank Adaptation of Large Language Models.", cite)
        self.assertIn("*arXiv preprint*.", cite)
        self.assertIn("https://doi.org/10.48550/arXiv.2106.09685", cite)
        self.assertIn("(Cited 1200 times)", cite)

    def test_mla_formatting(self):
        formatter = CitationFormatter("MLA")
        cite = formatter._mla(self.paper, 2)
        # Expected components in MLA: First author Last, First, et al. "Title." *Venue*, Year. DOI
        self.assertIn("[2] Hu, Edward J., et al.", cite)
        self.assertIn('"LoRA: Low-Rank Adaptation of Large Language Models."', cite)
        self.assertIn("*arXiv preprint*", cite)
        self.assertIn("2021.", cite)
        self.assertIn("https://doi.org/10.48550/arXiv.2106.09685.", cite)

    def test_ieee_formatting(self):
        formatter = CitationFormatter("IEEE")
        cite = formatter._ieee(self.paper, 3)
        # Expected components in IEEE: [3] E. J. Hu et al., "Title," *Venue*, Year, doi: DOI.
        self.assertIn('[3] E. J. Hu, Y. Shen, P. Wallis, et al., "LoRA: Low-Rank Adaptation of Large Language Models,"', cite)
        self.assertIn("*arXiv preprint*", cite)
        self.assertIn("2021", cite)
        self.assertIn("doi: 10.48550/arXiv.2106.09685.", cite)

    def test_inline_citation_apa(self):
        formatter = CitationFormatter("APA")
        inline = formatter.inline_cite(self.paper, 1)
        self.assertEqual(inline, "(Hu et al., 2021)")

    def test_inline_citation_ieee(self):
        formatter = CitationFormatter("IEEE")
        inline = formatter.inline_cite(self.paper, 3)
        self.assertEqual(inline, "[3]")

    def test_inline_citation_mla(self):
        formatter = CitationFormatter("MLA")
        inline = formatter.inline_cite(self.paper, 1)
        self.assertEqual(inline, "(Hu)")

if __name__ == "__main__":
    unittest.main()
