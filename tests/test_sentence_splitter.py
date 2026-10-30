import unittest
from src.services.detector.sentence_splitter import split_into_sentences


class TestSentenceSplitter(unittest.TestCase):
    def test_basic_sentences(self):
        text = "Large Language Models are powerful. They can hallucinate."
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0], "Large Language Models are powerful.")
        self.assertEqual(sents[1], "They can hallucinate.")

    def test_abbreviations_not_split(self):
        text = "Dr. Smith et al. published the paper on e.g. hallucination in LLMs. The results were impressive."
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertIn("Dr. Smith et al.", sents[0])
        self.assertEqual(sents[1], "The results were impressive.")

    def test_bullet_points(self):
        text = "- First claim about LLM.\n- Second claim about RAG.\n- Third claim."
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 3)

    def test_empty_and_whitespace(self):
        self.assertEqual(split_into_sentences(""), [])
        self.assertEqual(split_into_sentences("   \n\t  "), [])


if __name__ == "__main__":
    unittest.main()
