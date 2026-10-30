import unittest
from src.services.detector.detector_service import DetectorService


class TestDetectorService(unittest.TestCase):
    def setUp(self):
        self.detector = DetectorService()

    def test_analyze_grounded_answer(self):
        query = "What speedup does LLM-Check offer?"
        answer = "LLM-Check demonstrates up to 45x speedup over existing baselines."
        retrieved_chunks = [
            {"text": "LLM-Check demonstrates up to 45x speedup over existing baselines without extra model calls."}
        ]

        result = self.detector.analyze_answer(query, answer, retrieved_chunks)
        self.assertEqual(result["total_sentences"], 1)
        sent = result["sentences"][0]
        self.assertEqual(sent["verdict"], "factual")
        self.assertLess(sent["risk_score"], 0.40)
        self.assertIn("signals", sent)
        self.assertIn("grounding", sent["signals"])

    def test_analyze_hallucinated_answer(self):
        query = "What is the speedup?"
        answer = "The spacecraft flew at warp speed through the Martian cosmos."
        retrieved_chunks = [
            {"text": "LLM-Check demonstrates up to 45x speedup over existing baselines without extra model calls."}
        ]

        result = self.detector.analyze_answer(query, answer, retrieved_chunks)
        self.assertEqual(result["total_sentences"], 1)
        sent = result["sentences"][0]
        self.assertEqual(sent["verdict"], "hallucinated")
        self.assertGreater(sent["risk_score"], 0.50)
        self.assertTrue(result["has_hallucination"])


if __name__ == "__main__":
    unittest.main()
