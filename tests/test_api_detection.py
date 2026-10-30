import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.main import app


class TestDetectionAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_metrics(self):
        res = self.client.get("/api/detector/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("comparison_table", data)
        self.assertIn("weights", data)
        # Check comparison table has 4 rows (3 single signals + 1 fused)
        self.assertEqual(len(data["comparison_table"]), 4)

    def test_ask_returns_detection_payload(self):
        # Ingest minimal text into test pipeline
        self.client.app.state.pipeline.ingest_text(
            "Large Language Models (LLMs) can produce hallucinations.",
            "test_source.txt"
        )
        test_answer = "Large Language Models (LLMs) can produce hallucinations in RAG systems."
        with patch.object(self.client.app.state.pipeline, "generate", return_value=test_answer):
            res = self.client.post("/api/ask", json={
                "query": "What can LLMs produce?",
                "top_k": 2,
                "model_name": "auto",
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("answer", data)
            self.assertIn("detection", data)
            detection = data["detection"]
            self.assertIn("sentences", detection)
            self.assertIn("overall_risk", detection)
            self.assertIn("has_hallucination", detection)
            if detection["sentences"]:
                sent = detection["sentences"][0]
                self.assertIn("risk_score", sent)
                self.assertIn("signals", sent)
                self.assertIn("entropy", sent["signals"])
                self.assertIn("consistency", sent["signals"])
                self.assertIn("grounding", sent["signals"])
                self.assertIn("explanation", sent)


if __name__ == "__main__":
    unittest.main()
