import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.main import app
from src.services.benchmark_service import (
    _generate_source_benchmark_samples,
    create_synthetic_data,
    download_halueval_subset,
)

SAMPLE_DOC_TEXT = """
Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external knowledge base.
This grounds large language models on the most accurate, up-to-date information.
Hallucination detection aims to identify when generated answers deviate from reference knowledge.
Researchers have proposed token entropy and consistency checks to mitigate factual errors.
Selective mitigation rewrites only the hallucinated sentences instead of the entire passage.
This approach significantly reduces latency overhead while improving factual consistency.
Experiments demonstrate that multi-signal fusion outperforms individual detection signals.
Synthetic evaluation benchmarks provide reproducible metrics for measuring hallucination rates.
"""


class TestBenchmarkSampleGeneration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_source_samples_exact_counts_fallback(self):
        """Verify that _generate_source_benchmark_samples generates the exact count requested (10, 20, 50, 100)."""
        # Patch LLM to raise an exception, testing the robust fallback mechanism
        with patch("src.services.benchmark_service.generate_with_gemini", side_effect=Exception("API offline")), \
             patch("src.services.benchmark_service.generate_with_ollama", side_effect=Exception("Ollama offline")), \
             patch("src.services.benchmark_service.OLLAMA_AVAILABLE", False):
            for count in [10, 20, 50, 100]:
                samples = _generate_source_benchmark_samples("test_doc.pdf", SAMPLE_DOC_TEXT, count)
                self.assertEqual(
                    len(samples),
                    count,
                    f"Expected exactly {count} samples, got {len(samples)}"
                )
                # Verify structure
                for s in samples:
                    self.assertIn("question", s)
                    self.assertIn("reference_knowledge", s)
                    self.assertIn("category", s)
                    self.assertTrue(len(s["question"]) > 0)

    def test_source_samples_tops_up_undergenerated_llm_response(self):
        """If LLM returns only 3 samples when 20 are requested, verify it is topped up to exactly 20."""
        partial_llm_json = """{
            "samples": [
                {"question": "What is RAG?", "reference_knowledge": "RAG is an AI framework", "category": "easy_factual"},
                {"question": "What is hallucination detection?", "reference_knowledge": "Hallucination detection aims to identify errors", "category": "easy_factual"},
                {"question": "What does selective mitigation do?", "reference_knowledge": "Selective mitigation rewrites only hallucinated sentences", "category": "precise_extraction"}
            ]
        }"""

        with patch("src.services.benchmark_service.generate_with_gemini", return_value=partial_llm_json), \
             patch("src.services.benchmark_service.generate_with_ollama", return_value=partial_llm_json), \
             patch("src.services.benchmark_service.OLLAMA_AVAILABLE", False):
            samples = _generate_source_benchmark_samples("test_doc.pdf", SAMPLE_DOC_TEXT, num_samples=20)
            self.assertEqual(len(samples), 20, f"Expected 20 samples, got {len(samples)}")
            # First 3 should be from LLM
            self.assertEqual(samples[0]["generation_method"], "llm")
            self.assertEqual(samples[1]["generation_method"], "llm")
            self.assertEqual(samples[2]["generation_method"], "llm")
            # The remaining 17 should be topped up with fallback templates
            for s in samples[3:]:
                self.assertEqual(s["generation_method"], "template_fallback")

    def test_synthetic_data_exact_count(self):
        """Verify synthetic data generator returns the exact count for any number."""
        for count in [10, 20, 50]:
            data = create_synthetic_data(count)
            self.assertEqual(len(data), count)

    def test_api_generate_samples_respects_num_samples(self):
        """Verify the /api/benchmark/generate_samples endpoint returns the exact requested sample count.

        LLM calls are mocked so this test runs offline without touching the Gemini API.
        """
        # Build a minimal valid LLM response JSON that the benchmark service can parse
        def _mock_llm_response(*args, **kwargs):
            # Return just 3 samples — the service must top up to the requested count
            return """{
                "samples": [
                    {"question": "What is RAG?", "reference_knowledge": "RAG is an AI framework", "category": "easy_factual"},
                    {"question": "What is hallucination?", "reference_knowledge": "Hallucination detection identifies errors", "category": "easy_factual"},
                    {"question": "What does fusion do?", "reference_knowledge": "Fusion combines multiple signals", "category": "precise_extraction"}
                ]
            }"""

        with patch("src.api.routers.benchmark._load_source_text", return_value=SAMPLE_DOC_TEXT), \
             patch("src.services.benchmark_service.generate_with_gemini", side_effect=_mock_llm_response), \
             patch("src.services.benchmark_service.generate_with_ollama", side_effect=_mock_llm_response), \
             patch("src.services.benchmark_service.OLLAMA_AVAILABLE", False):
            for count in [10, 20, 50]:
                response = self.client.post(
                    "/api/benchmark/generate_samples",
                    data={"num_samples": count, "source_filter": "test_doc.pdf"}
                )
                self.assertEqual(response.status_code, 200)
                json_data = response.json()
                self.assertIn("samples", json_data)
                self.assertEqual(
                    len(json_data["samples"]),
                    count,
                    f"API should return {count} samples for num_samples={count}, got {len(json_data['samples'])}"
                )


    def test_download_halueval_subset_exact_count(self):
        """Verify download_halueval_subset returns exact count even when network fails."""
        with patch("src.services.benchmark_service.requests.get", side_effect=Exception("Network error")):
            for count in [10, 20, 50]:
                samples = download_halueval_subset(count)
                self.assertEqual(len(samples), count)


if __name__ == "__main__":
    unittest.main()
