import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient

from src.main import app
from src.services.detector.sentence_splitter import split_into_sentences
from src.services.detector.signals.entropy import compute_sentence_entropy
from src.services.detector.signals.consistency import compute_sentence_consistency, compute_consistency_risk
from src.services.detector.signals.grounding import compute_sentence_grounding, compute_grounding_risk
from src.services.detector.fusion import FusionEngine, train_and_evaluate_fusion
from src.services.detector.detector_service import DetectorService


class TestPhase2EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.detector = DetectorService()

    # ==========================================
    # 1. SENTENCE SPLITTER TESTS
    # ==========================================
    def test_sentence_splitter_boundary_accuracy(self):
        """Verify sentences are cleanly split without cutting abbreviations or decimals."""
        text = "Dr. Smith developed RAG in 2020. The accuracy improved by 14.5% vs. baseline. What next? Nothing."
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 4)
        self.assertIn("Dr. Smith developed RAG in 2020.", sents[0])
        self.assertIn("14.5%", sents[1])
        self.assertIn("vs.", sents[1])

    def test_sentence_splitter_bullet_points(self):
        """Verify bullet points are split as independent segments."""
        text = "- First claim about LLMs.\n- Second claim about vector search.\n* Third claim."
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 3)

    # ==========================================
    # 2. SIGNAL 1: TOKEN ENTROPY TESTS
    # ==========================================
    def test_token_entropy_hedged_vs_assertive(self):
        """Hedged language must produce higher uncertainty than direct assertions."""
        assertive = "Large Language Models use Transformer self-attention architectures."
        hedged = "I believe perhaps Large Language Models might maybe use something like attention, possibly."
        score_assertive = compute_sentence_entropy(assertive)
        score_hedged = compute_sentence_entropy(hedged)

        self.assertGreater(score_hedged, score_assertive)
        self.assertGreaterEqual(score_assertive, 0.0)
        self.assertLessEqual(score_hedged, 1.0)

    # ==========================================
    # 3. SIGNAL 2: SAMPLING CONSISTENCY TESTS
    # ==========================================
    def test_sampling_consistency_agreement_vs_contradiction(self):
        """Consistent generations have low risk, diverging generations have high risk."""
        sent = "FAISS is an efficient library for dense vector similarity search."
        consistent_samples = [
            "FAISS provides fast similarity search for dense vectors.",
            "Dense vector search is efficiently handled by Facebook AI Similarity Search (FAISS)."
        ]
        diverging_samples = [
            "FAISS is an optical telescope developed for astronomical imaging.",
            "Vectors are biological carriers of disease."
        ]
        c_agree = compute_sentence_consistency(sent, consistent_samples)
        c_diverge = compute_sentence_consistency(sent, diverging_samples)

        risk_agree = compute_consistency_risk(c_agree)
        risk_diverge = compute_consistency_risk(c_diverge)

        self.assertGreater(c_agree, c_diverge)
        self.assertLess(risk_agree, risk_diverge)

    # ==========================================
    # 4. SIGNAL 3: RETRIEVAL GROUNDING TESTS
    # ==========================================
    def test_retrieval_grounding_supported_vs_unsupported(self):
        """Supported statements must yield high grounding, fabricated statements yield low grounding."""
        chunks = [
            "Retrieval-Augmented Generation (RAG) grounds language models on retrieved context passages.",
            "Multi-signal fusion combines token entropy, consistency, and retrieval grounding."
        ]
        supported = "Multi-signal fusion combines token entropy and retrieval grounding."
        fabricated = "Quantum teleportation was discovered on Mars by alien spacecraft in 1842."

        g_supp = compute_sentence_grounding(supported, retrieved_chunks=chunks)
        g_fab = compute_sentence_grounding(fabricated, retrieved_chunks=chunks)

        risk_supp = compute_grounding_risk(g_supp)
        risk_fab = compute_grounding_risk(g_fab)

        self.assertGreater(g_supp, g_fab)
        self.assertLess(risk_supp, risk_fab)
        self.assertGreater(risk_fab, 0.65)

    # ==========================================
    # 5. FUSION ENGINE & CLASSIFIER TESTS
    # ==========================================
    def test_fusion_engine_predictions_and_monotonicity(self):
        """Higher individual risks must produce higher fused risk."""
        engine = FusionEngine()
        low_risk = engine.predict_risk(entropy_risk=0.1, consistency_risk=0.1, grounding_risk=0.1)
        high_risk = engine.predict_risk(entropy_risk=0.9, consistency_risk=0.9, grounding_risk=0.9)
        mixed_risk = engine.predict_risk(entropy_risk=0.5, consistency_risk=0.5, grounding_risk=0.5)

        self.assertLess(low_risk, mixed_risk)
        self.assertLess(mixed_risk, high_risk)
        self.assertGreaterEqual(low_risk, 0.0)
        self.assertLessEqual(high_risk, 1.0)

    def test_fusion_training_and_comparison_table(self):
        """Train fusion model on labeled dataset and verify comparison table structure."""
        report = train_and_evaluate_fusion("data/labeled_dataset.json")
        self.assertEqual(report["status"], "trained")
        self.assertIn("comparison_table", report)
        self.assertEqual(len(report["comparison_table"]), 4)

        # Check all 4 rows exist
        signals = [row["signal"] for row in report["comparison_table"]]
        self.assertIn("Token Entropy alone", signals)
        self.assertIn("Sampling Consistency alone", signals)
        self.assertIn("Retrieval Grounding alone", signals)
        self.assertIn("Fused Detector (Phase 2)", signals)

        # Verify weights
        weights = report["weights"]
        self.assertIn("entropy", weights)
        self.assertIn("consistency", weights)
        self.assertIn("grounding", weights)
        self.assertIn("bias", weights)

    # ==========================================
    # 6. DETECTOR SERVICE (ORCHESTRATOR) TESTS
    # ==========================================
    def test_detector_service_grounded_answer(self):
        """A fully grounded answer must receive verdict='factual' and risk < 0.35."""
        chunks = ["HDRS evaluates hallucination rates in RAG systems using token entropy and vector similarity."]
        answer = "HDRS evaluates hallucination rates in RAG systems using token entropy."
        result = self.detector.analyze_answer("What does HDRS do?", answer, chunks)

        self.assertFalse(result["has_hallucination"])
        self.assertLess(result["overall_risk"], 0.35)
        self.assertEqual(len(result["sentences"]), 1)
        self.assertEqual(result["sentences"][0]["verdict"], "factual")

    def test_detector_service_hallucinated_answer(self):
        """A fabricated answer must receive verdict='hallucinated' and risk >= 0.50."""
        chunks = ["HDRS evaluates hallucination rates in RAG systems."]
        answer = "HDRS was secretly created by pirates in Antarctica to navigate submarines."
        result = self.detector.analyze_answer("What is HDRS?", answer, chunks)

        self.assertTrue(result["has_hallucination"])
        self.assertGreaterEqual(result["overall_risk"], 0.50)
        self.assertEqual(result["sentences"][0]["verdict"], "hallucinated")

    def test_detector_service_partial_hallucination(self):
        """A two-sentence answer with 1 grounded and 1 fabricated sentence must flag only the bad one."""
        chunks = ["RAG retrieves chunks from vector databases to ground language model answers."]
        answer = "RAG retrieves chunks from vector databases. The moon is made entirely of green cheddar cheese."
        result = self.detector.analyze_answer("How does RAG work?", answer, chunks)

        self.assertEqual(len(result["sentences"]), 2)
        s1 = result["sentences"][0]
        s2 = result["sentences"][1]

        self.assertEqual(s1["verdict"], "factual")
        self.assertEqual(s2["verdict"], "hallucinated")
        self.assertTrue(result["has_hallucination"])

    def test_detector_service_honest_abstention(self):
        """Honest abstentions ('I do not know...') must be marked safe with low risk."""
        chunks = ["The document discusses deep learning algorithms."]
        answer = "I do not know based on the provided sources."
        result = self.detector.analyze_answer("What is the CEO salary?", answer, chunks)

        self.assertFalse(result["has_hallucination"])
        self.assertLessEqual(result["overall_risk"], 0.20)
        self.assertEqual(result["sentences"][0]["verdict"], "factual")

    # ==========================================
    # 7. FASTAPI API ROUTER INTEGRATION TESTS
    # ==========================================
    def test_api_detector_metrics_endpoint(self):
        """GET /api/detector/metrics returns valid Phase 2 comparison table."""
        res = self.client.get("/api/detector/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("comparison_table", data)
        self.assertIn("weights", data)
        self.assertIn("sample_count", data)

    def test_api_detector_analyze_endpoint(self):
        """POST /api/detector/analyze performs on-the-fly sentence verification."""
        payload = {
            "sentence": "Semantic chunking preserves sentence context across chunk boundaries.",
            "context": "Semantic chunking preserves sentence context across chunk boundaries.",
        }
        res = self.client.post("/api/detector/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("risk_score", data)
        self.assertIn("verdict", data)
        self.assertIn("signals", data)
        self.assertEqual(data["verdict"], "factual")

    def test_api_ask_end_to_end_pipeline(self):
        """POST /api/ask runs end-to-end RAG and attaches full detection payload."""
        self.client.app.state.pipeline.ingest_text(
            "Multi-signal hallucination detection combines entropy, consistency, and grounding.",
            "e2e_test_doc.txt"
        )
        mock_answer = "Multi-signal hallucination detection combines entropy, consistency, and grounding."
        with patch.object(self.client.app.state.pipeline, "generate", return_value=mock_answer):
            res = self.client.post("/api/ask", json={
                "query": "What signals does detection combine?",
                "top_k": 2,
                "model_name": "auto",
                "source_filter": "e2e_test_doc.txt"
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("answer", data)
            self.assertIn("detection", data)
            det = data["detection"]
            self.assertIn("sentences", det)
            self.assertIn("overall_risk", det)
            self.assertIn("has_hallucination", det)
            self.assertEqual(len(det["sentences"]), 1)
            self.assertEqual(det["sentences"][0]["verdict"], "factual")


if __name__ == "__main__":
    unittest.main()
