import os
import unittest
from src.services.detector.fusion import FusionEngine, train_and_evaluate_fusion, get_latest_metrics


class TestFusionEngine(unittest.TestCase):
    def test_predict_risk_linear_and_bounds(self):
        engine = FusionEngine()
        # Zero risks should yield very low risk
        low_risk = engine.predict_risk(entropy_risk=0.0, consistency_risk=0.0, grounding_risk=0.0)
        self.assertGreaterEqual(low_risk, 0.0)
        self.assertLessEqual(low_risk, 0.3)

        # High risks should yield high risk (above 0.5 decision boundary)
        high_risk = engine.predict_risk(entropy_risk=1.0, consistency_risk=1.0, grounding_risk=1.0)
        self.assertGreater(high_risk, 0.5)
        self.assertLessEqual(high_risk, 1.0)

    def test_train_and_evaluate_on_labeled_dataset(self):
        dataset_path = "data/labeled_dataset.json"
        if not os.path.exists(dataset_path):
            self.skipTest(f"{dataset_path} does not exist yet.")

        report = train_and_evaluate_fusion(dataset_path)
        self.assertIn("comparison_table", report)
        self.assertIn("weights", report)
        self.assertGreater(len(report["comparison_table"]), 3)

        # Verify fused detector is in the comparison table
        signal_names = [row["signal"] for row in report["comparison_table"]]
        self.assertTrue(any("Fused" in s for s in signal_names))


if __name__ == "__main__":
    unittest.main()
