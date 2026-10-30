import unittest
from src.services.detector.signals.entropy import compute_sentence_entropy
from src.services.detector.signals.consistency import compute_sentence_consistency, compute_consistency_risk
from src.services.detector.signals.grounding import compute_sentence_grounding, compute_grounding_risk


class TestDetectionSignals(unittest.TestCase):
    def test_entropy_with_hedges(self):
        confident_sent = "LLM-Check operates in a single response setting."
        hedged_sent = "It is supposedly believed that the model might perhaps hallucinate around 80%."
        
        c_entropy = compute_sentence_entropy(confident_sent)
        h_entropy = compute_sentence_entropy(hedged_sent)
        self.assertGreater(h_entropy, c_entropy)

    def test_grounding_high_vs_low(self):
        retrieved_chunks = [
            {"text": "LLM-Check demonstrates up to 45x speedup over existing baselines without extra model calls."}
        ]
        grounded_sent = "LLM-Check achieves up to 45x speedup compared to baselines."
        hallucinated_sent = "The spaceship traveled across the Martian galaxy at warp speed 9."

        g_score = compute_sentence_grounding(grounded_sent, retrieved_chunks)
        h_score = compute_sentence_grounding(hallucinated_sent, retrieved_chunks)

        self.assertGreater(g_score, h_score)
        self.assertLess(compute_grounding_risk(g_score), compute_grounding_risk(h_score))

    def test_consistency_agreement(self):
        sentence = "LLM-Check is an efficient hallucination detector."
        agreeing_samples = [
            "The proposed method LLM-Check detects hallucinations efficiently.",
            "LLM-Check provides an effective and fast way to detect LLM hallucinations.",
        ]
        disagreeing_samples = [
            "Photosynthesis is the biological process of turning sunlight into chemical energy.",
            "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
        ]

        high_cons = compute_sentence_consistency(sentence, agreeing_samples)
        low_cons = compute_sentence_consistency(sentence, disagreeing_samples)

        self.assertGreater(high_cons, low_cons)
        self.assertLess(compute_consistency_risk(high_cons), compute_consistency_risk(low_cons))


if __name__ == "__main__":
    unittest.main()
