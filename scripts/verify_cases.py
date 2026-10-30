import sys
sys.path.insert(0, ".")
from src.services.detector.detector_service import DetectorService

detector = DetectorService()
retrieved_chunks = [
    {"text": "Retrieval-Augmented Generation (RAG) reduces hallucination by grounding language models on external knowledge bases. However, LLMs can still misread context, over-generalize, or introduce hallucinated statements into answers.", "metadata": {}, "score": 0.89},
    {"text": "Multi-signal fusion combines token entropy (confidence), sampling consistency (agreement across stochastic samples), and retrieval grounding (semantic entailment against retrieved passages).", "metadata": {}, "score": 0.85},
]

cases = [
    ("Case 1 - Grounded", "How does multi-signal fusion work?",
     "Multi-signal fusion combines token entropy, sampling consistency, and retrieval grounding. This grounds language models on external knowledge bases to reduce hallucinated statements.",
     "FACTUAL expected"),
    ("Case 2 - Partial Hallucination", "What signals are used?",
     "Multi-signal fusion combines token entropy and sampling consistency. The algorithm was first invented by Isaac Newton during the French Revolution.",
     "Sentence 2 HALLUCINATED expected"),
    ("Case 3 - Full Fabrication", "What is RAG?",
     "RAG is a cryptographic satellite protocol launched by NASA to mine bitcoin from Mars.",
     "HALLUCINATED expected"),
    ("Case 4 - Abstention", "What is the CEO salary?",
     "I do not know based on the provided sources.",
     "FACTUAL/SAFE expected"),
]

all_pass = True
for name, query, answer, expectation in cases:
    r = detector.analyze_answer(query, answer, retrieved_chunks)
    overall = r["overall_risk"]
    has_h = r["has_hallucination"]
    print(f"\n=== {name} [{expectation}] ===")
    print(f"Overall Risk: {overall*100:.1f}%  | has_hallucination: {has_h}")
    for s in r["sentences"]:
        v = s["verdict"].upper()
        rsk = s["risk_score"] * 100
        c_r = s["signals"]["consistency_risk"]
        g_r = s["signals"]["grounding_risk"]
        txt = s["text"][:80]
        print(f"  [{v:12}] {rsk:5.1f}% | C_risk:{c_r:.3f} G_risk:{g_r:.3f} | {txt}")

print("\n=== SUMMARY ===")

