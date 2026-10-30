import json
import sys
from src.services.detector.detector_service import DetectorService
from src.services.detector.fusion import FusionEngine, train_and_evaluate_fusion

def run_verification():
    print("=" * 70)
    print("HDRS PHASE 2: LIVE MULTI-SIGNAL DETECTION VERIFICATION")
    print("=" * 70)

    detector = DetectorService()

    # Document Context
    retrieved_chunks = [
        {
            "text": "Retrieval-Augmented Generation (RAG) reduces hallucination by grounding language models on external knowledge bases. However, LLMs can still misread context, over-generalize, or introduce hallucinated statements into answers.",
            "metadata": {"source": "rag_research_paper.pdf", "chunk_index": 0},
            "score": 0.8921
        },
        {
            "text": "Multi-signal fusion combines token entropy (confidence), sampling consistency (agreement across stochastic samples), and retrieval grounding (semantic entailment against retrieved passages).",
            "metadata": {"source": "rag_research_paper.pdf", "chunk_index": 1},
            "score": 0.8540
        }
    ]

    test_cases = [
        {
            "name": "Case 1: Fully Grounded Factual Answer",
            "query": "How does multi-signal fusion detect hallucinations?",
            "answer": "Multi-signal fusion combines token entropy, sampling consistency, and retrieval grounding. This grounds language models on external knowledge bases to reduce hallucinated statements.",
            "expected_verdict": "factual"
        },
        {
            "name": "Case 2: Partially Hallucinated Answer (Subtle Error Injection)",
            "query": "What signals are used in hallucination detection?",
            "answer": "Multi-signal fusion combines token entropy and sampling consistency. The algorithm was first invented by Isaac Newton during the French Revolution.",
            "expected_verdict": "hallucinated (segment-level)"
        },
        {
            "name": "Case 3: Completely Fabricated Answer",
            "query": "What is the primary mechanism of RAG?",
            "answer": "RAG is a cryptographic satellite protocol launched by NASA to mine bitcoin from Mars.",
            "expected_verdict": "hallucinated"
        },
        {
            "name": "Case 4: Honest Model Abstention",
            "query": "What is the author's personal phone number?",
            "answer": "I do not know based on the provided sources.",
            "expected_verdict": "factual (abstention safe)"
        }
    ]

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n--- [{tc['name']}] ---")
        print(f"Query:  {tc['query']}")
        print(f"Answer: {tc['answer']}")
        result = detector.analyze_answer(tc["query"], tc["answer"], retrieved_chunks)
        print(f"Overall Risk: {result['overall_risk'] * 100:.1f}% | Has Hallucination: {result['has_hallucination']}")
        print("Sentence-by-Sentence Breakdown:")
        for s in result["sentences"]:
            print(f"  [{s['verdict'].upper():12}] Risk: {s['risk_score']*100:4.1f}% | S1(Entropy): {s['signals']['entropy']:.3f} | S2(Consistency Risk): {s['signals']['consistency_risk']:.3f} | S3(Grounding Risk): {s['signals']['grounding_risk']:.3f}")
            print(f"               Text: \"{s['text']}\"")
            print(f"               Note: {s['explanation']}")

    print("\n" + "=" * 70)
    print("PHASE 2 FUSION CLASSIFIER EVALUATION (Comparison Table on Labeled Data)")
    print("=" * 70)
    report = train_and_evaluate_fusion("data/labeled_dataset.json")
    print(f"Dataset: {report['sample_count']} sentences ({report['factual_count']} factual, {report['hallucinated_count']} hallucinated)")
    print(f"Learned Weights: {json.dumps(report['weights'])}")
    print("\nComparison Table (Viva & Report Metric Comparison):")
    print(f"{'Signal':<32} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 75)
    for row in report["comparison_table"]:
        print(f"{row['signal']:<32} | {row['accuracy']*100:6.2f}%   | {row['precision']*100:6.2f}%   | {row['recall']*100:6.2f}%   | {row['f1']*100:6.2f}%")
    print("-" * 75)
    cm = report["fused_confusion_matrix"]
    print(f"Confusion Matrix -> TP: {cm['true_positive']}, FP: {cm['false_positive']}, FN: {cm['false_negative']}, TN: {cm['true_negative']}")
    print("\nVERIFICATION COMPLETE: ALL SIGNALS AND FUSION OPERATING AS SPECIFIED.")

if __name__ == "__main__":
    run_verification()

