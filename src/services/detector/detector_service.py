import logging
from typing import List, Dict, Any, Union

from src.services.detector.sentence_splitter import split_into_sentences
from src.services.detector.signals.entropy import compute_sentence_entropy
from src.services.detector.signals.consistency import compute_sentence_consistency, compute_consistency_risk
from src.services.detector.signals.grounding import compute_sentence_grounding, compute_grounding_risk
from src.services.detector.fusion import FusionEngine

logger = logging.getLogger(__name__)


class DetectorService:
    def __init__(self):
        self.fusion_engine = FusionEngine()

    def analyze_answer(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Union[str, Dict[str, Any]]],
        alternative_samples: List[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a generated answer sentence-by-sentence using the Phase 2

        multi-signal detector (Token Entropy, Sampling Consistency, Retrieval Grounding,
        and Supervised Fusion).
        """
        if not answer or not answer.strip():
            return {
                "answer": answer,
                "sentences": [],
                "overall_risk": 0.0,
                "has_hallucination": False,
            }

        # Check for honest model abstention / refusal to guess
        abstain_patterns = [
            "i do not know", "i don't know", "cannot find", "no information",
            "not mentioned in the text", "not provided in the source",
            "not found in the document", "does not contain information",
            "no relevant context found", "unable to answer based on",
        ]
        lower_ans = answer.strip().lower()
        if any(p in lower_ans for p in abstain_patterns):
            return {
                "answer": answer,
                "sentences": [
                    {
                        "index": 0,
                        "text": answer.strip(),
                        "risk_score": 0.05,
                        "verdict": "factual",
                        "signals": {
                            "entropy": 0.05,
                            "consistency": 1.0,
                            "consistency_risk": 0.0,
                            "grounding": 1.0,
                            "grounding_risk": 0.0,
                        },
                        "explanation": "Honest model abstention — model safely refrained from fabricating ungrounded claims.",
                    }
                ],
                "overall_risk": 0.05,
                "has_hallucination": False,
                "hallucinated_sentences_count": 0,
                "total_sentences": 1,
            }

        sentences = split_into_sentences(answer)
        if not sentences:
            sentences = [answer.strip()]

        context_text = "\n\n".join(
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in (retrieved_chunks or [])
        )

        analyzed_sentences = []
        hallucinated_count = 0
        risk_scores = []

        # For consistency checking: use each retrieved chunk text individually as an "alternative sample".
        # This is semantically correct — each chunk represents a fragment the model "should" agree with
        # if it stayed grounded. Using context[:500] (the concatenation) gave artificially low cosine
        # similarity because topically distinct chunks dilute each other when merged.
        chunk_texts_for_consistency = [
            chunk.get("text", "")[:600] if isinstance(chunk, dict) else str(chunk)[:600]
            for chunk in (retrieved_chunks or [])
            if (chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)).strip()
        ]
        if alternative_samples:
            consistency_samples = alternative_samples
        elif chunk_texts_for_consistency:
            consistency_samples = chunk_texts_for_consistency
        else:
            consistency_samples = []

        for idx, sent in enumerate(sentences):
            sent_text = sent.strip()
            if not sent_text:
                continue

            # Signal 1: Token Entropy
            entropy = compute_sentence_entropy(sent_text, context=context_text)

            # Signal 2: Sampling Consistency
            consistency_raw = compute_sentence_consistency(sent_text, alternative_samples=consistency_samples)
            consistency_risk = compute_consistency_risk(consistency_raw)

            # Signal 3: Retrieval Grounding
            grounding_raw = compute_sentence_grounding(sent_text, retrieved_chunks=retrieved_chunks)
            grounding_risk = compute_grounding_risk(grounding_raw)

            # Fusion Model Risk Score
            fused_risk = self.fusion_engine.predict_risk(
                entropy_risk=entropy,
                consistency_risk=consistency_risk,
                grounding_risk=grounding_risk,
            )
            fused_risk = round(float(fused_risk), 4)
            risk_scores.append(fused_risk)

            # Determine verdict and explanation (0.50 standard decision threshold for logistic regression)
            if fused_risk >= 0.50:
                verdict = "hallucinated"
                hallucinated_count += 1
            elif fused_risk >= 0.30:
                verdict = "uncertain"
            else:
                verdict = "factual"

            explanation = self._generate_explanation(entropy, consistency_risk, grounding_risk, verdict)

            analyzed_sentences.append({
                "index": idx,
                "text": sent_text,
                "risk_score": fused_risk,
                "verdict": verdict,
                "signals": {
                    "entropy": round(float(entropy), 4),
                    "consistency": round(float(consistency_raw), 4),
                    "consistency_risk": round(float(consistency_risk), 4),
                    "grounding": round(float(grounding_raw), 4),
                    "grounding_risk": round(float(grounding_risk), 4),
                },
                "explanation": explanation,
            })

        overall_risk = round(float(sum(risk_scores) / len(risk_scores)), 4) if risk_scores else 0.0

        return {
            "answer": answer,
            "sentences": analyzed_sentences,
            "overall_risk": overall_risk,
            "has_hallucination": hallucinated_count > 0,
            "hallucinated_sentences_count": hallucinated_count,
            "total_sentences": len(analyzed_sentences),
        }

    def _generate_explanation(
        self,
        entropy_risk: float,
        consistency_risk: float,
        grounding_risk: float,
        verdict: str,
    ) -> str:
        if verdict == "factual":
            return "Claim is well-grounded in source passages with high model confidence."

        reasons = []
        if grounding_risk >= 0.55:
            reasons.append("Claim not supported by retrieved source text")
        if consistency_risk >= 0.60:
            reasons.append("Inconsistent across stochastic samples")
        if entropy_risk >= 0.50:
            reasons.append("High model token uncertainty/hedging")

        if not reasons:
            reasons.append("Moderate uncertainty across combined signals")

        prefix = "Hallucination flagged: " if verdict == "hallucinated" else "Borderline confidence: "
        return prefix + "; ".join(reasons) + "."
