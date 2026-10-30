from fastapi import APIRouter, HTTPException
import logging

from src.api.schemas import SentenceAnalyzeRequest
from src.services.detector.detector_service import DetectorService
from src.services.detector.fusion import get_latest_metrics, train_and_evaluate_fusion
from src.services.supabase_store import supabase_report_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detector", tags=["detector"])
_detector_service = None


def get_detector() -> DetectorService:
    global _detector_service
    if _detector_service is None:
        _detector_service = DetectorService()
    return _detector_service


@router.get("/metrics")
def get_detector_metrics():
    """Retrieve the latest Phase 2 model metrics and comparison table

    (Token Entropy vs. Sampling Consistency vs. Retrieval Grounding vs. Fused Detector).
    """
    try:
        cloud_metrics = supabase_report_store.latest_detector_metrics()
        return cloud_metrics or get_latest_metrics()
    except Exception as e:
        logger.exception("Failed to get detector metrics: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
def train_detector():
    """Train the supervised Phase 2 Logistic Regression fusion model

    on the benchmark data in data/labeled_dataset.json.
    """
    try:
        metrics = train_and_evaluate_fusion()
        supabase_report_store.save_detector_metrics(metrics)
        return {
            "message": "Phase 2 fusion model trained successfully.",
            "metrics": metrics,
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        logger.exception("Failed to train fusion detector: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
def analyze_sentence(payload: SentenceAnalyzeRequest):
    """Analyze a single sentence or text against provided context

    using the Phase 2 multi-signal fusion pipeline.
    """
    try:
        detector = get_detector()
        chunks = payload.retrieved_chunks or ([payload.context] if payload.context else [])
        result = detector.analyze_answer(
            query="Ad-hoc analysis",
            answer=payload.sentence,
            retrieved_chunks=chunks,
        )
        if result["sentences"]:
            sent = result["sentences"][0]
            return {
                "sentence": sent["text"],
                "risk_score": sent["risk_score"],
                "verdict": sent["verdict"],
                "signals": sent["signals"],
                "explanation": sent["explanation"],
                "overall_risk": result["overall_risk"],
                "has_hallucination": result["has_hallucination"],
            }
        return {
            "sentence": payload.sentence,
            "risk_score": 0.0,
            "verdict": "factual",
            "signals": {},
            "explanation": "No valid sentence extracted.",
            "overall_risk": 0.0,
            "has_hallucination": False,
        }
    except Exception as e:
        logger.exception("Failed to analyze sentence: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
