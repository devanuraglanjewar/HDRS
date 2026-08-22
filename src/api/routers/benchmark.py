import json
import os

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Query
from fastapi.responses import Response

from src.services.benchmark_service import LABELED_DATA_FILE, RESULTS_FILE, STATUS_FILE, get_status, run_benchmark, download_halueval_subset, _generate_source_benchmark_samples, _load_source_text

router = APIRouter(prefix="/api", tags=["benchmark"])


@router.post("/benchmark/generate_samples")
def generate_samples(
    num_samples: int = Form(12),
    source_filter: str = Form(None),
):
    """
    Generate a small set of QA samples for preview without running the full benchmark.
    Returns JSON: { samples: [...] }
    """
    source_filter = (source_filter or "").strip() or None

    try:
        if source_filter:
            source_text = _load_source_text(source_filter)
            if not source_text:
                raise HTTPException(status_code=404, detail=f"No indexed text found for source '{source_filter}'.")
            samples = _generate_source_benchmark_samples(source_filter, source_text, num_samples)
        else:
            samples = download_halueval_subset(num_samples)
        return {"samples": samples}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error generating samples: {error}")


@router.post("/benchmark/run")
def trigger_benchmark(
    background_tasks: BackgroundTasks,
    num_samples: int = Form(50),
    model_name: str = Form("auto"),
    source_filter: str = Form(None),
):
    status = get_status()
    if status["status"] == "running":
        return {"status": "already_running", "message": "A benchmark run is already in progress."}

    background_tasks.add_task(run_benchmark, num_samples=num_samples, model_name=model_name, source_filter=source_filter)
    scope = source_filter.strip() if source_filter and source_filter.strip() else "HaluEval QA dataset"
    return {"status": "started", "message": f"Benchmark triggered with {num_samples} samples using {model_name} for {scope}."}


@router.get("/benchmark/status")
def get_benchmark_status():
    return get_status()


@router.get("/benchmark/results")
def get_benchmark_results():
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(LABELED_DATA_FILE):
        return {"available": False, "message": "No benchmark results found. Please run the benchmark first."}

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return {"available": True, "summary": summary, "dataset": dataset[:15]}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error reading benchmark results: {error}")


@router.get("/benchmark/report")
def download_benchmark_report(source_filter: str = Query(None)):
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(LABELED_DATA_FILE):
        raise HTTPException(status_code=404, detail="No benchmark report found. Please run the benchmark first.")

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        selected_source = (source_filter or "").strip() or None
        if selected_source and summary.get("source_filter") and summary.get("source_filter") != selected_source:
            raise HTTPException(
                status_code=409,
                detail=f"Saved benchmark report belongs to '{summary.get('source_filter')}', not '{selected_source}'.",
            )
        if selected_source and summary.get("mode") != "source_scoped":
            raise HTTPException(status_code=409, detail="Saved benchmark report is not source-scoped.")

        report = {
            "summary": summary,
            "dataset": dataset,
            "selected_source": selected_source,
        }

        filename_scope = selected_source or summary.get("benchmark_scope") or "benchmark"
        filename = f"{filename_scope}_benchmark_report.json".replace(os.sep, "_")
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=payload, media_type="application/json", headers=headers)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error creating benchmark report: {error}")


@router.delete("/benchmark/report")
def delete_benchmark_report(source_filter: str = Query(None)):
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(LABELED_DATA_FILE):
        raise HTTPException(status_code=404, detail="No benchmark report found. Please run the benchmark first.")

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            summary = json.load(f)

        selected_source = (source_filter or "").strip() or None
        saved_source = summary.get("source_filter")

        if selected_source and saved_source and saved_source != selected_source:
            raise HTTPException(
                status_code=409,
                detail=f"Saved benchmark report belongs to '{saved_source}', not '{selected_source}'.",
            )
        if selected_source and summary.get("mode") != "source_scoped":
            raise HTTPException(status_code=409, detail="Saved benchmark report is not source-scoped.")
        if selected_source and not saved_source:
            raise HTTPException(status_code=409, detail="Saved benchmark report does not belong to a specific source.")

        for path in (RESULTS_FILE, LABELED_DATA_FILE, STATUS_FILE):
            if os.path.exists(path):
                os.remove(path)

        return {
            "status": "deleted",
            "message": f"Benchmark report deleted for {selected_source or summary.get('benchmark_scope') or 'current scope'}.",
            "deleted_scope": selected_source or summary.get("benchmark_scope"),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error deleting benchmark report: {error}")
