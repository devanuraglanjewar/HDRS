"""Migrate existing local extracted data and detector artifacts to Supabase.

Run after applying supabase/benchmark_reports.sql:
    ./.venv/bin/python scripts/migrate_local_data_to_supabase.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.llm import OLLAMA_MODEL  # noqa: F401 - initializes project settings
from src.services.rag_pipeline import RAGPipeline
from src.services.supabase_store import supabase_report_store


def main():
    if not supabase_report_store.configured:
        raise SystemExit("Supabase is not configured")

    pipeline = RAGPipeline()
    data = pipeline.vector_store.get_all() or {}
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    grouped = {}
    for idx, text in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        source = metadata.get("source", "Unknown Document")
        grouped.setdefault(source, []).append((metadata.get("chunk_index", idx), text))

    migrated_documents = 0
    for source, chunks in grouped.items():
        chunks.sort(key=lambda item: item[0])
        full_text = "\n\n".join(text or "" for _, text in chunks)
        if supabase_report_store.save_document(source, full_text, len(chunks)):
            migrated_documents += 1

    migrated_reports = 0
    if os.path.exists("data/benchmark_results.json") and os.path.exists("data/labeled_dataset.json"):
        with open("data/benchmark_results.json", encoding="utf-8") as f:
            summary = json.load(f)
        with open("data/labeled_dataset.json", encoding="utf-8") as f:
            dataset = json.load(f)
        if supabase_report_store.save(summary, dataset):
            migrated_reports = 1

    migrated_metrics = 0
    if os.path.exists("data/fusion_metrics.json"):
        with open("data/fusion_metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)
        if supabase_report_store.save_detector_metrics(metrics):
            migrated_metrics = 1

    print(json.dumps({
        "documents": migrated_documents,
        "benchmark_reports": migrated_reports,
        "detector_metrics": migrated_metrics,
    }))


if __name__ == "__main__":
    main()
