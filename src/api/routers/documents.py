import json
import os

from fastapi import APIRouter, HTTPException, Request

from src.services.benchmark_service import reset_benchmark_artifacts

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents")
def get_all_documents(request: Request):
    pipeline = request.app.state.pipeline
    try:
        data = pipeline.vector_store.get_all()
        if not data or not data.get("documents"):
            return []

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])

        grouped_docs = {}
        for idx, doc_text in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            source = meta.get("source", "Unknown Document")
            chunk_idx = meta.get("chunk_index", 0)
            grouped_docs.setdefault(source, []).append({
                "chunk_index": chunk_idx,
                "text": doc_text,
                "id": ids[idx] if idx < len(ids) else f"chunk_{idx}",
            })

        result = []
        for source, chunks in grouped_docs.items():
            chunks.sort(key=lambda x: x["chunk_index"])
            reconstructed_text = "\n\n".join([c["text"] for c in chunks])
            result.append({
                "source": source,
                "chunks_count": len(chunks),
                "full_text": reconstructed_text,
                "chunks": chunks,
            })
        return result
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/db/clear")
def clear_database(request: Request):
    pipeline = request.app.state.pipeline
    try:
        pipeline.vector_store.clear()
        reset_benchmark_artifacts()
        return {
            "status": "success",
            "message": "Vector store database cleared successfully.",
            "benchmark_reset": True,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.delete("/documents/{source_name:path}")
def delete_document_source(request: Request, source_name: str):
    pipeline = request.app.state.pipeline
    if not source_name.strip():
        raise HTTPException(status_code=400, detail="Source name cannot be empty.")
    try:
        deleted_count = pipeline.vector_store.delete_by_source(source_name)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Source not found.")
        remaining_count = pipeline.vector_store.get_count()
        benchmark_reset = False
        if remaining_count == 0:
            reset_benchmark_artifacts()
            benchmark_reset = True
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} chunk(s) from {source_name}.",
            "deleted_chunks": deleted_count,
            "source": source_name,
            "remaining_chunks": remaining_count,
            "benchmark_reset": benchmark_reset,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
