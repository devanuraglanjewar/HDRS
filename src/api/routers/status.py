from fastapi import APIRouter, Request

from src.core.llm import OLLAMA_MODEL, provider_status
from src.core.settings import GOOGLE_API_KEY, NVIDIA_API_KEY, NEMOTRON_BASE_URL
from src.core.vector_store import CHROMA_AVAILABLE
from src.services.supabase_store import supabase_report_store

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
def get_db_status(request: Request):
    pipeline = request.app.state.pipeline
    return {
        "vector_store_type": "ChromaDB" if not pipeline.vector_store.use_fallback else "NumPy JSON Fallback",
        "vector_store_connection": getattr(getattr(pipeline.vector_store, "store", None), "connection_mode", "unknown"),
        "chromadb_available": CHROMA_AVAILABLE,
        "indexed_chunks_count": pipeline.vector_store.get_count(),
        "gemini_api_key_configured": bool(GOOGLE_API_KEY),
        "ollama_available": provider_status()["ollama"]["configured"],
        "nemotron_api_key_configured": bool(NVIDIA_API_KEY),
        "nemotron_endpoint": NEMOTRON_BASE_URL,
        "provider_status": provider_status(),
        "routing_order": ["gemini", "nemotron", "ollama"],
        "supabase_configured": supabase_report_store.configured,
        "report_storage": "supabase+local_backup" if supabase_report_store.configured else "local_only",
        "document_storage": "supabase+chroma_cloud" if supabase_report_store.configured else "chroma_cloud_or_local",
        "active_generator": "auto: Gemini → Nemotron → Ollama",
        "active_judge": "auto: Gemini → Nemotron → Ollama",
    }
