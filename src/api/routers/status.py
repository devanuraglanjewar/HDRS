from fastapi import APIRouter, Request

from src.core.llm import OLLAMA_AVAILABLE, OLLAMA_MODEL
from src.core.settings import GOOGLE_API_KEY
from src.core.vector_store import CHROMA_AVAILABLE

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
        "ollama_available": OLLAMA_AVAILABLE,
        "active_generator": OLLAMA_MODEL if OLLAMA_AVAILABLE else "gemini-3.6-flash (API)",
        "active_judge": OLLAMA_MODEL if OLLAMA_AVAILABLE else "gemini-3.6-flash (API)",
    }
