import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers.benchmark import router as benchmark_router
from src.api.routers.chat import router as chat_router
from src.api.routers.documents import router as documents_router
from src.api.routers.ingest import router as ingest_router
from src.api.routers.status import router as status_router
from src.core.settings import DEV_SERVER_URL
from src.services.benchmark_service import reset_benchmark_artifacts
from src.services.rag_pipeline import RAGPipeline


def create_app() -> FastAPI:
    app = FastAPI(title="Multi-Signal Hallucination Detector (Phase 1)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.pipeline = RAGPipeline()
    app.state.reset_benchmark_artifacts = reset_benchmark_artifacts
    app.state.ingest_status = {"status": "idle", "progress": 0, "step": "idle", "message": ""}

    app.include_router(status_router)
    app.include_router(ingest_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(benchmark_router)

    if DEV_SERVER_URL:
        @app.get("/")
        def redirect_to_dev_server():
            return RedirectResponse(url=DEV_SERVER_URL)
    else:
        os.makedirs("static", exist_ok=True)
        app.mount("/", StaticFiles(directory="static", html=True), name="static")

    return app
