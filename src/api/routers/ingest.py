import os
import shutil

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from pypdf import PdfReader

from src.api.schemas import TextIngestRequest

router = APIRouter(prefix="/api", tags=["ingest"])


def _set_ingest_status(request: Request, status: str, progress: float, step: str, message: str):
    request.app.state.ingest_status = {
        "status": status,
        "progress": progress,
        "step": step,
        "message": message,
    }


def _finalize_text_ingest(request: Request, text_content: str, source_name: str):
    pipeline = request.app.state.pipeline
    try:
        _set_ingest_status(request, "running", 35, "chunking", "Chunking the extracted text into retrieval units.")

        def progress_callback(step: str, progress: float, message: str):
            _set_ingest_status(request, "running", progress, step, message)

        chunks_added = pipeline.ingest_text(text_content, source_name, progress_callback=progress_callback)
        _set_ingest_status(request, "completed", 100, "done", f"Ingested {chunks_added} chunk(s) from {source_name}.")
    except Exception as error:
        _set_ingest_status(request, "failed", 100, "failed", str(error))


def _process_file_ingest(request: Request, temp_path: str, filename: str):
    pipeline = request.app.state.pipeline
    try:
        text_content = ""
        if filename.lower().endswith(".pdf"):
            _set_ingest_status(request, "running", 18, "extracting", "Extracting text from the uploaded PDF.")
            reader = PdfReader(temp_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        elif filename.lower().endswith((".txt", ".md", ".json")):
            _set_ingest_status(request, "running", 18, "reading", "Reading the uploaded text file.")
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, TXT, MD or JSON.")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No readable text found in file.")

        _finalize_text_ingest(request, text_content, filename)
    except Exception as error:
        _set_ingest_status(request, "failed", 100, "failed", str(error))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/ingest/text")
def ingest_text(request: Request, background_tasks: BackgroundTasks, payload: TextIngestRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        _set_ingest_status(request, "running", 10, "received", "Received text. Preparing to chunk source.")
        background_tasks.add_task(_finalize_text_ingest, request, payload.text, payload.source_name)
        return {"status": "accepted", "message": "Text ingestion started.", "source": payload.source_name}
    except Exception as error:
        _set_ingest_status(request, "failed", 100, "failed", str(error))
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/ingest/file")
async def ingest_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        _set_ingest_status(request, "running", 5, "uploading", f"Uploading {file.filename} to the server.")
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        background_tasks.add_task(_process_file_ingest, request, temp_path, file.filename)
        return {"status": "accepted", "message": "File upload received. Ingestion started.", "filename": file.filename}
    except Exception as error:
        _set_ingest_status(request, "failed", 100, "failed", str(error))
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/ingest/status")
def ingest_status(request: Request):
    return getattr(request.app.state, "ingest_status", {"status": "idle", "progress": 0, "step": "idle", "message": ""})
