from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import QueryRequest

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/ask")
def query_rag(request: Request, payload: QueryRequest):
    pipeline = request.app.state.pipeline
    try:
        res = pipeline.ask(payload.query, payload.top_k, payload.model_name, payload.source_filter)
        return {
            "query": res["query"],
            "answer": res["answer"],
            "model_used": res.get("model_used", "unknown"),
            "retrieved_chunks": [
                {
                    "text": chunk["text"],
                    "source": chunk["metadata"].get("source", "Unknown"),
                    "chunk_index": chunk["metadata"].get("chunk_index", 0),
                    "score": round(chunk["score"], 4),
                }
                for chunk in res["retrieved_chunks"]
            ],
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
