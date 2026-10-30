from typing import Optional, List
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    model_name: str = "auto"
    source_filter: str = None


class TextIngestRequest(BaseModel):
    text: str
    source_name: str


class SentenceAnalyzeRequest(BaseModel):
    sentence: str
    context: Optional[str] = ""
    retrieved_chunks: Optional[List[str]] = None
