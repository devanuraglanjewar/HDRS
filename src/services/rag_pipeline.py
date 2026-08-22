import re
import time
import uuid
import logging

from src.core.chunker import SemanticChunker
from src.core.llm import (
    OLLAMA_AVAILABLE,
    OLLAMA_MODEL,
    embed_text,
    generate_with_gemini,
    generate_with_ollama,
)
from src.core.prompts import rag_system_instruction
from src.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, persist_dir="data/chroma_db", collection_name="rag_documents"):
        self.chunker = SemanticChunker()
        self.vector_store = VectorStoreManager(persist_dir=persist_dir, collection_name=collection_name)

    def get_embedding(self, text: str) -> list:
        return embed_text(text)

    def ingest_text(self, text: str, source_name: str, progress_callback=None) -> int:
        chunks = self.chunker.chunk_text(text, source_name)
        if not chunks:
            return 0

        if progress_callback:
            progress_callback("chunked", 35, f"Chunked document into {len(chunks)} segment(s).")

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = []
        total_chunks = len(texts)
        for idx, txt in enumerate(texts):
            embeddings.append(self.get_embedding(txt))
            if progress_callback:
                pct = 35 + ((idx + 1) / total_chunks) * 45
                progress_callback("embedding", round(pct, 1), f"Embedding chunk {idx + 1} of {total_chunks}.")
        self.vector_store.add_documents(texts, embeddings, metadatas, ids)
        if progress_callback:
            progress_callback("saving", 92, f"Saving {len(chunks)} chunk(s) to the vector store.")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3, source_filter: str = None) -> list:
        """Run a vector query and return up to `top_k` unique, score-sorted chunks.

        If embedding fails or produces a zero-vector, fall back to a simple
        lexical similarity search over stored documents to avoid returning the
        same static context for every query.
        """
        query_embedding = self.get_embedding(query)

        # detect zero embedding (embedding service missing or failed)
        try:
            q_norm = sum(abs(x) for x in query_embedding)
        except Exception:
            q_norm = 0

        if not query_embedding or q_norm == 0:
            # Fallback lexical matching: score chunks by shared token overlap
            logger.debug("retrieve: embedding missing or zero for query; using lexical fallback. query=%s, top_k=%d, source_filter=%s", (query[:200] + '...') if query and len(query) > 200 else query, top_k, source_filter)
            store_all = self.vector_store.get_all() or {}
            docs = store_all.get("documents", [])
            metadatas = store_all.get("metadatas", [])
            candidates = []
            q_tokens = set((query or "").lower().split())
            for idx, doc_text in enumerate(docs):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                if source_filter and meta.get("source") != source_filter:
                    continue
                text = (doc_text or "").strip()
                if not text:
                    continue
                overlap = len(q_tokens.intersection(set(text.lower().split())))
                candidates.append({"text": text, "metadata": meta, "score": float(overlap)})
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            try:
                top_info = [{"source": c.get("metadata", {}).get("source"), "score": c.get("score", 0.0), "preview": (c.get("text") or "")[:120]} for c in candidates[:top_k]]
                logger.debug("retrieve(lexical_fallback) top candidates: %s", top_info)
                scores = [float(c.get("score", 0.0)) for c in candidates[:top_k]]
                if scores and all(s == scores[0] for s in scores):
                    logger.warning("retrieve(lexical_fallback): top %d scores identical (%s) for query=%s - may indicate poor lexical variation", top_k, scores, (query[:200] + "...") if query and len(query) > 200 else query)
            except Exception:
                pass
            return candidates[:top_k]

        # ask the underlying store for candidates (allow store to apply its own top_k heuristics)
        results = self.vector_store.query(query_embedding, top_k=top_k, source_filter=source_filter)
        if not results:
            return []

        # ensure results are sorted by score descending
        try:
            results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        except Exception:
            logger.debug("Failed to sort retrieval results by score; returning as-is.", exc_info=True)

        # Log detailed score info for diagnosis
        try:
            score_info = []
            for r in results[: max(top_k, 10) ]:
                meta = r.get("metadata") or {}
                score_info.append({
                    "source": meta.get("source"),
                    "score": float(r.get("score", 0.0)),
                    "preview": (r.get("text") or "")[:120],
                })
            logger.debug("retrieve: top candidates for query=%s -> %s", (query[:200] + "...") if query and len(query) > 200 else query, score_info)
            scores = [float(r.get("score", 0.0)) for r in results[:top_k]]
            if scores and all(s == scores[0] for s in scores):
                logger.warning("retrieve: top %d scores identical (%s) for query=%s - may indicate embedding failures or static vectors", top_k, scores, (query[:200] + "...") if query and len(query) > 200 else query)
        except Exception:
            pass

        unique = []
        seen_texts = set()
        for r in results:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            if text in seen_texts:
                # skip exact duplicate text fragments
                continue
            seen_texts.add(text)
            unique.append(r)
            if len(unique) >= top_k:
                break

        # if dedup removed too many items, fall back to filling up from original candidates
        if len(unique) < top_k:
            for r in results:
                if r in unique:
                    continue
                unique.append(r)
                if len(unique) >= top_k:
                    break

        logger.debug("retrieve(query=%s, top_k=%d, source_filter=%s) -> %d unique chunks (requested %d)", query if len(query) < 200 else query[:200] + "...", top_k, source_filter, len(unique), top_k)
        return unique

    def generate(self, query: str, retrieved_chunks: list, model_name: str = "auto") -> str:
        if not retrieved_chunks:
            return "No relevant context found to answer the query."

        context_str = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
        system_instruction = rag_system_instruction()
        full_prompt = f"""Context Passages:
{context_str}

Query: {query}

Answer:"""

        use_ollama = ((model_name == "auto" and OLLAMA_AVAILABLE) or model_name.startswith("ollama:"))
        if use_ollama:
            ollama_model = OLLAMA_MODEL if model_name in ("auto", "ollama") else model_name.replace("ollama:", "")
            try:
                return self._sanitize_generated_answer(generate_with_ollama(full_prompt, system_instruction, ollama_model))
            except Exception:
                pass

        gemini_model = "gemini-3.6-flash" if model_name == "auto" else model_name
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._sanitize_generated_answer(generate_with_gemini(full_prompt, system_instruction, gemini_model))
            except Exception as error:
                err_msg = str(error).lower()
                is_rate_limit = any(x in err_msg for x in ["429", "quota", "resourceexhausted", "rate limit"])
                if is_rate_limit and attempt < max_retries - 1:
                    time.sleep(2.0 + attempt)
                else:
                    return f"Error generating answer: {error}"

    def _sanitize_generated_answer(self, answer: str) -> str:
        if not answer:
            return answer
        cleaned = re.sub(r"(?i)\baccording to the context passage from\s+[^\n,.!?]+(?:\s*\(chunk\s*\d+\))?,?\s*", "", answer)
        cleaned = re.sub(r"(?i)\bchunk\s*\d+\b", "", cleaned)
        cleaned = re.sub(r"(?i)\bHaluEval_Doc_\d+\b", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
        return cleaned

    def ask(self, query: str, top_k: int = 3, model_name: str = "auto", source_filter: str = None) -> dict:
        chunks = self.retrieve(query, top_k, source_filter)
        answer = self.generate(query, chunks, model_name)
        retrieved_context = "\n\n".join(chunk["text"] for chunk in chunks)
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "retrieved_context": retrieved_context,
            "model_used": OLLAMA_MODEL if (model_name == "auto" and OLLAMA_AVAILABLE) else ("gemini-3.6-flash" if model_name == "auto" else model_name),
        }
