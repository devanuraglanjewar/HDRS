import json
import os

import numpy as np

from src.core.settings import CHROMA_API_KEY, CHROMA_DATABASE, CHROMA_HOST, CHROMA_TENANT

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMA_AVAILABLE = False


class ChromaVectorStore:
    def __init__(self, persist_dir="data/chroma_db", collection_name="rag_documents"):
        self.connection_mode = "local_persistent"
        self.client = None

        if CHROMA_API_KEY and chromadb:
            try:
                if CHROMA_HOST:
                    self.client = chromadb.CloudClient(
                        cloud_host=CHROMA_HOST,
                        cloud_port=443,
                        tenant=CHROMA_TENANT or None,
                        database=CHROMA_DATABASE or None,
                        api_key=CHROMA_API_KEY,
                    )
                else:
                    self.client = chromadb.CloudClient(
                        tenant=CHROMA_TENANT or None,
                        database=CHROMA_DATABASE or None,
                        api_key=CHROMA_API_KEY,
                    )
                self.client.heartbeat()
                self.connection_mode = "chroma_cloud"
            except Exception:
                self.client = None

        if self.client is None:
            self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, texts, embeddings, metadatas, ids):
        self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

    def get_persist_dir(self):
        return getattr(self.client, "path", None)

    def query(self, query_embedding, top_k=3):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        formatted = []
        if results and results.get("documents") and results["documents"][0]:
            for idx in range(len(results["documents"][0])):
                dist = results["distances"][0][idx] if results.get("distances") else 0.0
                formatted.append({
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx] if results.get("metadatas") else {},
                    "score": float(1.0 - dist),
                })
        return formatted

    def get_count(self):
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_all(self):
        return self.collection.get()

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_by_source(self, source_name: str) -> int:
        items = self.collection.get(where={"source": source_name})
        ids = items.get("ids", []) if items else []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)


class NumpyVectorStore:
    def __init__(self, persist_path="data/numpy_store.json"):
        self.persist_path = persist_path
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        self.load()

    def load(self):
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.embeddings = [np.array(e) for e in data.get("embeddings", [])]
            self.metadatas = data.get("metadatas", [])
            self.ids = data.get("ids", [])

    def save(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "embeddings": [e.tolist() for e in self.embeddings],
                "metadatas": self.metadatas,
                "ids": self.ids,
            }, f)

    def add_documents(self, texts, embeddings, metadatas, ids):
        self.documents.extend(texts)
        self.embeddings.extend([np.array(e) for e in embeddings])
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        self.save()

    def query(self, query_embedding, top_k=3):
        if not self.embeddings:
            return []
        q_emb = np.array(query_embedding)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return []
        similarities = []
        for idx, emb in enumerate(self.embeddings):
            emb_norm = np.linalg.norm(emb)
            sim = 0.0 if emb_norm == 0 else float(np.dot(q_emb, emb) / (q_norm * emb_norm))
            similarities.append((sim, idx))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": self.documents[idx], "metadata": self.metadatas[idx], "score": sim}
            for sim, idx in similarities[:top_k]
        ]

    def get_count(self):
        return len(self.documents)

    def get_all(self):
        return {"documents": self.documents, "metadatas": self.metadatas, "ids": self.ids}

    def clear(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        if os.path.exists(self.persist_path):
            os.remove(self.persist_path)

    def delete_by_source(self, source_name: str) -> int:
        keep_documents, keep_embeddings, keep_metadatas, keep_ids = [], [], [], []
        deleted = 0
        for doc, emb, meta, doc_id in zip(self.documents, self.embeddings, self.metadatas, self.ids):
            if meta.get("source") == source_name:
                deleted += 1
                continue
            keep_documents.append(doc)
            keep_embeddings.append(emb)
            keep_metadatas.append(meta)
            keep_ids.append(doc_id)
        self.documents, self.embeddings, self.metadatas, self.ids = keep_documents, keep_embeddings, keep_metadatas, keep_ids
        self.save()
        return deleted


class VectorStoreManager:
    def __init__(self, persist_dir="data/chroma_db", collection_name="rag_documents"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.use_fallback = False
        self.hybrid_mode = False
        self.cloud_store = None
        self.local_store = None
        if not CHROMA_AVAILABLE:
            self.use_fallback = True
        else:
            try:
                self.cloud_store = ChromaVectorStore(persist_dir, collection_name)
                self.store = self.cloud_store
            except Exception:
                self.use_fallback = True
        if self.use_fallback:
            self.local_store = NumpyVectorStore()
            self.store = self.local_store

    def _switch_to_local_fallback(self):
        if self.hybrid_mode or not self.cloud_store:
            return
        self.local_store = NumpyVectorStore()
        self.store = self.local_store
        self.use_fallback = True
        self.hybrid_mode = True

    def add_documents(self, texts, embeddings, metadatas, ids):
        try:
            self.store.add_documents(texts, embeddings, metadatas, ids)
        except Exception as error:
            err_str = str(error).lower()
            is_quota = any(x in err_str for x in ["quota", "number of records", "numrecords", "exceeds limit"])
            if not self.hybrid_mode and is_quota:
                self._switch_to_local_fallback()
                self.local_store.add_documents(texts, embeddings, metadatas, ids)
            else:
                raise

    def query(self, query_embedding, top_k=3, source_filter=None):
        if self.hybrid_mode and self.cloud_store and self.local_store:
            cloud_results = self.cloud_store.query(query_embedding, top_k=top_k, source_filter=source_filter)
            local_results = self.local_store.query(query_embedding, top_k=top_k)
            if source_filter:
                local_results = [m for m in local_results if m["metadata"].get("source") == source_filter]
            combined = cloud_results + local_results
            combined.sort(key=lambda x: x["score"], reverse=True)
            return combined[:top_k]

        if source_filter and not self.use_fallback:
            results = self.store.collection.query(query_embeddings=[query_embedding], n_results=top_k, where={"source": source_filter})
            formatted = []
            if results and results.get("documents") and results["documents"][0]:
                for idx in range(len(results["documents"][0])):
                    dist = results["distances"][0][idx] if results.get("distances") else 0.0
                    formatted.append({
                        "text": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx] if results.get("metadatas") else {},
                        "score": float(1.0 - dist),
                    })
            return formatted
        if source_filter and self.use_fallback:
            all_matches = self.store.query(query_embedding, top_k=len(self.store.documents))
            return [m for m in all_matches if m["metadata"].get("source") == source_filter][:top_k]
        return self.store.query(query_embedding, top_k)

    def get_all(self):
        if self.hybrid_mode and self.cloud_store and self.local_store:
            cloud_data = self.cloud_store.get_all() or {}
            local_data = self.local_store.get_all() or {}
            return {
                "documents": (cloud_data.get("documents", []) or []) + (local_data.get("documents", []) or []),
                "metadatas": (cloud_data.get("metadatas", []) or []) + (local_data.get("metadatas", []) or []),
                "ids": (cloud_data.get("ids", []) or []) + (local_data.get("ids", []) or []),
            }
        return self.store.get_all()

    def get_count(self):
        if self.hybrid_mode and self.cloud_store and self.local_store:
            return self.cloud_store.get_count() + self.local_store.get_count()
        return self.store.get_count()

    def clear(self):
        if self.hybrid_mode and self.cloud_store and self.local_store:
            self.cloud_store.clear()
            self.local_store.clear()
            return
        self.store.clear()

    def delete_by_source(self, source_name: str) -> int:
        if self.hybrid_mode and self.cloud_store and self.local_store:
            return self.cloud_store.delete_by_source(source_name) + self.local_store.delete_by_source(source_name)
        return self.store.delete_by_source(source_name)
