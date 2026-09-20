from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunker:
    """Splits text into chunks using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size=1500, chunk_overlap=300):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk_text(self, text: str, source_name: str) -> list:
        langchain_docs = self.splitter.create_documents(
            texts=[text],
            metadatas=[{"source": source_name}]
        )
        
        chunks = []
        for i, doc in enumerate(langchain_docs):
            chunks.append({
                "text": doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source"),
                    "chunk_index": i
                }
            })
        return chunks
