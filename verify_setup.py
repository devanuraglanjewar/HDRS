import os
import sys
from dotenv import load_dotenv

def main():
    print("==================================================")
    print("HDRS Phase 1 Verification Script")
    print("==================================================")
    
    # 1. Check virtual environment python
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    
    # 2. Load .env
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[-] ERROR: GOOGLE_API_KEY is not set in .env!")
        sys.exit(1)
    else:
        # Mask the key for display
        masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "..."
        print(f"[+] SUCCESS: Loaded GOOGLE_API_KEY: {masked}")

    # 3. Test Library Imports
    try:
        import chromadb
        print("[+] SUCCESS: ChromaDB is installed.")
    except ImportError:
        print("[-] WARNING: ChromaDB is missing. Fallback will be triggered.")

    try:
        import google.generativeai as genai
        print("[+] SUCCESS: google-generativeai is installed.")
    except ImportError as e:
        print(f"[-] ERROR: Failed to import google-generativeai: {e}")
        sys.exit(1)

    # 4. Initialize RAG Core and run checks
    try:
        from src.rag_core import RAGPipeline
        print("[+] SUCCESS: src/rag_core.py imports successfully.")
        
        pipeline = RAGPipeline()
        print(f"[+] SUCCESS: Initialized Vector Store (Fallback mode: {pipeline.vector_store.use_fallback})")
        
        # Test Gemini embedding
        print("Testing Gemini embedding API (models/gemini-embedding-001)...")
        emb = pipeline.get_embedding("Verify embedding API is reachable and active.")
        if len(emb) > 0 and sum(emb) != 0.0:
            print(f"[+] SUCCESS: Generated embedding vector of length {len(emb)}")
        else:
            print("[-] ERROR: Generated embedding is empty or zeroed!")
            sys.exit(1)
            
        # Test chunking
        test_text = "The quick brown fox jumps over the lazy dog. A second sentence verifies chunking bounds."
        chunks = pipeline.chunker.chunk_text(test_text, "test_doc")
        print(f"[+] SUCCESS: Chunker works. Split test text into {len(chunks)} chunks.")
        
    except Exception as e:
        print(f"[-] ERROR: Failed during pipeline verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("==================================================")
    print("[+] ALL CHECKS PASSED: Environment and RAG Core are operational!")
    print("==================================================")

if __name__ == "__main__":
    main()
