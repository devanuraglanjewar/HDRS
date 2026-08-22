# Hallucination Detection RAG System (HDRS)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Ollama-Llama%203.1%208B-black?style=flat-square" />
  <img src="https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=flat-square&logo=google" />
</p>

A full-stack **Retrieval-Augmented Generation (RAG)** system with built-in **hallucination detection and benchmarking**, featuring a side-by-side document reader and a context-locked chat interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Document Workspace** | Upload PDFs/TXT files and chat about them side-by-side with a live document reader |
| 🔍 **Context-Locked RAG Chat** | Each chat session is scoped to a single document for precise retrieval |
| 🔦 **Passage Highlighting** | Retrieved context chunks are highlighted in the document reader with cosine similarity scores |
| 🧠 **Dual LLM Support** | Llama 3.1 (8B) via Ollama as primary generator; Gemini 2.5 Flash as fallback |
| 📊 **Hallucination Benchmark** | Runs HaluEval-style evaluation to measure sentence-level and answer-level hallucination rates |
| 🏛️ **ChromaDB Persistence** | Embeddings and chunks persisted to disk via ChromaDB; NumPy JSON fallback if ChromaDB unavailable |
| 🎨 **Apple-Inspired UI** | Clean, minimal light-theme React frontend built with Vite + TailwindCSS v4 |

---

## 🏗️ Architecture

```
HDRS/
├── src/
│   ├── main.py          # FastAPI app — thin API layer and route wiring
│   ├── rag_core.py      # RAG pipeline entry point and compatibility wrapper
│   ├── benchmark.py     # HaluEval hallucination benchmarking engine
│   ├── api/
│   │   ├── app_factory.py
│   │   ├── schemas.py
│   │   └── routers/
│   │       ├── status.py
│   │       ├── ingest.py
│   │       ├── chat.py
│   │       ├── documents.py
│   │       └── benchmark.py
│   ├── core/
│   │   ├── settings.py  # Environment/config loading
│   │   ├── llm.py      # Gemini/Ollama helpers and retries
│   │   ├── chunker.py   # Sentence-aware chunking
│   │   ├── prompts.py   # Prompt templates for generation, judging, benchmark creation
│   │   └── vector_store.py # Chroma Cloud/local fallback storage adapters
│   └── services/
│       ├── rag_pipeline.py
│       └── benchmark_service.py
│
├── frontend/
│   └── src/
│       ├── App.jsx                    # Main app shell & navigation
│       └── components/
│           ├── Workspace.jsx          # Side-by-side doc reader + RAG chat
│           ├── ControlCenter.jsx      # Document ingestion & system config
│           └── BenchmarkPanel.jsx     # Hallucination evaluation dashboard
│
├── data/
│   ├── chroma_db/       # ChromaDB persistent vector store
│   └── numpy_store.json # NumPy fallback store (auto-created if ChromaDB fails)
│
├── static/              # Built frontend assets (served by FastAPI)
├── .env                 # API keys (not committed to git)
└── requirements.txt     # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com/) installed locally (optional but recommended)
- A Google API key with Gemini access

---

### 1. Clone & Set Up Python Environment

```bash
git clone <your-repo-url>
cd HDRS

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

---

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

> **Get your Google API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

---

### 3. Set Up Ollama (Recommended - No API quota limits)

```bash
# Install Ollama from https://ollama.com/
# Then pull the Llama 3.1 8B model
ollama pull llama3.1:8b

# Verify it's running
ollama list
```

The system will automatically use Ollama as the **primary generator**. If Ollama is unavailable, it falls back to the Gemini API seamlessly.

---

### 4. Development Workflow

For live UI updates while editing React files, run the Vite dev server in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then start FastAPI with the dev-server URL set so the backend redirects to Vite:

```bash
# From the project root
set HDRS_DEV_SERVER_URL=http://127.0.0.1:5173
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)** and the UI will hot-reload as you edit files in `frontend/src`.

If you want to serve the production build instead, keep the old flow:

```bash
cd frontend
npm install
npm run build
cd ..
```

---

### 5. Start the Server

```bash
# From the project root
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)** for the production build, or use the dev workflow above for hot reload.

---

## 📖 Usage Guide

### Uploading a Document

1. Go to the **Workspace** tab (default view)
2. Click the upload zone in the left sidebar, or drag & drop a PDF/TXT file
3. The document is chunked semantically, embedded using Gemini `gemini-embedding-001`, and stored in ChromaDB
4. It will appear in the **Library** list on the left

### Chatting with a Document

1. Click on a document card in the Library
2. Type your question in the chat input at the bottom center
3. The AI retrieves the most relevant chunks and generates a grounded answer
4. Matching passages are highlighted in the **Document Reader** on the right (with cosine similarity scores on hover)

### Running a Benchmark

1. Navigate to the **Benchmark** tab
2. Select the number of samples and click **Run Evaluation**
3. The system pulls questions from HaluEval QA dataset, runs RAG, and judges answers sentence-by-sentence
4. Results show donut charts for sentence-level and answer-level hallucination rates with detailed case studies

### Ingesting via Paste Text

1. Go to the **Control Center** tab
2. Switch to the **Paste Text** tab
3. Enter a document name and paste the content
4. Click **Ingest Document**

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System status — ChromaDB, Ollama, chunk count |
| `POST` | `/api/ingest/file` | Upload and ingest a PDF/TXT file |
| `POST` | `/api/ingest/text` | Ingest raw text with a source name |
| `POST` | `/api/ask` | RAG query with optional source filter |
| `GET` | `/api/documents` | Retrieve all indexed documents grouped by source |
| `POST` | `/api/db/clear` | Clear the entire vector store |
| `POST` | `/api/benchmark/run` | Trigger hallucination benchmark (background) |
| `GET` | `/api/benchmark/status` | Poll benchmark progress |
| `GET` | `/api/benchmark/results` | Retrieve latest benchmark results |

---

## 🧩 Backend Structure

The backend is now modularized by responsibility:

- `src/api/` owns the FastAPI app factory, request schemas, and routers
- `src/services/` owns the RAG pipeline and benchmark orchestration
- `src/core/` owns config, chunking, prompts, LLM helpers, and vector storage
- `src/main.py` and `src/rag_core.py` remain as compatibility entry points so the app still boots the same way

That means prompt text, storage logic, and route handlers are separated instead of being bundled into a single backend file.

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required for Gemini embeddings and fallback generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local model name for generation and evaluation |

---

## 🧰 Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
- [ChromaDB](https://www.trychroma.com/) — Vector store with cosine similarity
- [Google Generative AI](https://ai.google.dev/) — Gemini embeddings + generation fallback
- [Ollama](https://ollama.com/) — Local LLM inference (Llama 3.1 8B)
- [pypdf](https://github.com/py-pdf/pypdf) — PDF text extraction

**Frontend**
- [React 18](https://react.dev/) + [Vite](https://vitejs.dev/)
- [TailwindCSS v4](https://tailwindcss.com/)
- [GSAP](https://gsap.com/) — Animations
- [Recharts](https://recharts.org/) — Benchmark charts
- [Axios](https://axios-http.com/) — HTTP client

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
