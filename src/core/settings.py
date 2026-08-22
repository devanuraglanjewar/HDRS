import os
from dotenv import load_dotenv

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CHROMA_HOST = os.getenv("CHROMA_HOST", "").strip()
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "").strip()
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "").strip()
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "").strip()
DEV_SERVER_URL = os.getenv("HDRS_DEV_SERVER_URL", "").strip()

# Benchmark runtime configuration (tunable via environment variables)
try:
    _cpu_count = os.cpu_count() or 2
except Exception:
    _cpu_count = 2

BENCHMARK_CONCURRENCY = int(os.getenv("BENCHMARK_CONCURRENCY", str(min(4, _cpu_count))))
# Allowed Gemini calls per minute (used to compute a minimum inter-call spacing). Set to 0 to disable spacing.
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "15"))

# Timeouts for external calls (in seconds). Tunable via env to avoid hung LLM calls.
EMBED_CALL_TIMEOUT = int(os.getenv("EMBED_CALL_TIMEOUT", "10"))
LLM_CALL_TIMEOUT = int(os.getenv("LLM_CALL_TIMEOUT", "60"))
JUDGE_CALL_TIMEOUT = int(os.getenv("JUDGE_CALL_TIMEOUT", "60"))
# Per-sample hard timeout: maximum wall-clock seconds allowed per sample before it's cancelled and recorded as failed.
PER_SAMPLE_TIMEOUT = int(os.getenv("PER_SAMPLE_TIMEOUT", "300"))
