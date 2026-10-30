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

# Supabase benchmark report persistence. The service-role key is backend-only;
# never expose it to the Vite frontend or commit it to source control.
SUPABASE_URL = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

# ── NVIDIA / Nemotron ──────────────────────────────────────────────────────────
# Free tier: https://build.nvidia.com  (sign up → get API key → paste below)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
# NVIDIA's OpenAI-compatible inference endpoint
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
# Nemotron is served through an OpenAI-compatible API.  The existing project
# uses NVIDIA_API_KEY for this credential; support OpenRouter-style free keys
# automatically while still allowing an explicit endpoint override.
NEMOTRON_BASE_URL = os.getenv("NEMOTRON_BASE_URL", "").strip()
if not NEMOTRON_BASE_URL:
    NEMOTRON_BASE_URL = (
        "https://openrouter.ai/api/v1"
        if NVIDIA_API_KEY.startswith("sk-or-")
        else NVIDIA_BASE_URL
    )
# Default Nemotron model for generation (70B) and judge tasks. OpenRouter's
# `:free` suffix is important: without it, a free-tier key may select a paid
# route or fail unexpectedly. NVIDIA's native endpoint uses the unsuffixed ID.
_NEMOTRON_DEFAULT_MODEL = (
    "nvidia/nemotron-3.5-lightning:free"
    if NVIDIA_API_KEY.startswith("sk-or-")
    else "nvidia/llama-3.1-nemotron-70b-instruct"
)
NEMOTRON_GENERATE_MODEL = os.getenv("NEMOTRON_GENERATE_MODEL", _NEMOTRON_DEFAULT_MODEL)
NEMOTRON_JUDGE_MODEL = os.getenv("NEMOTRON_JUDGE_MODEL", _NEMOTRON_DEFAULT_MODEL)

# ── LLM Router cascade ─────────────────────────────────────────────────────────
# How long (seconds) to skip a provider after it hits a quota/credit error.
# After the TTL expires the provider is retried automatically — no restart needed.
PROVIDER_DEGRADATION_TTL = int(os.getenv("PROVIDER_DEGRADATION_TTL", "300"))  # 5 minutes default

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
LLM_CALL_TIMEOUT = int(os.getenv("LLM_CALL_TIMEOUT", "180"))
JUDGE_CALL_TIMEOUT = int(os.getenv("JUDGE_CALL_TIMEOUT", "180"))
# Per-sample hard timeout: maximum wall-clock seconds allowed per sample before it's cancelled and recorded as failed.
PER_SAMPLE_TIMEOUT = int(os.getenv("PER_SAMPLE_TIMEOUT", "300"))
