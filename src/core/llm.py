import re
import time
import logging
from typing import Dict, Tuple
from functools import lru_cache

import requests

from src.core.settings import (
    GOOGLE_API_KEY,
    NVIDIA_API_KEY,
    NEMOTRON_BASE_URL,
    NEMOTRON_GENERATE_MODEL,
    NEMOTRON_JUDGE_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PROVIDER_DEGRADATION_TTL,
)

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import ollama as ollama_client
except ImportError:
    ollama_client = None

if GOOGLE_API_KEY:
    if genai:
        genai.configure(api_key=GOOGLE_API_KEY)

from src.core.settings import EMBED_CALL_TIMEOUT, LLM_CALL_TIMEOUT, JUDGE_CALL_TIMEOUT
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


def _run_with_timeout(func, timeout, *args, **kwargs):
    # Run a blocking function in a thread with a timeout and bubble up exceptions.
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeoutError:
            fut.cancel()
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")

_ollama_cache = {"val": False, "ts": 0.0}

def is_ollama_available() -> bool:
    now = time.time()
    if now - _ollama_cache["ts"] < 4.0:
        return _ollama_cache["val"]
    if not ollama_client:
        _ollama_cache["val"] = False
        _ollama_cache["ts"] = now
        return False
    try:
        _models = ollama_client.list()
        val = any(m.model.startswith(OLLAMA_MODEL.split(":")[0]) for m in _models.models)
        _ollama_cache["val"] = val
        _ollama_cache["ts"] = now
        return val
    except Exception:
        _ollama_cache["val"] = False
        _ollama_cache["ts"] = now
        return False

OLLAMA_AVAILABLE = is_ollama_available()

def __getattr__(name: str):
    if name == "OLLAMA_AVAILABLE":
        return is_ollama_available()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_retry_delay(error: Exception) -> float:
    err_str = str(error)
    match_sec = re.search(r"Please retry in (\d+\.?\d*)s", err_str)
    if match_sec:
        return float(match_sec.group(1)) + 1.5
    match_seconds = re.search(r"seconds:\s*(\d+)", err_str)
    if match_seconds:
        return float(match_seconds.group(1)) + 1.5
    return 60.0


_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Fast local cache lookup to avoid network ping delays
            try:
                _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            except Exception:
                _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            pass
    return _embedding_model

@lru_cache(maxsize=2048)
def _embed_text_cached(text: str) -> tuple:
    model = _get_embedding_model()
    if not model:
        return tuple([0.0] * 384)
    
    try:
        embedding = model.encode(text)
        return tuple(embedding.tolist())
    except Exception as error:
        return tuple([0.0] * 384)

def embed_text(text: str) -> list:
    return list(_embed_text_cached((text or "").strip()))


def generate_with_ollama(prompt: str, system_instruction: str, model_name: str) -> str:
    if not ollama_client:
        raise RuntimeError("Ollama client is not installed.")

    def _call():
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2},
        )
        return response.message.content

    return _run_with_timeout(_call, LLM_CALL_TIMEOUT)


def generate_with_gemini(prompt: str, system_instruction: str, model_name: str) -> str:
    if not genai:
        raise RuntimeError("google.generativeai is not installed.")

    def _call():
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2),
        )
        return response.text

    return _run_with_timeout(_call, LLM_CALL_TIMEOUT)


def judge_with_ollama(prompt: str, model_name: str) -> str:
    if not ollama_client:
        raise RuntimeError("Ollama client is not installed.")

    def _call():
        response = ollama_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return response.message.content

    return _run_with_timeout(_call, JUDGE_CALL_TIMEOUT)


def judge_with_gemini(prompt: str, model_name: str) -> str:
    if not genai:
        raise RuntimeError("google.generativeai is not installed.")

    def _call():
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return response.text

    return _run_with_timeout(_call, JUDGE_CALL_TIMEOUT)


# ---------------------------------------------------------------------------
# Provider cascade
# ---------------------------------------------------------------------------
# The cascade deliberately lives below the RAG and benchmark layers so every
# caller gets identical behavior: Gemini -> managed Nemotron -> Ollama.
_provider_state: Dict[str, Dict[str, object]] = {
    "gemini": {"degraded_until": 0.0, "last_error": ""},
    "nemotron": {"degraded_until": 0.0, "last_error": ""},
    "ollama": {"degraded_until": 0.0, "last_error": ""},
}


def _provider_is_configured(provider: str) -> bool:
    if provider == "gemini":
        return bool(GOOGLE_API_KEY and genai)
    if provider == "nemotron":
        return bool(NVIDIA_API_KEY and NEMOTRON_BASE_URL and requests)
    if provider == "ollama":
        return is_ollama_available()
    return False


def _provider_is_degraded(provider: str) -> bool:
    return time.time() < float(_provider_state[provider]["degraded_until"])


def _mark_provider_failure(provider: str, error: Exception):
    message = str(error)
    lowered = message.lower()
    # Quota/rate-limit/outage/auth failures are usually persistent for a short
    # period.  A circuit breaker prevents burning free-tier credits repeatedly.
    persistent = any(token in lowered for token in (
        "429", "quota", "rate limit", "resourceexhausted", "credit",
        "timeout", "timed out", "connection", "unavailable", "401", "403",
        "502", "503", "504",
    ))
    if persistent:
        _provider_state[provider]["degraded_until"] = time.time() + PROVIDER_DEGRADATION_TTL
    _provider_state[provider]["last_error"] = message[:300]


def _mark_provider_success(provider: str):
    _provider_state[provider]["degraded_until"] = 0.0
    _provider_state[provider]["last_error"] = ""


def _nemotron_chat(prompt: str, system_instruction: str, model_name: str, timeout: int) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    response = requests.post(
        f"{NEMOTRON_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"model": model_name, "messages": messages, "temperature": 0.2},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Nemotron returned an unexpected response: {payload}") from error
    if not content:
        raise RuntimeError("Nemotron returned an empty response")
    return content


def generate_with_nemotron(prompt: str, system_instruction: str, model_name: str = None) -> str:
    return _nemotron_chat(prompt, system_instruction, model_name or NEMOTRON_GENERATE_MODEL, LLM_CALL_TIMEOUT)


def judge_with_nemotron(prompt: str, model_name: str = None) -> str:
    return _nemotron_chat(prompt, "", model_name or NEMOTRON_JUDGE_MODEL, JUDGE_CALL_TIMEOUT)


def is_nemotron_configured() -> bool:
    return bool(NVIDIA_API_KEY and NEMOTRON_BASE_URL)


def _requested_provider(model_name: str) -> Tuple[str, str]:
    value = (model_name or "auto").strip()
    lowered = value.lower()
    for prefix, provider in (("gemini:", "gemini"), ("nemotron:", "nemotron"), ("ollama:", "ollama")):
        if lowered.startswith(prefix):
            return provider, value[len(prefix):] or None
    if lowered in ("auto", ""):
        return "auto", None
    if lowered in ("gemini", "gemini-api") or lowered.startswith("gemini-"):
        return "gemini", value
    if lowered in ("nemotron", "nvidia", "nvidia-nemotron") or "nemotron" in lowered:
        return "nemotron", None if lowered in ("nemotron", "nvidia", "nvidia-nemotron") else value
    if lowered == "ollama":
        return "ollama", None
    return "gemini", value


def _call_provider(provider: str, prompt: str, system_instruction: str, model_name: str, judge: bool):
    if judge:
        if provider == "gemini":
            return judge_with_gemini(prompt, model_name or "gemini-3.6-flash")
        if provider == "nemotron":
            return judge_with_nemotron(prompt, model_name)
        return judge_with_ollama(prompt, model_name or OLLAMA_MODEL)
    if provider == "gemini":
        return generate_with_gemini(prompt, system_instruction, model_name or "gemini-3.6-flash")
    if provider == "nemotron":
        return generate_with_nemotron(prompt, system_instruction, model_name)
    return generate_with_ollama(prompt, system_instruction, model_name or OLLAMA_MODEL)


def call_with_fallback(prompt: str, system_instruction: str = "", model_name: str = "auto", judge: bool = False) -> Tuple[str, str, str]:
    """Call one provider and return (text, provider, resolved_model).

    Auto mode follows the requested order. Explicit provider/model selections
    remain explicit, which makes the UI useful for diagnostics and benchmarking.
    """
    requested, requested_model = _requested_provider(model_name)
    providers = ["gemini", "nemotron", "ollama"] if requested == "auto" else [requested]
    failures = []
    for provider in providers:
        if not _provider_is_configured(provider):
            failures.append(f"{provider}: not configured or unavailable")
            continue
        if _provider_is_degraded(provider):
            failures.append(f"{provider}: temporarily skipped after a prior outage/quota error")
            continue
        resolved_model = requested_model
        if provider == "nemotron" and not resolved_model:
            resolved_model = NEMOTRON_JUDGE_MODEL if judge else NEMOTRON_GENERATE_MODEL
        if provider == "ollama" and not resolved_model:
            resolved_model = OLLAMA_MODEL
        if provider == "gemini" and not resolved_model:
            resolved_model = "gemini-3.6-flash"
        try:
            text = _call_provider(provider, prompt, system_instruction, resolved_model, judge)
            _mark_provider_success(provider)
            return text, provider, resolved_model
        except Exception as error:
            _mark_provider_failure(provider, error)
            failures.append(f"{provider}: {error}")
            logger.warning("LLM provider %s failed; trying next provider: %s", provider, error)
            if requested != "auto":
                break
    raise RuntimeError("All configured LLM providers failed. " + " | ".join(failures))


def provider_status() -> dict:
    now = time.time()
    return {
        name: {
            "configured": _provider_is_configured(name),
            "degraded": now < float(state["degraded_until"]),
            "retry_at": float(state["degraded_until"]) if now < float(state["degraded_until"]) else None,
            "last_error": state["last_error"],
        }
        for name, state in _provider_state.items()
    }
