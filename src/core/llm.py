import re
import time
from functools import lru_cache

from src.core.settings import GOOGLE_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL

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

try:
    if ollama_client:
        _models = ollama_client.list()
        OLLAMA_AVAILABLE = any(m.model.startswith(OLLAMA_MODEL.split(":")[0]) for m in _models.models)
    else:
        OLLAMA_AVAILABLE = False
except Exception:
    OLLAMA_AVAILABLE = False


def get_retry_delay(error: Exception) -> float:
    err_str = str(error)
    match_sec = re.search(r"Please retry in (\d+\.?\d*)s", err_str)
    if match_sec:
        return float(match_sec.group(1)) + 1.5
    match_seconds = re.search(r"seconds:\s*(\d+)", err_str)
    if match_seconds:
        return float(match_seconds.group(1)) + 1.5
    return 60.0


@lru_cache(maxsize=2048)
def _embed_text_cached(text: str) -> tuple:
    if not genai:
        return tuple([0.0] * 3072)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = _run_with_timeout(
                genai.embed_content,
                EMBED_CALL_TIMEOUT,
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document" if len(text) > 100 else "retrieval_query",
            )
            return tuple(response["embedding"])
        except TimeoutError as te:
            # Treat timeout similarly to other errors but allow retries
            err_msg = str(te).lower()
            if attempt < max_retries - 1:
                time.sleep(1.0 + attempt)
                continue
            return tuple([0.0] * 3072)
        except Exception as error:
            err_msg = str(error).lower()
            is_rate_limit = any(x in err_msg for x in ["429", "quota", "resourceexhausted", "rate limit"])
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(get_retry_delay(error))
            else:
                return tuple([0.0] * 3072)


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
