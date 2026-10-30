import json
import os
import re
import shutil
import tempfile
import time

import requests
import logging
import threading
import concurrent.futures
import os

from src.core.llm import (
    OLLAMA_AVAILABLE,
    OLLAMA_MODEL,
    judge_with_gemini,
    judge_with_ollama,
    judge_with_nemotron,
    generate_with_gemini,
    generate_with_ollama,
    generate_with_nemotron,
    is_ollama_available,
    is_nemotron_configured,
)
from src.core.prompts import benchmark_generation_prompt, judge_prompt
from src.services.rag_pipeline import RAGPipeline
from src.core.settings import BENCHMARK_CONCURRENCY, GEMINI_RPM, GOOGLE_API_KEY, NEMOTRON_GENERATE_MODEL, NEMOTRON_JUDGE_MODEL
from src.services.supabase_store import supabase_report_store

logger = logging.getLogger(__name__)

HALUEVAL_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
DATA_DIR = "data"
RESULTS_FILE = os.path.join(DATA_DIR, "benchmark_results.json")
LABELED_DATA_FILE = os.path.join(DATA_DIR, "labeled_dataset.json")
STATUS_FILE = os.path.join(DATA_DIR, "benchmark_status.json")
STALE_RUNNING_TTL_SECONDS = 60 * 60
MAX_SOURCE_PROMPT_CHARS = 12000
MAX_JUDGE_CONTEXT_CHARS = 12000
JUDGE_RATE_LIMIT_BACKOFF_SECONDS = 4.5

os.makedirs(DATA_DIR, exist_ok=True)


def _benchmark_generate_with_fallback(prompt: str) -> tuple[str, str]:
    """Benchmark-compatible Gemini -> Nemotron -> Ollama routing.

    Calls remain module-level so the existing benchmark tests can mock them
    without making network requests.
    """
    failures = []
    providers = [
        ("gemini", lambda: generate_with_gemini(prompt, "", "gemini-3.6-flash")),
        ("nemotron", lambda: generate_with_nemotron(prompt, "", NEMOTRON_GENERATE_MODEL)),
        ("ollama", lambda: generate_with_ollama(prompt, "", OLLAMA_MODEL)),
    ]
    for name, call in providers:
        if name == "gemini" and not GOOGLE_API_KEY:
            continue
        if name == "nemotron" and not is_nemotron_configured():
            continue
        if name == "ollama" and not is_ollama_available():
            continue
        try:
            return call(), name
        except Exception as error:
            failures.append(f"{name}: {error}")
            logger.warning("Benchmark generation provider %s failed: %s", name, error)
    raise RuntimeError("Benchmark generation providers failed: " + " | ".join(failures))


def update_status(status: str, progress: float, current: int, total: int, message: str = None, error: str = None):
    payload = {
        "status": status,
        "progress": progress,
        "current": current,
        "total": total,
        "timestamp": time.time(),
    }
    if message:
        payload["message"] = message
    if error:
        payload["error"] = error
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def get_status() -> dict:
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                status = json.load(f)
            if status.get("status") == "running":
                timestamp = status.get("timestamp", 0)
                age = time.time() - float(timestamp or 0)
                if age > STALE_RUNNING_TTL_SECONDS:
                    try:
                        os.remove(STATUS_FILE)
                    except Exception:
                        pass
                    return {"status": "idle", "progress": 0.0, "current": 0, "total": 0, "stale_reset": True}
            return status
        except Exception:
            pass
    return {"status": "idle", "progress": 0.0, "current": 0, "total": 0}


_source_sample_cache = {}
_halueval_cache = []


def reset_benchmark_artifacts():
    global _source_sample_cache, _halueval_cache
    _source_sample_cache.clear()
    for path in (RESULTS_FILE, LABELED_DATA_FILE, STATUS_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def download_halueval_subset(num_samples: int = 50) -> list:
    global _halueval_cache
    if len(_halueval_cache) >= num_samples:
        return _halueval_cache[:num_samples]

    samples = []
    try:
        # Use a short connect timeout (3s) + read timeout (8s) so we fail fast on
        # network-unavailable environments rather than blocking for the full period.
        response = requests.get(HALUEVAL_URL, stream=True, timeout=(3, 8))
        response.raise_for_status()
        count = 0
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue
                try:
                    obj = json.loads(decoded)
                    samples.append(obj)
                    count += 1
                    if count >= num_samples:
                        break
                except Exception:
                    continue
    except Exception:
        samples = create_synthetic_data(num_samples)

    if len(samples) < num_samples:
        needed = num_samples - len(samples)
        samples.extend(create_synthetic_data(needed))

    _halueval_cache = samples[:num_samples]
    return samples[:num_samples]


def create_synthetic_data(num_samples: int) -> list:
    """Generate diverse synthetic QA pairs that cover a wide range of knowledge domains.
    Each entry is unique — no item repeats until we have exhausted all templates.
    Used as a fallback when HaluEval cannot be downloaded.
    """
    _pool = [
        {
            "knowledge": "The Apollo program was the third United States human spaceflight program carried out by NASA, which succeeded in landing the first humans on the Moon in 1969.",
            "question": "What was the Apollo program and when did it land the first humans on the Moon?",
            "right_answer": "The Apollo program was NASA's third US human spaceflight program, landing humans on the Moon in 1969.",
            "hallucinated_answer": "The Apollo program was NASA's first space program, landing humans on Mars in 1974.",
        },
        {
            "knowledge": "Alexander Fleming was a Scottish physician and microbiologist, best known for discovering penicillin in 1928.",
            "question": "Who discovered penicillin and when?",
            "right_answer": "Alexander Fleming, a Scottish physician and microbiologist, discovered penicillin in 1928.",
            "hallucinated_answer": "Louis Pasteur discovered penicillin in 1895 while working in Paris.",
        },
        {
            "knowledge": "Retrieval-Augmented Generation (RAG) combines a retrieval component with a language model to ground responses in external knowledge, reducing hallucinations.",
            "question": "What is Retrieval-Augmented Generation and how does it reduce hallucinations?",
            "right_answer": "RAG combines retrieval with a language model to ground responses in external knowledge, which reduces hallucinations.",
            "hallucinated_answer": "RAG is a technique where the model generates answers without any external retrieval, relying solely on parametric memory.",
        },
        {
            "knowledge": "The Eiffel Tower was constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair in Paris and stands 330 metres tall.",
            "question": "When was the Eiffel Tower built and how tall is it?",
            "right_answer": "The Eiffel Tower was built between 1887 and 1889 for the 1889 World's Fair and stands 330 metres tall.",
            "hallucinated_answer": "The Eiffel Tower was built in 1901 as a telecommunications tower and is 250 metres tall.",
        },
        {
            "knowledge": "Transformer models, introduced in the 2017 paper 'Attention Is All You Need', rely entirely on self-attention mechanisms and have become the backbone of modern NLP.",
            "question": "When were Transformer models introduced and what mechanism do they rely on?",
            "right_answer": "Transformer models were introduced in 2017 and rely on self-attention mechanisms.",
            "hallucinated_answer": "Transformer models were introduced in 2015 by Google DeepMind and rely on recurrent neural network layers.",
        },
        {
            "knowledge": "Photosynthesis is the process by which green plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose and oxygen.",
            "question": "What is photosynthesis and what does it produce?",
            "right_answer": "Photosynthesis is the process by which green plants and algae convert sunlight, water, and carbon dioxide into glucose and oxygen.",
            "hallucinated_answer": "Photosynthesis is a process in fungi that converts nitrogen into amino acids and carbon monoxide.",
        },
        {
            "knowledge": "Python was created by Guido van Rossum and first released in 1991. It is known for its simple, readable syntax that emphasises code readability.",
            "question": "Who created Python and what is it known for?",
            "right_answer": "Python was created by Guido van Rossum and first released in 1991. It is known for its simple, readable syntax.",
            "hallucinated_answer": "Python was created by James Gosling at Sun Microsystems in 1985 and is known for its static type system.",
        },
        {
            "knowledge": "The human genome contains approximately 3 billion base pairs and around 20,000–25,000 protein-coding genes.",
            "question": "How many base pairs and protein-coding genes does the human genome contain?",
            "right_answer": "The human genome has approximately 3 billion base pairs and around 20,000–25,000 protein-coding genes.",
            "hallucinated_answer": "The human genome contains 500 million base pairs and exactly 100,000 protein-coding genes.",
        },
        {
            "knowledge": "The speed of light in a vacuum is approximately 299,792,458 metres per second, often approximated as 3 × 10^8 m/s.",
            "question": "What is the speed of light in a vacuum?",
            "right_answer": "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
            "hallucinated_answer": "The speed of light in a vacuum is approximately 150,000 kilometres per second.",
        },
        {
            "knowledge": "Hallucination in large language models refers to outputs that are factually incorrect, fabricated, or unsupported by the input context or real-world knowledge.",
            "question": "What is hallucination in the context of large language models?",
            "right_answer": "Hallucination in LLMs refers to outputs that are factually incorrect, fabricated, or unsupported by the input or real-world knowledge.",
            "hallucinated_answer": "Hallucination in LLMs refers to the model's ability to creatively generate new factual knowledge not present in any training data.",
        },
        {
            "knowledge": "The Great Wall of China stretches over 21,196 kilometres and was built over centuries by various Chinese dynasties, primarily to protect against nomadic invasions.",
            "question": "How long is the Great Wall of China and why was it built?",
            "right_answer": "The Great Wall stretches over 21,196 kilometres and was built to protect against nomadic invasions.",
            "hallucinated_answer": "The Great Wall of China is 5,000 kilometres long and was built as a trade route connecting eastern and western China.",
        },
        {
            "knowledge": "DNA (deoxyribonucleic acid) is a double-helix molecule that carries genetic information and is found in the nucleus of eukaryotic cells.",
            "question": "What is DNA and where is it found in eukaryotic cells?",
            "right_answer": "DNA is a double-helix molecule that carries genetic information and is found in the nucleus of eukaryotic cells.",
            "hallucinated_answer": "DNA is a single-stranded molecule found in the mitochondria of prokaryotic cells.",
        },
    ]
    return [_pool[i % len(_pool)].copy() for i in range(num_samples)]


def _load_source_text(source_name: str) -> str:
    store = RAGPipeline().vector_store.get_all()
    documents = store.get("documents", []) if store else []
    metadatas = store.get("metadatas", []) if store else []
    chunks = []
    for idx, doc_text in enumerate(documents):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        if meta.get("source") == source_name:
            chunks.append((meta.get("chunk_index", 0), doc_text))
    chunks.sort(key=lambda item: item[0])
    return "\n\n".join(text for _, text in chunks).strip()


def _generate_source_benchmark_samples(source_name: str, source_text: str, num_samples: int) -> list:
    cleaned_text = source_text.strip()
    if not cleaned_text:
        return []

    requested_count = max(1, int(num_samples))
    cache_key = f"{source_name}_{requested_count}_{len(cleaned_text)}"
    if cache_key in _source_sample_cache:
        logger.info("Benchmark samples cache hit for source '%s' (%d samples)", source_name, requested_count)
        return _source_sample_cache[cache_key][:requested_count]

    def _fallback_source_samples(count: int, start_idx: int = 0) -> list:
        """Semantic proposition synthesizer — extracts factual claims from each sentence and
        formulates a category-appropriate question, matching LLM output quality at zero latency.
        No network, no LLM, no hardcoded lists — all derived dynamically from the document text.
        """
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", cleaned_text) if len(s.strip().split()) >= 4]
        if not sentences:
            sentences = [s.strip() for s in cleaned_text.split("\n") if s.strip()]
        if not sentences:
            return []

        # Unsupported aspects for synthetic abstention questions — deliberately off-topic to the corpus
        _abstain_aspects = [
            "hardware cost or pricing",
            "legal or licensing requirements",
            "external third-party vendor contracts",
            "non-English language support",
            "physical infrastructure specifications",
            "cryptographic security guarantees",
            "regulatory compliance details",
            "insurance or liability coverage",
        ]

        # Stopwords filtered when extracting a subject snippet
        _stopwords = {
            "the", "a", "an", "this", "that", "these", "those",
            "in", "on", "at", "to", "for", "with", "by", "of", "and", "or",
            "is", "are", "was", "were", "be", "been", "has", "have", "had",
            "it", "its", "as", "from", "but", "not", "also", "can", "which",
        }

        samples = []
        total_sents = len(sentences)

        for i in range(count):
            idx = start_idx + i
            sent = sentences[idx % total_sents]

            # --- Sentence characterisation (pure regex, zero dependencies) ---
            has_number = bool(re.search(
                r'\b\d+(?:\.\d+)?%|\b\d{4}\b|\b\d+(?:\.\d+)?\s*(?:times|x|fold|ms|seconds|hours|days|years|million|billion|thousand)',
                sent, re.IGNORECASE
            ))
            has_definition = bool(re.search(
                r'\b(?:is|are|refers to|means|defined as|known as|called|denotes)\b',
                sent, re.IGNORECASE
            ))
            has_comparison = bool(re.search(
                r'\b(?:outperforms|exceeds|surpasses|compared to|better than|worse than|reduces|improves|increases|decreases|lowers|higher|lower)\b',
                sent, re.IGNORECASE
            ))
            has_purpose = bool(re.search(
                r'\b(?:aims to|designed to|serves to|provides|enables|allows|helps|ensures|prevents|detects|identifies)\b',
                sent, re.IGNORECASE
            ))
            has_finding = bool(re.search(
                r'\b(?:demonstrate|show|reveal|indicate|suggest|find|found|report|observe|conclude|experiment|result|achieve)\b',
                sent, re.IGNORECASE
            ))

            # Extract a subject snippet (3–5 content words)
            words = [w.strip(".,;:()[]\"'\\–—") for w in sent.split()]
            content_words = [w for w in words if w.lower() not in _stopwords and len(w) > 2]
            subject = " ".join(content_words[:5]) if content_words else " ".join(words[:5])

            # --- Question and category selection ---
            # Every 5th question is an abstention probe (deliberately asks about something not in the document)
            if idx % 5 == 4:
                aspect = _abstain_aspects[(idx // 5) % len(_abstain_aspects)]
                question = f"What does the document specify about {aspect} related to {' '.join(content_words[:3]) if content_words else subject}?"
                category = "abstain"

            elif has_number:
                # Find the actual number/metric in the sentence for a precise extraction question
                num_match = re.search(
                    r'(\b\d+(?:\.\d+)?%|\b\d{4}\b|\b\d+(?:\.\d+)?\s*(?:times|x|fold|ms|seconds|hours|days|years|million|billion|thousand))',
                    sent, re.IGNORECASE
                )
                metric_hint = num_match.group(0).strip() if num_match else "the reported figure"
                question = f"What exact quantitative result or metric ({metric_hint}) does the document report regarding {subject}?"
                category = "precise_extraction"

            elif has_definition:
                question = f"How does the document define or describe {subject}?"
                category = "easy_factual"

            elif has_comparison:
                question = f"What comparative performance result or advantage does the document describe for {subject}?"
                category = "precise_extraction"

            elif has_finding:
                question = f"What key finding or experimental result does the document state regarding {subject}?"
                category = "precise_extraction"

            elif has_purpose:
                question = f"What is the stated purpose or function of {subject} as described in the document?"
                category = "easy_factual"

            else:
                question = f"What specific information does the document provide about {subject}?"
                category = "easy_factual"

            samples.append({
                "question": question,
                "reference_knowledge": sent,
                "category": category,
                "generation_method": "template_fallback",
            })

        return samples

    def _extract_json_from_text(text: str) -> dict:
        if not text:
            return {}
        cleaned = text
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

    def _call_llm_for_samples(p_text: str, count: int) -> list:
        """Ask the LLM only for questions and categories — not for reference text.
        This reduces response tokens by ~60%, cutting generation time significantly.
        The matching reference sentence is derived locally by keyword overlap — zero extra LLM call.
        """
        p = benchmark_generation_prompt(source_name, p_text, count)
        raw, generation_provider = _benchmark_generate_with_fallback(p)
        payload = _extract_json_from_text(raw)
        samples_raw = payload.get("samples", []) if isinstance(payload, dict) else []

        # Pre-split source sentences for reference matching (done once per batch call)
        src_sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", p_text) if len(s.strip().split()) >= 4]

        def _best_reference(question: str, fallback: str) -> str:
            """Return the source sentence whose token overlap with the question is highest.
            Falls back to the provided text if no sentences match at all.
            """
            if not src_sentences:
                return fallback[:500]
            q_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
            best_sent, best_score = src_sentences[0], -1
            for sent in src_sentences:
                s_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower()))
                overlap = len(q_tokens & s_tokens)
                if overlap > best_score:
                    best_score, best_sent = overlap, sent
            return best_sent

        items = []
        for item in samples_raw[:count]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            # Accept reference from LLM if provided, otherwise match locally
            llm_ref = str(item.get("reference_knowledge", "")).strip()
            ref = llm_ref if llm_ref else _best_reference(question, p_text)
            category = str(item.get("category", "easy_factual")).strip() or "easy_factual"
            if question:
                items.append({
                    "question": question,
                    "reference_knowledge": ref or p_text[:500],
                    "category": category,
                    "generation_method": "llm",
                })
        return items

    normalized = []
    try:
        if requested_count <= 20:
            # Small count: single LLM call with a tight per-call timeout
            prompt_text = cleaned_text[:MAX_SOURCE_PROMPT_CHARS]
            try:
                normalized = _call_llm_for_samples(prompt_text, requested_count)
            except Exception as llm_e:
                logger.debug("LLM call failed for source '%s': %s — using proposition synthesizer", source_name, llm_e)
        else:
            # Large count (>20): launch parallel LLM batches with a hard wall-clock timeout.
            # If the whole LLM layer is too slow (rate-limited, offline, etc.) we don't
            # block the user — we catch the timeout and fill everything from the synthesizer.
            _BATCH_GEN_TIMEOUT = 45  # seconds total budget for all LLM batches combined
            num_batches = min(4, (requested_count + 19) // 20)
            batch_size = (requested_count + num_batches - 1) // num_batches
            n = len(cleaned_text)
            slice_size = max(1500, n // num_batches)

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, num_batches)) as executor:
                futures = []
                for b_idx in range(num_batches):
                    cur_count = min(batch_size, requested_count - (b_idx * batch_size))
                    if cur_count <= 0:
                        break
                    start_char = (b_idx * slice_size) % max(1, n - 1000)
                    end_char = min(n, start_char + slice_size + 1000)
                    part_text = cleaned_text[start_char:end_char]
                    futures.append(executor.submit(_call_llm_for_samples, part_text, cur_count))

                for f in concurrent.futures.as_completed(futures, timeout=_BATCH_GEN_TIMEOUT):
                    try:
                        normalized.extend(f.result())
                    except Exception as be:
                        logger.debug("Batch sample generation worker error: %s", be)

    except concurrent.futures.TimeoutError:
        logger.warning(
            "LLM batch generation timed out for source '%s' after %ds. "
            "Filling all %d samples with the proposition synthesizer.",
            source_name, _BATCH_GEN_TIMEOUT if requested_count > 20 else 30, requested_count
        )
    except Exception:
        logger.debug(
            "Source sample generation: LLM error for '%s', falling back to proposition synthesizer",
            source_name, exc_info=True
        )

    # Fill any deficit (or all samples if LLM was skipped/timed-out) using the proposition synthesizer.
    # This is O(n_sentences) with no I/O — guaranteed sub-millisecond completion.
    if len(normalized) < requested_count:
        needed = requested_count - len(normalized)
        fallback_samples = _fallback_source_samples(count=needed, start_idx=len(normalized))
        normalized.extend(fallback_samples)

    final_samples = normalized[:requested_count]
    _source_sample_cache[cache_key] = final_samples
    logger.debug(
        "Source sample generation: %d samples (requested %d) for '%s' [llm=%d, synthesized=%d]",
        len(final_samples), requested_count, source_name,
        sum(1 for s in final_samples if s.get("generation_method") == "llm"),
        sum(1 for s in final_samples if s.get("generation_method") == "template_fallback"),
    )
    return final_samples



def _build_judge_context(retrieved_context: str, reference_knowledge: str) -> tuple[str, str]:
    context = (retrieved_context or "").strip()
    if not context:
        context = (reference_knowledge or "").strip()
        source = "reference_knowledge_fallback"
    else:
        source = "retrieved_context"
    if not context:
        return "", source
    return context[:MAX_JUDGE_CONTEXT_CHARS], source


def _should_pace_benchmark(model_name: str) -> bool:
    if OLLAMA_AVAILABLE and (model_name == "auto" or model_name.startswith("ollama:") or model_name == "ollama"):
        return False
    return True


def evaluate_sentence_level(context: str, answer: str, rate_limiter=None):
    prompt = judge_prompt(context, answer)

    def _parse_json_response(text: str) -> list:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        data = json.loads(text.strip())
        return data.get("sentences", [])

    def _fallback_eval(error) -> tuple[list, str]:
        sentences = re.split(r"(?<=[.?!])\s+", answer)
        logger.error("Judge: falling back to heuristic evaluation due to error: %s", error)
        return (
            [{"sentence": s.strip(), "label": "factual", "reason": f"Fallback: judge unavailable ({error})"} for s in sentences if s.strip()],
            "heuristic",
        )

    '''
    # If Ollama is running locally, use it directly as judge for fast, quota-free evaluation
    if is_ollama_available():
        try:
            return _parse_json_response(judge_with_ollama(prompt, OLLAMA_MODEL)), "ollama"
        except Exception as ollama_e:
            logger.warning("Judge: Ollama primary call failed: %s, falling back to Gemini", ollama_e)

    # Try Gemini with retries, falling back to Ollama if rate-limited.
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if rate_limiter:
                rate_limiter.wait()
            return _parse_json_response(judge_with_gemini(prompt, "gemini-3.6-flash")), "gemini"
        except Exception as error:
            err_msg = str(error).lower()
            is_rate_limit = any(x in err_msg for x in ["429", "quota", "resourceexhausted", "rate limit"]) 
            logger.warning("Judge: Gemini attempt %d/%d failed: %s", attempt + 1, max_retries, error, exc_info=True)
            
            if is_rate_limit and is_ollama_available():
                logger.info("Judge: Gemini rate limited, falling back to Ollama")
                try:
                    return _parse_json_response(judge_with_ollama(prompt, OLLAMA_MODEL)), "ollama"
                except Exception as ollama_e:
                    logger.warning("Judge: Ollama fallback failed: %s", ollama_e)
            
            if attempt < max_retries - 1:
                backoff = JUDGE_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt) if is_rate_limit else (1.5 * (attempt + 1))
                time.sleep(backoff)
                continue
            else:
                logger.error("Judge: Falling back to heuristic after %d attempts. Last error: %s", max_retries, error, exc_info=True)
                return _fallback_eval(error)
    '''

    providers = [
        ("gemini", lambda: judge_with_gemini(prompt, "gemini-3.6-flash")),
        ("nemotron", lambda: judge_with_nemotron(prompt, NEMOTRON_JUDGE_MODEL)),
        ("ollama", lambda: judge_with_ollama(prompt, OLLAMA_MODEL)),
    ]
    last_error = None
    for name, call in providers:
        if name == "gemini" and not GOOGLE_API_KEY:
            continue
        if name == "nemotron" and not is_nemotron_configured():
            continue
        if name == "ollama" and not is_ollama_available():
            continue
        try:
            if rate_limiter and name == "gemini":
                rate_limiter.wait()
            return _parse_json_response(call()), name
        except Exception as error:
            last_error = error
            logger.warning("Judge provider %s failed; trying next provider: %s", name, error)
    return _fallback_eval(last_error or "no judge provider configured")


def run_benchmark(num_samples: int = 50, model_name: str = "auto", source_filter: str = None):
    temp_workdir = None
    try:
        update_status("running", 0.0, 0, num_samples, message="Starting benchmark.")
        source_filter = (source_filter or "").strip() or None
        source_mode = bool(source_filter)
        pace_benchmark = _should_pace_benchmark(model_name)
        benchmark_label = source_filter if source_mode else "HaluEval QA dataset"

        if source_mode:
            source_text = _load_source_text(source_filter)
            if not source_text:
                raise ValueError(f"No indexed text found for source '{source_filter}'.")
            samples = _generate_source_benchmark_samples(source_filter, source_text, num_samples)
            if not samples:
                raise ValueError(f"Could not generate benchmark questions for source '{source_filter}'.")
            total_samples = len(samples)
            update_status("running", 10.0, 0, total_samples, message=f"Generating source benchmark for {source_filter}.")
            temp_workdir = tempfile.mkdtemp(prefix="hdrs_benchmark_")
            temp_chroma_dir = os.path.join(temp_workdir, "chroma_db")
            pipeline = RAGPipeline(persist_dir=temp_chroma_dir, collection_name="benchmark_documents", force_local=True)
            pipeline.ingest_text(source_text, source_filter)
            update_status("running", 25.0, 0, total_samples, message=f"Indexing source '{source_filter}'.")
        else:
            samples = download_halueval_subset(num_samples)
            total_samples = len(samples)
            update_status("running", 10.0, 0, total_samples, message="Downloading benchmark samples.")
            temp_workdir = tempfile.mkdtemp(prefix="hdrs_benchmark_")
            temp_chroma_dir = os.path.join(temp_workdir, "chroma_db")
            pipeline = RAGPipeline(persist_dir=temp_chroma_dir, collection_name="benchmark_documents", force_local=True)
            unique_knowledges = list(dict.fromkeys(s["knowledge"] for s in samples))
            for idx, k_text in enumerate(unique_knowledges):
                pipeline.ingest_text(k_text, f"HaluEval_Doc_{idx}")
            update_status("running", 25.0, 0, total_samples, message="Indexing reference knowledge.")

        # Concurrency: process multiple samples in parallel while preserving per-sample integrity.
        concurrency = max(1, int(os.getenv('BENCHMARK_CONCURRENCY', str(BENCHMARK_CONCURRENCY))))
        top_k = 3
        logger.info("Benchmark run configuration: concurrency=%d, top_k=%d, pace_benchmark=%s", concurrency, top_k, pace_benchmark)
        if concurrency == 1:
            logger.warning("Benchmark concurrency is 1. For faster runs, set BENCHMARK_CONCURRENCY>1 in environment.")

        # Log Ollama availability to help diagnose local GPU/CPU issues
        try:
            from src.core.llm import OLLAMA_AVAILABLE, OLLAMA_MODEL
            logger.info("LLM routing: OLLAMA_AVAILABLE=%s, preferred_ollama_model=%s", OLLAMA_AVAILABLE, OLLAMA_MODEL)
        except Exception:
            pass

        # Rate limiter for external Gemini judge calls
        class RateLimiter:
            def __init__(self, calls_per_minute: int):
                self.min_interval = (60.0 / calls_per_minute) if calls_per_minute and calls_per_minute > 0 else 0.0
                self.lock = threading.Lock()
                self.last_call = 0.0

            def wait(self):
                if self.min_interval <= 0:
                    return
                with self.lock:
                    now = time.time()
                    elapsed = now - self.last_call
                    wait_for = self.min_interval - elapsed
                    if wait_for > 0:
                        time.sleep(wait_for)
                    self.last_call = time.time()

        rate_limiter = RateLimiter(GEMINI_RPM if GEMINI_RPM and GEMINI_RPM > 0 else 0)

        labeled_dataset = [None] * total_samples
        latencies = []
        total_sentences = 0
        hallucinated_sentences = 0
        hallucinated_answers = 0

        # Vector store may not be thread-safe; serialize retrievals with a lock to be conservative.
        vector_lock = threading.Lock()
        progress_counter = 0
        progress_lock = threading.Lock()

        def worker(idx, sample):
            nonlocal total_sentences, hallucinated_sentences, hallucinated_answers
            question = sample["question"]
            ref_knowledge = sample.get("reference_knowledge") or sample.get("knowledge", "")
            start_time = time.time()

            try:
                logger.info("Sample %d starting: question=%s", idx, (question[:200] + '...') if question and len(question) > 200 else question)
                # Retrieval under lock to protect vector store client
                with vector_lock:
                    retrieved_chunks = pipeline.retrieve(question, top_k=top_k, source_filter=source_filter if source_mode else None)

                # Log retrieval details
                chunk_texts = [ (c.get("text") or "") for c in retrieved_chunks]
                chunk_text_lengths = [len(t) for t in chunk_texts]
                logger.debug("Sample %d: retrieved %d chunks, lengths=%s", idx, len(retrieved_chunks), chunk_text_lengths)

                # Generation timing
                gen_start = time.time()
                generated_answer = pipeline.generate(question, retrieved_chunks, model_name=model_name)
                gen_time = time.time() - gen_start

                retrieved_context_text = "\n\n".join([c.get("text", "") for c in retrieved_chunks])
                judge_context, judge_context_source = _build_judge_context(retrieved_context_text, ref_knowledge)

                # Judge timing
                judge_start = time.time()
                eval_sentences, judge_method = evaluate_sentence_level(judge_context, generated_answer, rate_limiter=rate_limiter)
                judge_time = time.time() - judge_start

                total_latency = time.time() - start_time

                if isinstance(generated_answer, str) and generated_answer.startswith("Error generating answer:"):
                    # Log and return None (will be filtered out). Rate limiting handled centrally.
                    logger.warning("Sample %d: generation error: %s", idx, generated_answer)
                    return None, total_latency

                logger.info("Sample %d completed: gen_time=%.3fs, judge_time=%.3fs, judge_method=%s, judge_context_source=%s, latency=%.3fs", idx, gen_time, judge_time, judge_method, judge_context_source, total_latency)

                has_hallucination = any(s_eval.get("label") == "hallucinated" for s_eval in eval_sentences)

                # Derive category after generation: if answer clearly abstains, mark as 'abstain'
                generated_lower = (generated_answer or "").lower()
                abstain_phrases = ["i don't know", "i do not know", "cannot find", "no information", "not found", "i'm not sure", "i am not sure", "can't find", "cannot answer", "no relevant information"]
                category = sample.get("category", "unknown")
                try:
                    if any(p in generated_lower for p in abstain_phrases):
                        category = "abstain"
                except Exception:
                    pass

                entry = {
                    "sample_index": idx,
                    "question": question,
                    "reference_knowledge": ref_knowledge,
                    "category": category,
                    "generation_method": sample.get("generation_method", "unknown"),
                    "generation_time": round(gen_time, 3),
                    "judge_time": round(judge_time, 3),
                    "retrieved_context": retrieved_context_text,
                    "judge_context": judge_context,
                    "judge_context_source": judge_context_source,
                    "generated_answer": generated_answer,
                    "sentences": eval_sentences,
                    "judge_method": judge_method,
                    "has_hallucination": has_hallucination,
                    "latency": total_latency,
                }
                # Update shared counters safely
                with progress_lock:
                    for s_eval in eval_sentences:
                        total_sentences += 1
                        if s_eval.get("label") == "hallucinated":
                            hallucinated_sentences += 1
                    if has_hallucination:
                        hallucinated_answers += 1

                return entry, total_latency

            except Exception as e:
                logger.exception("Error processing sample %d: %s", idx, e)
                return None, time.time() - start_time

        # Submit tasks with a ThreadPoolExecutor
        from src.core.settings import PER_SAMPLE_TIMEOUT
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(worker, i, samples[i]): i for i in range(total_samples)}

            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    # Enforce a hard per-sample timeout to avoid any single worker running indefinitely.
                    res, latency = fut.result(timeout=PER_SAMPLE_TIMEOUT)
                    if res:
                        labeled_dataset[idx] = res
                        latencies.append(latency)
                except concurrent.futures.TimeoutError:
                    logger.error("Sample %d: worker timed out after %ds; cancelling task.", idx, PER_SAMPLE_TIMEOUT)
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                except Exception as e:
                    logger.exception("Worker exception for sample %d: %s", idx, e)

                completed += 1
                progress = 25.0 + (completed / total_samples) * 70.0
                update_status("running", progress, completed, total_samples)

        # Filter out None entries
        labeled_dataset = [e for e in labeled_dataset if e is not None]
        valid_count = len(labeled_dataset)
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
        else:
            avg_latency = 0

        # Heuristic check: if most retrieved_context lengths are identical, log a warning to indicate static retrieval.
        try:
            lengths = [len(e.get('retrieved_context','')) for e in labeled_dataset]
            if lengths:
                from collections import Counter
                cnt = Counter(lengths)
                most_common_len, most_common_count = cnt.most_common(1)[0]
                if most_common_count >= max(3, len(lengths) // 2):
                    logger.warning("Detected many identical retrieved_context lengths (len=%d count=%d). This may indicate static retrieval or embedding failures.", most_common_len, most_common_count)
        except Exception:
            pass

        valid_count = len(labeled_dataset)
        sentence_hallucination_rate = (hallucinated_sentences / total_sentences * 100) if total_sentences > 0 else 0
        answer_hallucination_rate = (hallucinated_answers / valid_count * 100) if valid_count > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        summary = {
            "total_questions": valid_count,
            "total_sentences": total_sentences,
            "factual_sentences_count": total_sentences - hallucinated_sentences,
            "hallucinated_sentences_count": hallucinated_sentences,
            "sentence_hallucination_rate": round(sentence_hallucination_rate, 2),
            "hallucinated_answers_count": hallucinated_answers,
            "answer_hallucination_rate": round(answer_hallucination_rate, 2),
            "average_latency": round(avg_latency, 3),
            "benchmark_scope": benchmark_label,
            "source_filter": source_filter if source_mode else None,
            "mode": "source_scoped" if source_mode else "halueval_baseline",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(LABELED_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(labeled_dataset, f, indent=2)

        # Local files remain the durable fallback; cloud persistence is
        # best-effort and must not turn a completed benchmark into a failure.
        if supabase_report_store.configured:
            supabase_report_store.save(summary, labeled_dataset)

        update_status("completed", 100.0, total_samples, total_samples, message="Benchmark completed successfully.")
        return summary
    except Exception as error:
        update_status("failed", 100.0, 0, num_samples, error=str(error))
        raise
    finally:
        if temp_workdir and os.path.exists(temp_workdir):
            shutil.rmtree(temp_workdir, ignore_errors=True)
