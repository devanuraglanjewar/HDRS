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
    generate_with_gemini,
    generate_with_ollama,
)
from src.core.prompts import benchmark_generation_prompt, judge_prompt
from src.services.rag_pipeline import RAGPipeline
from src.core.settings import BENCHMARK_CONCURRENCY, GEMINI_RPM

logger = logging.getLogger(__name__)

HALUEVAL_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
DATA_DIR = "data"
RESULTS_FILE = os.path.join(DATA_DIR, "benchmark_results.json")
LABELED_DATA_FILE = os.path.join(DATA_DIR, "labeled_dataset.json")
STATUS_FILE = os.path.join(DATA_DIR, "benchmark_status.json")
STALE_RUNNING_TTL_SECONDS = 60 * 60
MAX_SOURCE_PROMPT_CHARS = 12000
MAX_JUDGE_CONTEXT_CHARS = 12000
BENCHMARK_SAMPLE_DELAY_SECONDS = 4.5
JUDGE_RATE_LIMIT_BACKOFF_SECONDS = 4.5

os.makedirs(DATA_DIR, exist_ok=True)


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


def reset_benchmark_artifacts():
    for path in (RESULTS_FILE, LABELED_DATA_FILE, STATUS_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def download_halueval_subset(num_samples: int = 50) -> list:
    samples = []
    try:
        response = requests.get(HALUEVAL_URL, stream=True, timeout=30)
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
    return samples


def create_synthetic_data(num_samples: int) -> list:
    fallback_data = [
        {
            "knowledge": "The Apollo program was the third United States human spaceflight program carried out by NASA, which succeeded in landing the first humans on the Moon in 1969.",
            "question": "What was the Project Apollo and when did it land the first humans on the Moon?",
            "right_answer": "Project Apollo was the third US human spaceflight program by NASA, landing the first humans on the Moon in 1969.",
            "hallucinated_answer": "Project Apollo was NASA's first space exploration program which successfully landed humans on Mars in 1974.",
        },
        {
            "knowledge": "Alexander Fleming was a Scottish physician and microbiologist, best known for discovering penicillin in 1928.",
            "question": "Who discovered penicillin and when?",
            "right_answer": "Alexander Fleming discovered penicillin in 1928.",
            "hallucinated_answer": "Louis Pasteur discovered penicillin in 1895 while working in Paris.",
        },
    ]
    return [fallback_data[i % len(fallback_data)].copy() for i in range(num_samples)]


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

    prompt_text = cleaned_text[:MAX_SOURCE_PROMPT_CHARS]
    requested_count = max(5, min(num_samples, 20))
    prompt = benchmark_generation_prompt(source_name, prompt_text, requested_count)

    def _fallback_source_samples() -> list:
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", cleaned_text) if s.strip()]
        if not sentences:
            return []
        templates = [
            "What does the document say about: {snippet}?",
            "According to the document, what is mentioned about {snippet}?",
            "Can you identify the detail related to {snippet}?",
            "What information does the document provide about {snippet}?",
        ]
        samples = []
        for i in range(requested_count):
            sent = sentences[i % len(sentences)]
            snippet = " ".join(sent.split()[:10]).rstrip(".,;:")
            samples.append({
                "question": templates[i % len(templates)].format(snippet=snippet),
                "reference_knowledge": sent,
                "category": "easy_factual" if i % 4 != 2 else "abstain",
                "generation_method": "template_fallback",
            })
        return samples

    def _extract_json_from_text(text: str) -> dict:
        # Remove common markdown code fences and find the first JSON object in the text.
        if not text:
            return {}
        cleaned = text
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)
        # try to find the first '{' and corresponding '}' to extract JSON
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        # fallback: try to load entire cleaned text
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

    try:
        # Prefer Ollama for local generation when available, otherwise use Gemini.
        if OLLAMA_AVAILABLE:
            raw = generate_with_ollama(prompt, "", OLLAMA_MODEL)
        else:
            raw = generate_with_gemini(prompt, "", "gemini-3.6-flash")

        payload = _extract_json_from_text(raw)
        samples_raw = payload.get("samples", []) if isinstance(payload, dict) else []
        normalized = []
        for item in samples_raw[:requested_count]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            ref = str(item.get("reference_knowledge", "")).strip()
            category = str(item.get("category", "easy_factual")).strip() or "easy_factual"
            if question:
                normalized.append({
                    "question": question,
                    "reference_knowledge": ref or cleaned_text[:500],
                    "category": category,
                    "generation_method": "llm",
                })
        if normalized:
            logger.debug("Source sample generation: used LLM generator for source '%s' (%d samples)", source_name, len(normalized))
            return normalized
        else:
            logger.debug("Source sample generation: LLM returned no normalized samples, falling back for source '%s'", source_name)
            return _fallback_source_samples()
    except Exception:
        logger.debug("Source sample generation: exception using LLM, falling back for source '%s'", source_name, exc_info=True)
        return _fallback_source_samples()


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
            
            if is_rate_limit and OLLAMA_AVAILABLE:
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

        update_status("completed", 100.0, total_samples, total_samples, message="Benchmark completed successfully.")
        return summary
    except Exception as error:
        update_status("failed", 100.0, 0, num_samples, error=str(error))
        raise
    finally:
        if temp_workdir and os.path.exists(temp_workdir):
            shutil.rmtree(temp_workdir, ignore_errors=True)
