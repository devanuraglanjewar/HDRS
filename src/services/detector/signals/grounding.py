import logging
import numpy as np
from typing import List, Dict, Any, Union

from src.core.llm import embed_text

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_sentence_grounding(
    sentence: str,
    retrieved_chunks: List[Union[str, Dict[str, Any]]],
) -> float:
    """Compute the retrieval grounding score for a sentence against retrieved chunks.

    Returns a grounding score between 0.0 (completely ungrounded / unsupported claim)
    and 1.0 (fully grounded in source chunks).
    """
    if not sentence or not sentence.strip():
        return 0.0

    if not retrieved_chunks:
        return 0.0

    sent_clean = sentence.strip()
    sent_embedding = embed_text(sent_clean)
    sent_words = set(re_tokenize(sent_clean))

    chunk_scores = []
    for chunk in retrieved_chunks:
        chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        if not chunk_text.strip():
            continue

        chunk_embedding = embed_text(chunk_text[:1000])  # embed leading portion of chunk
        cos_sim = _cosine_similarity(sent_embedding, chunk_embedding)

        # Token overlap check (checks named entities and distinctive nouns)
        chunk_words = set(re_tokenize(chunk_text))
        if sent_words:
            overlap = len(sent_words.intersection(chunk_words)) / len(sent_words)
        else:
            overlap = 0.0

        # Grounding combines semantic embedding proximity (70%) and lexical overlap (30%)
        combined = 0.70 * cos_sim + 0.30 * overlap
        chunk_scores.append(combined)

    if not chunk_scores:
        return 0.0

    # Best-matching chunk represents the maximum evidence support
    best_support = max(chunk_scores)

    # Calibrate: baseline cosine similarity for non-matching English text in MiniLM is ~0.25-0.35.
    # Map [0.30, 0.85] -> [0.0, 1.0] for clear grounding discrimination
    calibrated = (best_support - 0.30) / (0.85 - 0.30)
    return float(min(1.0, max(0.0, calibrated)))


def compute_grounding_risk(grounding_score: float) -> float:
    """Invert grounding into risk: low grounding = high hallucination risk."""
    return float(min(1.0, max(0.0, 1.0 - grounding_score)))


def re_tokenize(text: str) -> List[str]:
    """Helper to tokenize text into lowercase word stems/tokens, filtering trivial stopwords."""
    import re
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "of", "with"}
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    return [t for t in tokens if t not in stopwords]
