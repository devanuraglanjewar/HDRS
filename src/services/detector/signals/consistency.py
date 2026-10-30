import re
import logging
import numpy as np
from typing import List, Optional

from src.core.llm import embed_text

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "it", "this", "that", "be",
    "have", "has", "had", "will", "would", "can", "could", "should", "may",
}


def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_sentence_consistency(
    sentence: str,
    alternative_samples: List[str],
) -> float:
    """Compute Propositional & Relational Consistency for a sentence against reference samples/evidence.

    Evaluates:
    1. Semantic Embedding Alignment: dense vector similarity against reference samples.
    2. Content-Word Token Coverage: checks key content tokens without stopword inflation.
    3. Dynamic Entity-Anchor Veracity: detects fabricated proper nouns, dates, or numbers absent from evidence.

    Returns a consistency score in [0.0, 1.0], where 1.0 = highly consistent, 0.0 = unsupported.
    """
    if not sentence or not sentence.strip():
        return 1.0

    if not alternative_samples:
        return 0.5

    sent_clean = sentence.strip()
    sent_lower = sent_clean.lower()
    sent_embedding = embed_text(sent_clean)
    sent_tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", sent_lower)
    content_tokens = [t for t in sent_tokens if t not in STOPWORDS]

    # Dynamic entity extraction: numbers, percentages, and capitalized proper nouns
    words = sent_clean.split()
    entities = []
    for idx, w in enumerate(words):
        w_clean = re.sub(r"[^a-zA-Z0-9]", "", w)
        if not w_clean:
            continue
        if re.match(r"^\d+(?:\.\d+)?%?$", w_clean):
            entities.append(w_clean.lower())
        elif idx > 0 and w_clean[0].isupper() and w_clean.lower() not in STOPWORDS:
            entities.append(w_clean.lower())

    combined_context = " ".join(alternative_samples).lower()

    # Dynamic entity-anchor verification across all samples
    if entities:
        unsupported = [e for e in entities if e not in combined_context]
        entity_penalty = len(unsupported) / len(entities)
    else:
        entity_penalty = 0.0

    sample_agreements = []
    for sample in alternative_samples:
        if not sample or not sample.strip():
            continue

        sample_lower = sample.lower()
        sample_embedding = embed_text(sample[:800])
        cos_sim = _cosine_similarity(sent_embedding, sample_embedding)

        sample_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", sample_lower))
        overlap = len(set(content_tokens).intersection(sample_tokens)) / max(1, len(content_tokens))

        # Combine dense semantic similarity (70%) and lexical content coverage (30%)
        sample_agreement = 0.70 * cos_sim + 0.30 * overlap
        sample_agreements.append(max(0.0, min(1.0, sample_agreement)))

    if not sample_agreements:
        return 0.5

    best_agreement = max(sample_agreements)
    # Penalize if the sentence invents ungrounded entities not present in the reference corpus
    adjusted = best_agreement * (1.0 - 0.60 * entity_penalty)
    calibrated = (adjusted - 0.30) / (0.80 - 0.30)
    return float(min(1.0, max(0.0, calibrated)))


def compute_consistency_risk(consistency_score: float) -> float:
    """Invert consistency into risk: low consistency = high hallucination risk."""
    return float(min(1.0, max(0.0, 1.0 - consistency_score)))

