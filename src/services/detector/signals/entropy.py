import math
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_sentence_entropy(
    sentence: str,
    context: str = "",
    token_logprobs: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """Compute the uncertainty / entropy score for a sentence.

    Returns a normalized value between 0.0 (high confidence / low entropy)
    and 1.0 (high uncertainty / high entropy).

    Reference: Semantic Entropy (Farquhar et al., Nature 2024).
    """
    if not sentence or not sentence.strip():
        return 0.0

    # 1. If explicit token logprobs are provided by the LLM generation:
    if token_logprobs:
        # Filter tokens that belong to this sentence
        sentence_tokens = []
        cleaned_sent = sentence.strip().lower()
        for t in token_logprobs:
            token_text = t.get("token", "")
            if token_text.lower() in cleaned_sent:
                sentence_tokens.append(t)

        if sentence_tokens:
            entropies = []
            for t in sentence_tokens:
                # Top logprobs distribution if available
                top_logprobs = t.get("top_logprobs")
                if top_logprobs and isinstance(top_logprobs, dict):
                    # H = - sum(p * log(p))
                    probs = [math.exp(lp) for lp in top_logprobs.values() if lp is not None]
                    prob_sum = sum(probs)
                    if prob_sum > 0:
                        norm_probs = [p / prob_sum for p in probs]
                        h = -sum(p * math.log2(p + 1e-12) for p in norm_probs)
                        entropies.append(h)
                else:
                    # Single token logprob: lower logprob = higher uncertainty
                    lp = t.get("logprob")
                    if lp is not None:
                        prob = min(1.0, max(1e-6, math.exp(lp)))
                        # Binary entropy proxy
                        h = -(prob * math.log2(prob) + (1 - prob) * math.log2(max(1e-6, 1 - prob)))
                        entropies.append(h)

            if entropies:
                avg_entropy = sum(entropies) / len(entropies)
                # Normalize to [0, 1] assuming max typical token entropy ~ 3.5 bits
                return float(min(1.0, max(0.0, avg_entropy / 3.5)))

    # 2. Semantic Token Uncertainty estimation:
    # Heuristic based on lexical uncertainty markers, hedges, and syntactic complexity
    # combined with term-specificity relative to context.
    return float(_estimate_semantic_uncertainty(sentence, context))


def _estimate_semantic_uncertainty(sentence: str, context: str) -> float:
    """Estimates semantic uncertainty when raw API token logprobs are omitted.

    Evaluates:
    1. Epistemic hedging markers (e.g., 'might', 'possibly', 'perhaps', 'supposedly', 'allegedly').
    2. Numerical / named-entity specificity vs. presence in context.
    3. Structural vagueness.
    """
    text = sentence.strip().lower()
    score = 0.15  # baseline confidence

    # Epistemic hedge phrases strongly correlated with model uncertainty
    high_uncertainty_hedges = [
        r"\b(might|maybe|perhaps|possibly|probably|conceivably)\b",
        r"\b(it is believed|it is assumed|it is rumored|supposedly|allegedly)\b",
        r"\b(i think|i believe|it seems|it appears that|hard to say)\b",
        r"\b(not entirely clear|unverified|uncertain|approximate|around)\b",
    ]

    hedge_matches = 0
    for pattern in high_uncertainty_hedges:
        if re.search(pattern, text):
            hedge_matches += 1

    score += hedge_matches * 0.22

    # Specific named entities/numbers appearing without context match
    entities_or_numbers = re.findall(r"\b([A-Z][a-z]+|\d+(?:\.\d+)?%?)\b", sentence)
    if entities_or_numbers and context:
        context_lower = context.lower()
        unmatched_entities = [e for e in entities_or_numbers if e.lower() not in context_lower]
        if unmatched_entities:
            # Unseen entities in hallucinated answers usually generate elevated entropy
            score += min(0.35, len(unmatched_entities) * 0.12)

    # Clean short factual assertions have lower entropy
    if len(text.split()) < 4 and not hedge_matches:
        score = min(score, 0.20)

    return min(1.0, max(0.0, score))
