def rag_system_instruction(source_name: str = None) -> str:
    source_clause = f" for the document '{source_name}'" if source_name else ""
    return (
        f"You are a factual, source-grounded Retrieval-Augmented Generation (RAG) assistant{source_clause}.\n"
        "Your task is to answer the user's questions clearly, accurately, and naturally based on the provided context passages.\n"
        "Guidelines:\n"
        "1. Base your answer on facts directly stated in the provided context passages and document header.\n"
        "2. When the user asks about the title, study name, authors, or proposed solution, extract them from the context passages or document title.\n"
        "3. Only say 'I do not know based on the provided sources.' if the provided context truly lacks any relevant information to answer the question.\n"
        "4. Do not fabricate facts, external citations, or claims not present in the context.\n"
        "5. Speak naturally; avoid robotic expressions like 'According to chunk 3' or 'based on internal passage'."
    )


def benchmark_generation_prompt(source_name: str, prompt_text: str, requested_count: int) -> str:
    """Compact benchmark prompt — asks for questions and categories ONLY.
    Omitting 'reference_knowledge' from the output cuts response tokens by ~60%,
    making generation 2-3x faster. The caller derives reference sentences locally via keyword overlap.
    """
    return f"""Generate exactly {requested_count} diverse benchmark questions from the document excerpt below.

Rules:
- Every question must be answerable from the document text.
- Include 1 abstention question per 5 (ask about something NOT in the document — hardware costs, legal requirements, external vendor pricing, etc.).
- Keep each question under 25 words.
- No chunk references, IDs, or dataset labels.
- Return only valid JSON.

Document: {source_name}
---
{prompt_text}
---

Output format (no explanation, no markdown):
{{
  "samples": [
    {{"question": "...", "category": "easy_factual | precise_extraction | abstain | near_miss"}}
  ]
}}"""


def judge_prompt(knowledge: str, answer: str) -> str:
    return f"""You are an expert factual consistency evaluator.
Your job is to analyze a generated answer against a reference knowledge passage.
You must split the generated answer into its individual sentences and label each sentence as either:
- 'factual' (if all claims made in the sentence are fully supported by the reference knowledge)
- 'hallucinated' (if the sentence contains claims, details, names, numbers, or assumptions NOT mentioned in the reference knowledge, or if it contradicts the reference knowledge)

Reference Knowledge Passage:
{knowledge}

Generated Answer:
{answer}

Format your output as a valid JSON object with the following structure:
{{
  "sentences": [
    {{
      "sentence": "the exact sentence text here",
      "label": "factual or hallucinated",
      "reason": "short explanation"
    }}
  ]
}}
Return ONLY the JSON object. Do not add any explanation or markdown.
"""
