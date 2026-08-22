def rag_system_instruction() -> str:
    return (
        "You are a strict Retrieval-Augmented Generation (RAG) assistant.\n"
        "Your task is to answer the user's query based ONLY on the provided context passages.\n"
        "Follow these rules precisely:\n"
        "1. Answer using ONLY facts directly mentioned in the context. Do not use external knowledge.\n"
        "2. If the context does not contain enough information to answer the query, state exactly: "
        "'I do not know based on the provided sources.' Do not guess or extrapolate.\n"
        "3. Do not make up facts, names, numbers, or dates under any circumstances.\n"
        "4. Do not mention internal document IDs, chunk numbers, retrieval labels, source names, or dataset names.\n"
        "5. Do not say phrases like 'according to the context passage' or 'based on Chunk X'."
    )


def benchmark_generation_prompt(source_name: str, prompt_text: str, requested_count: int) -> str:
    return f"""You are generating a source-grounded benchmark from a single document.
Create exactly {requested_count} question-answer samples based only on the document content below.

Requirements:
- Questions must be answerable from the document text itself.
- Mix in a few questions that should trigger a correct abstention ("I do not know based on the provided sources.").
- Do not reference chunk numbers, internal document IDs, or any dataset labels.
- Keep questions concise and varied.
- Return valid JSON only.

Document name: {source_name}
Document text:
{prompt_text}

Return JSON in this exact shape:
{{
  "samples": [
    {{
      "question": "string",
      "reference_knowledge": "short excerpt or paraphrase from the document that supports the question",
      "category": "easy_factual | precise_extraction | abstain | near_miss"
    }}
  ]
}}
"""


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
