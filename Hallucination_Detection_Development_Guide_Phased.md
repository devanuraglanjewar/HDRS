# Hallucination Detection and Reduction in RAG-based LLM Systems
## Phase-wise M.Tech Development Guide

---

## Project Summary

**Title:** Multi-Signal Hallucination Detection and Selective Mitigation in Retrieval-Augmented Generation Systems

**Core research question:** Does combining confidence, consistency, and retrieval-grounding signals detect hallucinations more reliably than any single signal alone — and can selective (segment-level) mitigation reduce hallucination rate without excessive latency cost?

**One-line pitch:** A RAG pipeline that checks its own answers sentence-by-sentence, flags risky statements, and fixes only those — instead of trusting the raw LLM output or re-verifying everything against sources (slow) or nothing at all (unreliable).

---

# PHASE 1 — "It Works and Hallucinates"

**What you show at this review:** the raw RAG pipeline answering a question live, plus your baseline hallucination rate (e.g., "22% of answers had errors").

## 1.1 The RAG Approach

Retrieval-Augmented Generation combines two systems:
1. A **retriever** — finds relevant text chunks from a document collection given a query
2. A **generator** (the LLM) — writes an answer conditioned on the retrieved chunks, not just its training data

This matters because RAG reduces hallucination but does **not** eliminate it. The LLM can still misread, over-generalize, or blend retrieved context with unrelated memorized knowledge. That gap is what Phase 2's detection layer will target — so Phase 1's job is just to prove the gap exists and measure it.

## 1.2 RAG Pipeline Components to Build

| Component | What it does | Tool |
|---|---|---|
| Document loader | Reads PDFs/text into chunks | LangChain `DocumentLoader` |
| Chunker | Splits documents into ~300–500 token pieces with overlap | LangChain `RecursiveCharacterTextSplitter` |
| Embedder | Converts chunks into vectors | `sentence-transformers` (local, free) or Gemini embeddings |
| Vector store | Stores and searches vectors | FAISS (local, free) or ChromaDB |
| Retriever | Fetches top-k relevant chunks per query | LangChain retriever wrapper over vector store |
| Generator | Produces the answer | Ollama (local Llama 3.1/Mistral) or Groq/Gemini API |

## 1.3 Chunking Strategy

- Chunk size: 300–500 tokens, with 50–100 token overlap so facts near chunk boundaries aren't split
- Use **semantic chunking** (split at sentence/paragraph boundaries, not raw character counts) — this directly affects Phase 2's "grounding" score, since a badly-cut chunk makes correct answers look ungrounded
- Store chunk metadata (source document, page number) — needed in Phase 3 so mitigation can cite exactly where a correction came from

## 1.4 Retrieval Strategy

- Start with simple **dense retrieval** (cosine similarity over embeddings), top-k = 3–5 chunks
- Optional upgrade for your literature review: **hybrid retrieval** (dense + BM25 keyword search combined) — improves recall on exact terms like drug names or numeric codes, relevant if your domain is clinical/legal text

## 1.5 Resources Needed for Phase 1

| Purpose | Tool | Notes |
|---|---|---|
| Local LLM (generation) | **Ollama** + Llama 3.1 8B or Mistral 7B | Free, no API key, no rate limits |
| Hosted LLM (comparison) | **Groq** (free API key) | Fast inference, good for a second model comparison |
| Hosted LLM (backup) | **Google Gemini API** (free tier via AI Studio) | |
| Embeddings | `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) | Free, local, no key |
| Vector store | **FAISS** (local) or **ChromaDB** | Free, no external service needed |
| RAG orchestration | **LangChain** | |
| Backend | **FastAPI** | |
| Benchmarks | **HaluEval**, **TruthfulQA** (start here) | Text-only, free/public |

## 1.6 Build Steps (Weeks 1–5)

1. Set up Ollama locally, pull Llama 3.1 8B or Mistral 7B
2. Build document ingestion: loader → chunker → embedder → FAISS index
3. Build basic RAG query pipeline (retrieve → generate)
4. Wrap in a FastAPI endpoint (`POST /ask`)
5. Select benchmark dataset(s) — start with a subset (50–100 questions) of HaluEval or TruthfulQA
6. Run baseline generation across the benchmark, no detection/mitigation
7. Label outputs: hallucinated vs. not (manual review for a subset, or use an existing labeled benchmark directly)

## 1.7 Phase 1 Deliverable

- Working RAG pipeline you can query live
- Baseline hallucination rate chart (e.g., "22% of answers had errors")
- Labeled dataset — this becomes Phase 2's training/validation data
- 2–3 example outputs: one correct answer, one hallucinated answer, side by side, with the hallucinated part highlighted

---

# PHASE 2 — "We Can Catch It"

**What you show at this review:** a hallucinated sentence getting flagged with a risk score live, plus a chart proving your combined signal beats any single detection method alone.

## 2.1 Core Idea

You combine three independent signals into one fused hallucination-risk score, computed **per sentence**, not per whole answer.

## 2.2 Signal 1 — Token-Level Confidence / Entropy

**Idea:** if the model was unsure while picking a word, that word is likely part of a hallucinated statement.

**How to compute:**
- For each generated token, get the log-probability the model assigned to it (available from most LLM APIs as `logprobs`, or directly from local models via Hugging Face `transformers`)
- Compute entropy of the probability distribution at each token position: `H = -Σ p(x) log p(x)`
- Average entropy across all tokens in a sentence → sentence-level confidence score
- High average entropy = high uncertainty = higher hallucination risk

**Reference technique:** Semantic entropy (Farquhar, Kossen, Kuhn, Gal — *Nature*, 2024, vol. 630, pp. 625–630)

## 2.3 Signal 2 — Sampling Consistency (SelfCheckGPT-style)

**Idea:** if the model is recalling a real fact, it should give roughly the same answer every time. If it's guessing, answers will vary across samples.

**How to compute:**
1. Generate the same answer 3–5 times at temperature > 0 (e.g., 0.7)
2. For each sentence in the original answer, check whether the same claim appears (semantically) in the other samples — use a lightweight NLI (Natural Language Inference) model or sentence-embedding similarity to check entailment/contradiction
3. Score = proportion of samples that agree with the claim
4. Low agreement = high hallucination risk

**Reference technique:** SelfCheckGPT (Manakul, Liusie, Gales, 2023, arXiv:2303.08896)

**Practical note:** this is the most compute-expensive signal (multiple generations per question) — this is exactly why Ollama local inference matters for your dev/testing loop, so you're not burning free-tier API quota running this repeatedly.

## 2.4 Signal 3 — Retrieval Grounding Score

**Idea:** does this sentence's claim actually appear in the retrieved source chunks?

**How to compute:**
1. For each generated sentence, compute its embedding
2. Compare against embeddings of the retrieved chunks (cosine similarity)
3. Optionally, use a Natural Language Inference model to check if the sentence is *entailed* by the retrieved text (stronger than simple similarity — catches subtle misstatements, not just topic mismatch)
4. Low similarity/entailment = ungrounded = high hallucination risk

## 2.5 Fusion Layer — Combining the Three Signals

This is your main technical contribution. Two options, pick based on time available:

**Option A (simpler, still valid for M.Tech):** Weighted sum
`risk_score = w1*entropy_score + w2*(1 - consistency_score) + w3*(1 - grounding_score)`
Tune weights (w1, w2, w3) on a small validation set.

**Option B (stronger contribution):** Train a small classifier
- Features: [entropy_score, consistency_score, grounding_score] per sentence
- Labels: hallucinated / not hallucinated (from Phase 1's labeled dataset)
- Model: Logistic Regression or a 2-layer MLP (scikit-learn or PyTorch)
- This lets you report precision/recall/F1 and directly compare fusion vs. each single signal — this comparison table is your strongest result to present

**Do not call this pipeline "training-free"** if you use Option B — say "lightweight, sentence-level fusion classifier" instead, since it requires training on labeled data.

## 2.6 Resources Needed for Phase 2

| Purpose | Tool | Notes |
|---|---|---|
| NLI model (for consistency/grounding checks) | `roberta-large-mnli` or `cross-encoder/nli-deberta-v3-base` (Hugging Face) | Free, local |
| Fusion classifier | **scikit-learn** (Logistic Regression) or **PyTorch** (small MLP) | scikit-learn is simpler to justify/explain in viva |

*(Everything else — Ollama, FAISS, LangChain, FastAPI — carries over from Phase 1.)*

## 2.7 Build Steps (Weeks 6–10)

1. Implement entropy scoring (extract logprobs from Ollama/API response)
2. Implement consistency checking (multi-sample generation + NLI comparison)
3. Implement grounding scoring (embedding similarity + optional NLI entailment against retrieved chunks)
4. Build the fusion model (start with weighted sum, then train a classifier on Phase 1's labeled data)
5. Evaluate: precision/recall/F1 for each signal alone vs. fused

## 2.8 Phase 2 Deliverable

- Detection module that scores any generated sentence live
- Comparison table: detection accuracy using entropy alone vs. consistency alone vs. grounding alone vs. fusion — fusion should win
- A confusion matrix or precision/recall table for your detector
- 2–3 example flagged outputs: original sentence + risk score + why it was flagged

---

# PHASE 3 — "We Can Fix It"

**What you show at this review:** the full live loop — question → flagged sentence → auto-corrected → clean final answer — plus before/after hallucination rate (e.g., "22% → 9%").

## 3.1 Mitigation Algorithm

Applied only to sentences whose fused risk score (from Phase 2) exceeds a threshold (tune this threshold on your validation set — there's a real precision/recall tradeoff to discuss in your report).

**Steps for a flagged sentence:**
1. **Targeted re-retrieval:** query the vector store again, but using the flagged sentence's specific claim as the query (not the original user question) — this pulls more precise supporting evidence than the original broad retrieval
2. **Check if evidence exists:**
   - If relevant evidence is found → regenerate just that sentence, conditioned explicitly on the new evidence, with a prompt like: *"Rewrite this claim using only the following source text: [evidence]. If the source does not support the claim, say so."*
   - If no relevant evidence is found even after targeted retrieval → **abstain**: replace the sentence with an explicit statement that the information wasn't found, rather than guessing again
3. Reassemble the final answer from: unflagged (original) sentences + corrected/abstained sentences

**Note on DPO:** Direct Preference Optimization is a *training-time* method — if you want to use it, it belongs in an optional offline step (periodically fine-tune your generator on pairs of [hallucinated version, corrected version] you've accumulated), not as part of the live per-question loop. Keep it out of the real-time pipeline unless you're doing a training-phase experiment separately.

## 3.2 Full Single-Question Lifecycle (End-to-End Diagram)

```
User Question
     |
     v
[Retriever] --- fetches top-k chunks from vector DB
     |
     v
[Generator] --- produces answer, split into sentences S1, S2, S3...
     |
     v
For each sentence Si:
     |
     +--> [Entropy Scorer] --------\
     +--> [Consistency Checker] ----+--> [Fusion Model] --> risk_score
     +--> [Grounding Checker] ------/
     |
     v
risk_score > threshold?
     |
    Yes -----------------------------No
     |                                |
     v                                v
[Targeted Re-retrieval]        Keep sentence as-is
     |
     v
Evidence found?
     |
   Yes---------No
    |            |
    v            v
[Regenerate    [Abstain:
 sentence      "not found
 w/ evidence]   in source"]
     |            |
     +-----+------+
           v
   [Reassemble Final Answer]
           |
           v
      Output to User
```

## 3.3 Resources Needed for Phase 3

No new tools — Phase 3 reuses the retriever, generator, and fusion model built in Phases 1–2. The only addition is the regeneration/abstention prompting logic layered on top.

## 3.4 Build Steps (Weeks 11–15)

1. Implement targeted re-retrieval for flagged sentences
2. Implement sentence-level regeneration conditioned on new evidence
3. Implement abstention path for ungrounded claims
4. Wire full pipeline: retrieve → generate → detect → mitigate → output
5. Re-run full benchmark with mitigation active, measure hallucination rate before/after
6. Measure latency overhead of mitigation step
7. Write up thesis/report + prepare final demo

## 3.5 Phase 3 Deliverable

- End-to-end working system, demoable live: question → flagged sentence in real time → re-retrieval → corrected sentence → clean output
- Before/after hallucination rate comparison (e.g., "22% → 9%")
- Latency/cost table (mitigation adds retrieval + regeneration — show it's still practical)
- One full annotated case study: a single question walked through every pipeline stage
- Final architecture diagram + thesis document

---

## Evaluation Metrics to Report (Across All Phases)

- **Hallucination rate** (%) — baseline (Phase 1) vs. after mitigation (Phase 3)
- **Detection precision/recall/F1** — per signal and fused (Phase 2)
- **Answer quality** (optional) — ROUGE/BLEU vs. reference answers, or LLM-as-judge score, to confirm mitigation doesn't degrade fluency (Phase 3)
- **Latency** — average time per question, with and without mitigation triggered (Phase 3)
- **Abstention rate** — how often the system says "not found" vs. fabricating (Phase 3)

---

## Verified References

*(checked against original sources — use these, not AI-generated citations; re-verify directly on arXiv/publisher sites yourself before submission)*

1. Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630, 625–630.
2. Huang, L., et al. (2025). A survey on hallucination in large language models. *ACM Transactions on Information Systems*, 43(2).
3. Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. arXiv:2303.08896.
4. Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv:2302.09664.
5. Abdeen, B., et al. (2025). Diversion decoding [hallucination detection via decoding-time feature classification]. In *Data and Applications Security and Privacy XXXIX* (DBSec 2025), Springer LNCS.
6. Li, H., et al. DeLask: Mitigating hallucinations in large language models via decoder layer skipping. arXiv preprint.
7. Zhang, C., & Wang, H. (2025). Hallucination detection and evaluation of large language model. arXiv:2512.22416.

---

## What to Say If a Reviewer Asks "Isn't This Just RAG?"

Plain RAG still hallucinates even with good retrieval, because the LLM can misread, misquote, or drift from retrieved context while generating. This project adds a verification layer — multi-signal detection plus selective, segment-level mitigation — that catches and corrects that drift, which plain RAG does not do on its own. Frame the contribution as **systematic integration and comparative evaluation** of existing detection techniques (semantic entropy, SelfCheckGPT-style consistency, retrieval grounding) via a fusion model, rather than a claim of fundamentally novel detection theory.
