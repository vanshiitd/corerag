# CoreRAG — Product & Technical Design Document

## 1. Overview

CoreRAG is a low-latency, hallucination-resistant retrieval-augmented generation (RAG)
system over AI-systems research literature (arXiv). It targets a specific failure mode
of naive RAG — confident, uncited, or poorly-grounded answers — by combining two-stage
retrieval, an agentic reflection loop, and mandatory per-claim citations, all measured
against a real, reproducible evaluation suite rather than anecdote.

See [`README.md`](README.md) for architecture diagrams, the full stack table, and
instructions to run it.

## 2. Goals & Non-Goals

**Goals:**
- Grounded answers: every claim traces to a numbered source passage; the system is
  instructed to abstain rather than fill gaps with unsupported claims.
- Retrieval quality validated quantitatively (hit@k, MRR, nDCG), not assumed.
- Answer quality validated via an independent LLM judge (RAGAS), not self-reported.
- Practical latency: fast enough to be genuinely usable in an interactive session on
  the target deployment (see [§6 Known Limitations](#6-known-limitations--future-work)
  for where this goal was and wasn't met).
- Reproducible and cheap to run: local embeddings/reranker, cloud LLM calls only where
  they add real value (generation, structured routing/grading).

**Non-goals (v1):**
- Knowledge-graph / multi-document relational retrieval (GraphRAG) — designed but not
  built; see [§6](#6-known-limitations--future-work).
- Propositional chunk decomposition — evaluated as an ablation arm only, not adopted
  (contextual chunking alone proved sufficient; see [§4](#4-evaluation)).
- Sub-second cold-query latency — explicitly deferred once real benchmarking showed the
  cross-encoder reranking stage alone costs ~4s p50 on portable CPU hardware; correctness
  and honest measurement were prioritized over chasing an unrealistic target.

## 3. Architecture & Design Decisions

The system has two pipelines: an offline ingestion pipeline (arXiv → parsed, chunked,
contextualized, hybrid-indexed corpus) and an online query pipeline (semantic cache →
agentic router → hybrid retrieve → rerank → reflection-gated generation). Full diagrams
are in the README; this section covers the *why* behind each major choice.

| Decision | Choice | Rationale |
| :-- | :-- | :-- |
| PDF parsing | GROBID (Docker), PyMuPDF as a per-file fallback | GROBID preserves section structure and references — the #1 driver of chunk quality for scientific PDFs, which are otherwise messy plain text. |
| Chunking | Structure-aware, per-chunk LLM contextualization | A short LLM-generated prefix per chunk (its role in the parent document) measurably improves dense retrieval on chunks that are ambiguous out of context — see the ablation in §4. |
| Retrieval | Hybrid dense + BM25 (Qdrant RRF fusion) | Dense embeddings alone miss exact-term matches (acronyms, method names); BM25 alone misses semantic paraphrase. Fusing both is the deployed default and clearly outperforms either alone (§4). |
| Reranking | Local cross-encoder, two-stage funnel (K candidates → top N) | A cross-encoder scoring full `(query, chunk)` attention is far more precise than bi-encoder cosine similarity, at the cost of being the single largest latency contributor — treated as a first-class benchmarking target, not an afterthought. |
| Reflection loop | Bounded grader + query-rewrite loop (max 2 retries), not open-ended ReAct/Reflexion | Open-ended agentic loops risk unbounded latency and cost. A hard retry cap with a graceful `low_confidence` fallback keeps worst-case latency predictable while still self-correcting on a bad first retrieval. |
| Semantic cache | Redis (RedisVL `SemanticCache`), cosine threshold 0.95 | Empirically validated against real opposite-intent query pairs (e.g. "advantages vs. disadvantages," "use vs. avoid") — all correctly land below 0.95, with margins thin enough that the threshold was deliberately *not* lowered for a higher hit rate. |
| LLM providers | OpenAI `gpt-4o-mini` for routing/grading (strict structured output); Groq Llama 3.3 for generation (low-latency streaming) | Splits by requirement: routing/grading need reliable structured JSON; generation needs fast token streaming. Both are wired through a provider-agnostic factory (`core/llm.py`) so either can be swapped by config with no code change. |
| Embeddings & reranker | Local (FastEmbed dense + BM25 sparse; `sentence-transformers` cross-encoder) | Zero marginal cost per call, fully reproducible without API keys, and removes two of the query path's three LLM calls from the cost/latency budget entirely. |

## 4. Evaluation

### 4.1 Methodology

- **Golden set**: 181 question/answer pairs synthesized from the real indexed corpus
  (RAGAS single-hop synthesis over sampled chunks), each with a recoverable ground-truth
  `chunk_id`.
- **Retrieval metrics** (`eval/retrieval_eval.py`): hit@k, MRR, nDCG@k against the
  golden set's chunk-level ground truth — deterministic, no LLM judge, fully reproducible.
- **Answer-quality metrics** (`eval/ragas_eval.py`): faithfulness, answer relevancy,
  context precision, context recall, judged by `gpt-4o-mini` via RAGAS, scored against
  real end-to-end agent-graph runs (not pre-computed contexts).
- **Ablation** (`eval/ablation.py`): a 3×2×2 design — contextualization strategy
  (none / doc-summary / per-chunk) × retrieval mode (hybrid / dense-only) × reranker
  (on / off) — on a reproducible 20-paper sample, isolating each component's real
  contribution.

Reproduce with `make eval` / `make eval-ablation` / `make eval-cache`.

### 4.2 Results

**Retrieval quality** (production config: hybrid + rerank, k=30, full 181-question set):

| Metric | Score |
| :-- | --: |
| hit@30 | 0.967 |
| MRR | 0.853 |
| nDCG@30 | 0.882 |

**Answer quality** (RAGAS, n=55 of 181 — a checkpointed, resumable run interrupted by
Groq's shared daily generation-token quota; see `eval/ragas_eval.py` for the resumable
design):

| Metric | Score |
| :-- | --: |
| Faithfulness | 0.797 |
| Context precision | 0.928 |
| Context recall | 0.932 |
| Answer relevancy | 0.046† |

Consistent with an earlier n=30 subset (0.811 / 0.884 / 0.925 / 0.032) — nearly double
the sample size, same result: faithful, well-grounded answers.

† RAGAS's `AnswerRelevancy` forces a score of 0 whenever it classifies an answer as
"noncommittal." CoreRAG's generator deliberately hedges ("the sources don't fully
establish X") rather than bluff when evidence is weak — verified in isolation, the
identical answer scored ~0.73–1.0 with the hedge removed and 0.0 with it present. This
is a documented blind spot of the metric for honesty-first systems, not a defect in
CoreRAG: faithfulness/precision/recall (which don't share this failure mode) are the
more trustworthy signal here, and are all strong.

**Ablation** (n=28 per cell, 20-paper sample):

| contextualize_strategy | mode | rerank | hit@30 | MRR | nDCG@30 |
| :-- | :-- | :-- | --: | --: | --: |
| none | hybrid | on | 0.964 | 0.830 | 0.864 |
| none | hybrid | off | 1.000 | 0.664 | 0.746 |
| none | dense | on | 0.750 | 0.682 | 0.699 |
| none | dense | off | 0.786 | 0.491 | 0.558 |
| doc_summary | hybrid | on | 0.964 | 0.830 | 0.864 |
| doc_summary | hybrid | off | 1.000 | 0.708 | 0.776 |
| doc_summary | dense | on | 0.786 | 0.694 | 0.717 |
| doc_summary | dense | off | 0.821 | 0.373 | 0.469 |
| per_chunk | hybrid | on | 0.964 | 0.830 | 0.864 |
| per_chunk | hybrid | off | 1.000 | 0.719 | 0.787 |
| per_chunk | dense | on | 0.857 | 0.729 | 0.762 |
| per_chunk | dense | off | 0.893 | 0.495 | 0.581 |

**Key findings:**

1. **Hybrid retrieval clearly, consistently beats dense-only** at every
   contextualization strategy and reranker setting — BM25's exact-term matching
   recovers cases dense embeddings alone miss on these factoid-style questions.
2. **Reranking substantially improves the ranking quality that actually reaches the
   user.** hit@30 looks flat with reranking on because the real deployed pipeline cuts
   the candidate pool to top-5 before scoring; on that real comparison, MRR jumps
   substantially every time reranking is on (e.g. 0.830 vs. 0.664 off).
3. **Contextualization's benefit is real but currently masked in the full pipeline.**
   Hybrid+rerank numbers are identical across strategies, because the reranker scores
   raw chunk text, not the context-prefixed embedding input — but dense-only retrieval
   shows a genuine +10.7pp hit@30 improvement from no context to per-chunk context,
   matching the expected effect of contextual retrieval on embedding quality. The
   benefit exists; hybrid fusion currently performs near-ceiling regardless.

**Semantic cache**: a repeat query returns in **~28ms** vs. **~12.8s** for a fresh
graph run — a **~450× speedup**, with zero LLM/retrieval calls on the hit path
(verified via absent trace log lines, not wall-clock feel alone).

## 5. Engineering Highlights

A curated selection of the harder problems this project's build surfaced and how they
were resolved — full detail lives in commit history.

- **Cross-encoder latency reality check.** ONNX/INT8 quantization was assumed to be the
  latency win for CPU-bound reranking; real benchmarking on Apple Silicon showed
  PyTorch's native CPU backend consistently *beating* ONNX Runtime across every
  quantization variant tested — the opposite of the initial hypothesis. The mode
  default was changed on the strength of that data rather than the original assumption.
- **Local-to-cloud latency divergence.** The same reranker config that ran in ~4s p50
  locally took 60–92s per batch on Google Cloud Run's shared vCPU tier — a ~15–25×
  slowdown. Three hypotheses were tested in turn (switching inference backend, capping
  thread counts to match the CPU allocation) before confirming neither closed the gap;
  the root cause is most likely fundamental per-core throughput differences on
  burstable cloud CPU vs. dedicated Apple Silicon cores for this specific workload —
  documented honestly as an open question rather than papered over.
- **Container image size, 22.9GB → 4.16GB.** Root-caused to two issues: PyPI's default
  Linux `torch` wheel bundling the full CUDA toolkit for a CPU-only workload (fixed via
  an explicit CPU-only wheel index), and a `chown -R` after large install layers forcing
  a full copy-up under overlayfs (fixed by creating the runtime user before any
  install/download step).
- **Upstream library compatibility bugs in RAGAS**, found and root-caused rather than
  worked around blindly: a broken transitive import chain (patched via a minimal module
  stub), a blocking synchronous telemetry call inside an async code path (fixed via the
  library's own documented opt-out, ~30x latency improvement), and an event-loop-per-stage
  bug causing intermittent connection errors under its default sync entry point (fixed
  by wrapping the call in a single outer event loop).
- **Semantic cache threshold, validated not assumed.** Rather than trusting a plausible-
  sounding cosine threshold, real opposite-intent query pairs were embedded and measured
  against the production embedder before locking the value in — including one pair that
  landed within 0.001 of the cutoff, informing a decision to keep the threshold
  conservative rather than raise the cache hit rate.
- **Ingestion-scale memory tuning.** GROBID's default memory limit caused a silent JVM
  OOM partway through a 150-PDF run, degrading over half the corpus to a lower-quality
  fallback parser without raising a visible error — caught via output auditing, not
  trusting a "successful" exit code, and fixed with a higher container memory limit.

## 6. Known Limitations & Future Work

- **Cloud deployment latency.** The backend runs correctly on Google Cloud Run against
  managed Qdrant/Redis, but real query latency there (100–240s) is far above the local
  benchmark (12–24s) for reasons not fully root-caused (see §5). Local execution is the
  recommended way to run and evaluate this project today; see [`README.md`](README.md#deployment).
- **Partial answer-quality coverage.** RAGAS answer-quality scoring covers 55 of 181
  golden questions, limited by a shared daily LLM-generation quota. The evaluation
  harness is checkpointed and resumable (`eval/ragas_eval.py`) — re-running it extends
  coverage with zero repeated cost on already-scored questions.
- **GraphRAG / relational retrieval.** A knowledge-graph-backed retrieval path (entity
  extraction at ingestion, a router branch choosing vector vs. graph retrieval for
  relational/trend-style queries) was scoped but not built, to keep v1 focused on making
  vector retrieval genuinely strong before adding a second retrieval paradigm.

## License

[MIT](LICENSE)
