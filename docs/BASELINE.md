# Retrieval baseline — vector-only

Measured **2026-09-08**, before any Phase 2 retrieval work. Reproduce with:

```bash
conda activate convmem
python scripts/run_eval.py                 # real embedder
python scripts/run_eval.py --json out.json
```

The benchmark builds its corpus in a throwaway directory and never touches the
private DB or index. No LLM is called — these numbers describe the **retriever
alone**.

## Setup

| | |
|---|---|
| Corpus | 52 messages, 5 conversations, **7 chunks** |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (384d, cosine on L2-normalised vectors) |
| Chunking | fixed count, 12 messages, 3 overlap |
| Retrieval | FAISS `IndexIDMap2`, vector-only. No BM25, no hybrid, no reranking |
| Questions | 44 — 38 answerable, 6 absent |
| Probe depth | k = 10 |

Questions live in [`src/evaluation/dataset.py`](../src/evaluation/dataset.py).
Expected messages are named by a distinctive substring; `resolve_expectations`
requires each to match **exactly one** message, so the benchmark fails loudly if
the synthetic corpus drifts rather than quietly scoring zero.

## Headline

| Metric | Value |
|---|---|
| Recall@1 | **68.4%** (26/38) |
| Recall@3 | **97.4%** (37/38) |
| Recall@5 | 97.4% (37/38) |
| Recall@10 | 100% (38/38) |
| MRR | **0.820** |
| Latency (mean / median / p95) | 14.4 / 14.2 / 15.5 ms |

> **Recall@5 and @10 are not meaningful on this corpus.** It produces only 7
> chunks, so k=10 returns everything and @10 is 100% by construction. Treat
> **Recall@1 and MRR** as the real signal; @5 and @10 become informative only on
> a larger corpus.

## Similarity scores

| | Value |
|---|---|
| Correct chunk (mean) | 0.3724 |
| Top *irrelevant* chunk (mean) | 0.2945 |
| Separation (correct − irrelevant) | **+0.0779** |
| Absent questions, top-1 (mean) | 0.2942 |
| Absent questions, top-1 (max) | **0.4322** |

The margin between right and wrong is ~0.08, and on 12 of 38 questions it is
**negative** — an irrelevant chunk scores higher than the correct one, which
still lands at rank 2–3.

## By category

| Category | n | R@1 | R@3 | correct | irrelevant |
|---|---|---|---|---|---|
| direct | 8 | 62.5% | 100% | 0.4409 | 0.3925 |
| paraphrase | 8 | 87.5% | 87.5% | 0.3214 | 0.2582 |
| contextual | 6 | 83.3% | 100% | 0.4809 | 0.2709 |
| multi_message | 6 | 83.3% | 100% | 0.4421 | 0.3360 |
| **exact_term** | 7 | **14.3%** | 100% | 0.2534 | 0.2845 |
| credential | 3 | 100% | 100% | 0.2465 | 0.1168 |
| absent | 6 | n/a | n/a | — | — |

**`exact_term` is the standout weakness: 14.3% Recall@1, and the only category
where the mean irrelevant score (0.2845) exceeds the mean correct score
(0.2534).** Queries like `TCS`, `CGPA`, `gaming night` are exactly what dense
retrieval is bad at and what BM25 is good at. This is the quantitative case for
prioritising the keyword half of hybrid retrieval.

## An absolute score floor cannot separate answerable from absent

`MIN_RETRIEVAL_SCORE` gates on the **top-1** score. Sweeping it:

| Floor | Answerable kept | Absent rejected | Youden |
|---|---|---|---|
| 0.10 | 37/38 | 1/6 | 0.140 |
| 0.20 | 36/38 | 1/6 | 0.114 |
| **0.25 (current)** | **30/38** | **1/6** | **−0.044** |
| 0.30 | 25/38 | 3/6 | 0.158 |
| 0.40 | 20/38 | 5/6 | 0.360 |
| 0.45 | 16/38 | 6/6 | 0.421 |

**At the shipped floor of 0.25 the Youden index is −0.044 — worse than a coin
flip.** It discards 8 answerable questions to reject a single absent one. The
best achievable floor (0.45) rejects all 6 absent questions but throws away 22
of 38 answerable ones.

No absolute cosine threshold works here, so no amount of tuning fixes it. The
gate needs a score worth thresholding on — i.e. reranking — which is why
`MIN_RETRIEVAL_SCORE` is deliberately left at 0.25 for now.

## Credential regression case

X1 `"What is my gmail password?"` retrieves the credential chunk at **rank 1**
(0.3041). **This benchmark does not reproduce the production failure**, and the
distinction matters:

- Here the credential sits in a 7-message conversation forming one chunk that
  contains the word *"gmail"* twice, so the chunk is easy to reach.
- In the real 141-message corpus it competed with 87 other messages and the
  chunk was diluted and truncated.

Retrieval succeeding also does not mean the *question* is answered: the token
itself is unlabelled, so the model can only reach it by inference. Closing that
needs exact-match retrieval. **Do not read X1's rank 1 as the bug being fixed.**

To reproduce the real failure the benchmark needs a corpus at production scale —
a follow-up worth doing before claiming Phase 2 succeeded.

## What this baseline is for

Any Phase 2 change (BM25, hybrid, reranking, semantic chunking) must be measured
against these numbers. The bar to beat:

- **Recall@1 68.4%**, MRR **0.820**, latency **14.4 ms**
- `exact_term` Recall@1 **14.3%** — the main target
- separation **+0.0779** — the number reranking should move most
