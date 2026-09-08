# Retrieval baseline — vector-only

Measured **2026-09-08**, before any Phase 2 retrieval work. Reproduce with:

```bash
conda activate convmem
python scripts/run_eval.py --scale large    # THE baseline to beat
python scripts/run_eval.py --scale sample   # small corpus, kept for comparison
python scripts/run_eval.py --scale large --json out.json
```

**Two scales.** `sample` is the four tracked chats (7 chunks); `large` wraps the
same gold answers in a production-scale corpus (1269 messages, 144 chunks) built
deterministically by [`src/evaluation/corpus.py`](../src/evaluation/corpus.py).
Use **`large`** for any Phase 2 comparison — `sample` is too small to
discriminate, and it cannot reproduce the credential failure.

## Headline: both scales

| Metric | `sample` (7 chunks) | `large` (144 chunks) |
|---|---|---|
| Recall@1 | 68.4% | **57.9%** |
| Recall@3 | 97.4% | **73.7%** |
| Recall@5 | 97.4% *(saturated)* | **76.3%** |
| Recall@10 | 100% *(saturated)* | **92.1%** |
| MRR | 0.820 | **0.682** |
| Latency (mean) | 14.4 ms | **15.9 ms** |
| Separation | +0.0779 | **+0.0566** |
| `exact_term` R@1 | 14.3% | **0.0%** |
| `credential` R@1 | 100% | **0.0%** |

Scaling the corpus costs ~10 points of Recall@1 and moves `exact_term` and
`credential` to zero. Latency barely moves (14.4 → 15.9 ms for 20× the chunks),
because FAISS search is not the bottleneck at this size — query embedding is.

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

## Credential regression case — reproduced at `large` scale

| Question | `sample` rank | `large` rank | correct | top irrelevant |
|---|---|---|---|---|
| X1 "What is my gmail password?" | 1 | **6** | 0.2315 | 0.2686 |
| X2 "What is the gmail address I made?" | 1 | **3** | 0.2360 | 0.2750 |
| X3 "What login did I share…?" | 1 | **7** | 0.0998 | 0.1208 |

At `sample` scale the credential owns a 7-message conversation whose single
chunk says *"gmail"* twice, so it is trivially reachable — the benchmark scored
100% while production was broken. At `large` scale it fails for exactly the
production reason. The top-6 for X1:

| Rank | Score | Contains credential | What it is |
|---|---|---|---|
| 1 | 0.2686 | no | unrelated hostel chatter |
| 2 | 0.2540 | no | **distractor** — "gmail id do", "email pe check kr lo" |
| 3 | 0.2451 | no | **distractor** — "portal password reset krna pdega" |
| 4 | 0.2392 | no | unrelated |
| 5 | 0.2373 | no | *"synthetic sample message"* — talks about the account |
| **6** | **0.2315** | **yes** | the chunk holding the credential |

**Chunks that discuss credentials outrank the chunk that contains one.** That is
the whole failure in one table, and no threshold fixes it: the correct chunk is
*below* five irrelevant ones, so it is a ranking problem, not a gating problem.

Retrieval succeeding would still not mean the question is answered — the token
is unlabelled, so a model can only reach it by inference. The fix is exact-match
retrieval.

**Phase 2 acceptance test: X1 must reach rank 1 at `large` scale.**

## `large`-scale detail

| Category | n | R@1 | R@3 | R@5 | R@10 | correct | irrelevant |
|---|---|---|---|---|---|---|---|
| direct | 8 | 62.5% | 100% | 100% | 100% | 0.4409 | 0.3961 |
| paraphrase | 8 | 87.5% | 87.5% | 87.5% | 87.5% | 0.3557 | 0.2613 |
| contextual | 6 | 83.3% | 83.3% | 83.3% | 100% | 0.4809 | 0.3495 |
| multi_message | 6 | 83.3% | 100% | 100% | 100% | 0.4421 | 0.3582 |
| **exact_term** | 7 | **0.0%** | 14.3% | 28.6% | 71.4% | 0.2834 | 0.3036 |
| **credential** | 3 | **0.0%** | 33.3% | 33.3% | 100% | 0.1891 | 0.2215 |

Three questions are **never retrieved within k=10**: P5 *"When are we going away
on holiday?"*, E1 *"What is Sneha building?"*, and E6 *"TCS"*. A bare entity
query returning nothing in the top 10 of 144 chunks is the clearest possible
argument for BM25.

`exact_term` and `credential` are the only categories where the mean irrelevant
score exceeds the mean correct score.

## What this baseline is for

Any Phase 2 change (BM25, hybrid, reranking, semantic chunking) must be measured
at **`large` scale** against these numbers. The bar to beat:

- **Recall@1 57.9%**, MRR **0.682**, latency **15.9 ms**
- `exact_term` Recall@1 **0.0%** — the main target
- `credential` X1 at **rank 6** — must reach rank 1
- separation **+0.0566** — the number reranking should move most
- P5, E1, E6 currently unretrievable within k=10

Quote `sample` numbers only when comparing to the pre-existing baseline. A Phase
2 result reported at `sample` scale is not evidence.
