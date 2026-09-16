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

> **Note (2026-09-16).** The vector baseline's Recall@3 and MRR were re-measured
> after the credential conversation's message text was replaced during a git
> history scrub (real phrasing removed). Recall@1/@5/@10 are unchanged, and the
> BM25, RRF and cross-encoder numbers are unchanged. Only the dense retriever is
> sensitive to that wording, because only it embeds the text.

## Headline: both scales

| Metric | `sample` (7 chunks) | `large` (144 chunks) |
|---|---|---|
| Recall@1 | 68.4% | **57.9%** |
| Recall@3 | 97.4% | **71.1%** |
| Recall@5 | 97.4% *(saturated)* | **76.3%** |
| Recall@10 | 100% *(saturated)* | **92.1%** |
| MRR | 0.820 | **0.677** |
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
| 5 | 0.2373 | no | the chunk that *talks about* setting up the account |
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

## Experiment 1 — BM25 (lexical), standalone

Measured **2026-09-09**, `large` scale, identical 44 questions.
`python scripts/run_eval.py --scale large --retriever bm25`

| Metric | Vector | BM25 | Δ |
|---|---|---|---|
| Recall@1 | 57.9% | **65.8%** | **+7.9** |
| Recall@3 | 71.1% | **81.6%** | **+10.5** |
| Recall@5 | 76.3% | **86.8%** | **+10.5** |
| Recall@10 | **92.1%** | 89.5% | −2.6 |
| MRR | 0.677 | **0.749** | **+0.067** |
| Latency (mean) | 15.9 ms | **2.9 ms** | **5.5× faster** |

| Category | n | Vector R@1 | BM25 R@1 | Δ |
|---|---|---|---|---|
| direct | 8 | 62.5% | **87.5%** | +25.0 |
| paraphrase | 8 | **87.5%** | 50.0% | **−37.5** |
| contextual | 6 | 83.3% | 83.3% | 0 |
| multi_message | 6 | **83.3%** | 50.0% | **−33.3** |
| **exact_term** | 7 | 0.0% | **85.7%** | **+85.7** |
| **credential** | 3 | 0.0% | 0.0% | 0 |

### The four target questions

| Q | Vector rank | BM25 rank |
|---|---|---|
| E6 `TCS` | never in top 10 | **1** |
| E1 "What is Sneha building?" | never in top 10 | **1** |
| P5 "When are we going away on holiday?" | never in top 10 | **2** |
| X1 "What is my gmail password?" | 6 | **never retrieved** |

BM25 rescues all three lexical misses outright. E7 `CGPA` and E5 `gaming night`
also go to rank 1.

### BM25 makes the credential case worse

X1 goes from rank 6 to **no result at all**, and X3 likewise. The cause is
structural, not tuning: the credential is a bare token, so the query terms
*gmail* and *password* share nothing with the chunk holding it — while the
password distractors contain both terms and rank confidently. Dense retrieval at
least placed the right chunk 6th on weak topical similarity; BM25 requires a
term in common and there is none.

**Neither retriever can answer X1, so fusing them will not answer it either.**

### Where BM25 loses

`paraphrase` (−37.5) and `multi_message` (−33.3) are the cost of having no
synonym knowledge. Two questions score zero on every chunk and return nothing:
P4 "Which firms are visiting campus to recruit?" (corpus says *"coming for the
pre-placement talk"*) and P8 "Is the pay any good?" (corpus says *"stipend"*).

Contributing to this: BM25Okapi's IDF is exactly 0 for a term in half the
chunks and negative beyond, so common-word queries can score zero everywhere.

### Scores are not comparable

Cosine is bounded in [−1, 1]; BM25 is an unbounded corpus-relative sum (max
observed 20.8). `MIN_RETRIEVAL_SCORE=0.25` is meaningless against BM25 scores —
it keeps 36/38 answerable and rejects 0/6 absent. Any fusion must combine
**ranks** (e.g. RRF) or normalise per-searcher; naive score addition would let
BM25 dominate entirely. The abstain gate is unchanged.

### Verdict

BM25 does exactly what the baseline predicted it would — it fixes `exact_term`
(0% → 85.7%) and every lexical miss — and it is complementary rather than
superior: it wins where dense retrieval fails and fails where dense retrieval
wins. It does **not** fix the credential failure that started this work.

## Experiment 2 — RRF hybrid (vector + BM25), rank fusion

Measured **2026-09-09**, `large` scale, identical 44 questions, `RRF_K=60`,
`RRF_CANDIDATES=50`.
`python scripts/run_eval.py --scale large --retriever rrf`

| Metric | Vector | BM25 | **RRF** |
|---|---|---|---|
| Recall@1 | 57.9% | **65.8%** | **65.8%** |
| Recall@3 | 71.1% | 81.6% | **89.5%** |
| Recall@5 | 76.3% | 86.8% | **92.1%** |
| Recall@10 | 92.1% | 89.5% | **92.1%** |
| MRR | 0.677 | 0.749 | **0.779** |
| Latency (mean) | 15.9 ms | **2.9 ms** | 21.0 ms |
| Separation | +0.0566 | +2.4848 | +0.0035 |

| Category R@1 | Vector | BM25 | RRF |
|---|---|---|---|
| direct | 62.5% | **87.5%** | 75.0% |
| paraphrase | **87.5%** | 50.0% | **87.5%** |
| contextual | 83.3% | 83.3% | 83.3% |
| multi_message | **83.3%** | 50.0% | 66.7% |
| exact_term | 0.0% | **85.7%** | 42.9% |
| **credential** | 0.0% | 0.0% | **0.0%** |

RRF is the best retriever at depth: Recall@3 +15.8 over vector and +7.9 over
BM25, Recall@5 92.1%, best MRR. It recovers `paraphrase` to the vector level
(87.5%) *while* keeping `exact_term` far above vector (42.9% vs 0.0%) — the
complementarity held. Every category reaches 100% by Recall@3 except
`paraphrase` and `credential`.

| Question | Vector | BM25 | RRF |
|---|---|---|---|
| E6 `TCS` | never | 1 | **1** |
| E1 "What is Sneha building?" | never | 1 | **2** |
| P5 "…going away on holiday?" | never | 2 | **4** |
| P4 "Which firms…recruit?" | 1 | never | **1** |
| P8 "Is the pay any good?" | 1 | never | **1** |
| **X1 "What is my gmail password?"** | **6** | never | **never** |

### RRF makes the credential case worse — and it is not a reach problem

The required verification, run directly against the candidate sets:

```
QUERY: 'What is my gmail password?'   (RRF_K=60, candidates=50)
  vector :  50 candidates | credential at rank(s) [6, 27]
  bm25   :  27 candidates | credential at rank(s) ABSENT
  -> credential chunk IS in the fused candidate union: True
  -> after fusion it lands at rank 28  (score 0.01515, provenance {'vector': 6})
```

**The credential chunk WAS in the candidate set.** Fusion had it and demoted it
from vector's rank 6 to 28. The cause is arithmetic, not tuning:

```
best possible single-retriever score = 1/(60 + 1)  = 0.016393
worst possible two-retriever score   = 2/(60 + 50) = 0.018182
```

At `RRF_K=60` with a 50-deep candidate list, **any chunk both retrievers return
outranks any chunk only one returns, whatever their positions.** The credential
is visible to vector alone, so it can never beat the crowd of chunks the two
retrievers agree on. All three credential questions go to 0.0% at every K,
including Recall@10, where vector alone had X1@6, X2@3, X3@7.

This is what RRF means, not a defect — but it is a real trade-off, pinned in
`test_agreement_strictly_dominates_rank_at_the_default_settings`. The dominance
disappears below roughly K=50 at this depth (K=30: single@1 0.0323 vs
double@50,50 0.0250).

**RRF does not fix the credential problem. It regresses it.**

### Scores after fusion

RRF scores are ~0.03 and span a range of 0.0035 between correct and best
irrelevant — they are fusion artifacts, not confidence. `MIN_RETRIEVAL_SCORE`
(0.25, untouched) would abstain on every query under RRF. Any gate on a fused
retriever needs a calibrated score, which fusion does not provide.

## Experiment 3 — RRF parameter sensitivity (K × candidate depth)

Measured **2026-09-09**, `large` scale, same 44 questions.
`python scripts/rrf_sweep.py`

Controlled by construction: the corpus is ingested once and the vector and BM25
searchers are built once, so every configuration fuses *identical* component
rankings. Only K and depth vary. **The production default is unchanged
(K=60, D=50).**

### When can a one-retriever chunk outrank a two-retriever chunk?

A chunk found by one retriever at rank `r` scores `1/(K+r)`. A chunk found by
both at ranks `a,b` scores `1/(K+a) + 1/(K+b)`; its weakest possible case is
`a=b=D`, scoring `2/(K+D)`. So the single-retriever chunk can win only when

```
1/(K + r) > 2/(K + D)   ⟺   K + D > 2K + 2r   ⟺   r < (D − K)/2      (1)
```

Setting r = 1 gives the regime boundary **D > K + 2**: unless depth exceeds K by
more than 2, *no* single-retriever chunk can ever outrank a doubly-retrieved
one, at any rank. Largest rank that can still win, by (1):

| D \ K | 5 | 10 | 20 | 30 | 60 | 120 |
|---|---|---|---|---|---|---|
| 10 | r≤2 | NEVER | NEVER | NEVER | NEVER | NEVER |
| 20 | r≤7 | r≤4 | NEVER | NEVER | NEVER | NEVER |
| 50 | r≤22 | r≤19 | r≤14 | r≤9 | **NEVER** | NEVER |
| 100 | r≤47 | r≤44 | r≤39 | r≤34 | r≤19 | NEVER |
| 144 | r≤69 | r≤66 | r≤61 | r≤56 | r≤41 | r≤11 |

The production default (K=60, D=50) sits in the NEVER region — confirmed
empirically: at that setting X1 beats **0** of the doubly-retrieved chunks.

**(1) bounds possibility, not outcome.** Beating the *worst* doubly-retrieved
chunk is not the same as beating well-ranked ones. At K=5/D=50 condition (1)
permits X1 (vector rank 6) to win, yet it still lands at rank 16 — because 13
chunks are ranked well by *both* retrievers.

### The sweep

| D | K | R@1 | R@3 | R@5 | R@10 | MRR | exact | cred R@10 | X1 | P5 | E1 | E6 | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 5–120 | 63.2% | 92.1% | 92.1% | **97.4%** | 0.779 | 28.6% | **66.7%** | **9** | 3 | 2 | 2 | **17.1** |
| 20 | 5–120 | **65.8%** | **92.1%** | 92.1% | 94.7% | **0.784** | 42.9% | 33.3% | – | 3 | 2 | 1 | 18.5 |
| 50 | 5 | **65.8%** | **92.1%** | 92.1% | 94.7% | **0.784** | 42.9% | 33.3% | – | 3 | 2 | 1 | 21.2 |
| 50 | 10 | 65.8% | 89.5% | 92.1% | 94.7% | 0.782 | 42.9% | 33.3% | – | 4 | 2 | 1 | 21.1 |
| **50** | **60** | 65.8% | 89.5% | 92.1% | 92.1% | 0.779 | 42.9% | **0.0%** | – | 4 | 2 | 1 | 21.2 |
| 100 | 5 | 65.8% | 92.1% | 92.1% | 94.7% | 0.784 | 42.9% | 33.3% | – | 3 | 2 | 1 | 25.2 |
| 144 | 120 | 65.8% | 89.5% | 92.1% | 92.1% | 0.777 | 42.9% | 0.0% | – | 5 | 2 | 1 | 29.0 |

(bold row = current production default. Full 30-configuration grid in the
script output.)

### Findings

**1. K is almost inert; depth is the live parameter.** Across K ∈ {5…120} at
fixed depth, Recall@1 is *identical* and MRR moves by ≤0.005. Per-category
Recall@1 is byte-identical across every K at every depth. K's only visible
effect is on Recall@3/@10 at D≥50, where K=5 keeps 92.1%/94.7% versus
89.5%/92.1% at K≥20.

**2. Depth drives everything, through the size of the agreed set.** Depth
controls how many chunks *both* retrievers return, and those are what a
single-retriever chunk must outrank. For X1:

| D | chunks in both lists | X1's fused rank |
|---|---|---|
| 10 | 3 | **9** |
| 20 | 11 | 13–15 |
| 50 | 20 | 16–23 |
| 144 | 27 | 18–28 |

**3. Shallow depth partially recovers the credential case.** At D=10, X1 reaches
rank 9 and credential Recall@10 is 66.7% — the best in the grid. But no
configuration reaches vector-alone's credential Recall@10 of **100%**, and
**credential Recall@1 is 0.0% in all 30 configurations.** Parameter tuning does
not fix this failure.

**4. exact_term needs depth ≥ 20.** D=10 drops it to 28.6% (from 42.9%), because
BM25's evidence is truncated before fusion.

**5. paraphrase is completely insensitive** — 87.5% in every configuration.

### Trade-offs

| Axis | Best | Cost elsewhere |
|---|---|---|
| Overall quality | D=20 or D=50/K=5 (MRR 0.784) | none measured |
| Recall@10 | D=10 (97.4%) | exact_term −14.3 |
| exact_term | D≥20 (42.9%) | credential R@10 −33.3 |
| credential | D=10 (R@10 66.7%) | exact_term 28.6%; still R@1 0% |
| paraphrase | insensitive | — |
| Stability | K irrelevant; depth decisive | — |
| Latency | D=10 (17.1 ms) | scales ~linearly: D=144 → 29.0 ms |

**No configuration is best on every axis.** exact_term and credential pull in
opposite directions on depth: exact_term wants ≥20, credential wants 10.

### Not a recommendation to change the default

The evidence does say the current default is *dominated*: **D=20 (any K)** and
**D=50/K=5** each match it on Recall@1 and exact_term while beating it on
Recall@3 (+2.6), Recall@10 (+2.6), MRR (+0.005) and credential Recall@10
(+33.3); D=20 is also 2.7 ms faster. That is a consistent, if small, improvement
on a 38-question sample where one question is 2.6 points — well within noise for
a single corpus.

Changing the default should be a deliberate decision made on more than this one
synthetic corpus, and none of these configurations addresses the credential
failure, which remains 0% Recall@1 everywhere.

## Experiment 4 — cross-encoder reranking

Measured **2026-09-09**, `large` scale, same 44 questions.
`python scripts/rerank_sweep.py`

Pipeline: `vector + BM25 → RRF → candidate pool → cross-encoder → final`.
Model `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, CPU, `max_length=512`),
configurable via `RERANK_MODEL` / `RERANK_CANDIDATES` / `RERANK_MAX_LENGTH`.
Vector, BM25, RRF and `MIN_RETRIEVAL_SCORE` are untouched.

### Depth sweep

| Depth | R@1 | R@3 | R@5 | R@10 | MRR | exact | cred | X1 | P5 | E1 | E6 | latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 68.4% | 86.8% | 92.1% | 92.1% | 0.785 | 57.1% | 0.0% | – | 4 | 2 | 1 | **259 ms** |
| 20 | 68.4% | 86.8% | 92.1% | 92.1% | 0.785 | 57.1% | 0.0% | – | 4 | 2 | 1 | 519 ms |
| 30 | 68.4% | 86.8% | 92.1% | 92.1% | 0.785 | 57.1% | 0.0% | – | 4 | 2 | 1 | 773 ms |
| 50 | 68.4% | 86.8% | 92.1% | 92.1% | 0.785 | 57.1% | 0.0% | – | 4 | 2 | 1 | 1120 ms |
| *RRF* | *65.8%* | *89.5%* | *92.1%* | *92.1%* | *0.779* | *42.9%* | *0.0%* | *–* | *4* | *2* | *1* | *21 ms* |

**Depth changes nothing but cost.** Quality is identical at every depth; latency
scales linearly, 259 ms → 1120 ms. Deeper pools add candidates the reranker
never promotes.

### Acceptance tests

| # | Test | Result |
|---|---|---|
| 1 | X1 moves substantially up, ideally rank 1 | ❌ **FAILED** |
| 2 | credential Recall@1 improves over 0% | ❌ **FAILED** — 0.0% at every depth |
| 3 | exact_term R@1 does not regress from 42.9% | ✅ **PASSED** — 57.1% (+14.2) |
| 4 | MRR improves over 0.779 | ✅ **PASSED** — 0.785 (+0.006) |

Also: R@1 68.4% (+2.6), `direct` R@1 100% (from 75%), but **R@3 regressed
89.5% → 86.8%**.

### The reranker had the credential chunk and demoted it

| Rerank depth | credential chunk in RRF pool at | after reranking |
|---|---|---|
| 10 | absent | absent |
| 20 | absent | absent |
| 30 | **rank 23** | **rank 28** |
| 50 | **ranks 23, 36** | **ranks 26, 45** |

At depths 30 and 50 the chunk was in the pool and the cross-encoder ranked it
*lower* than RRF had. This is now conclusive across all four stages: dense
retrieval, lexical retrieval, rank fusion, and a cross-encoder that reads query
and chunk together all fail to connect "gmail password" to an unlabelled token.
**No ranking method can, because the chunk contains no evidence that the token
is a password.** The information needed is not in the corpus.

### The real win: scores that finally separate

| Retriever | correct − best irrelevant | best Youden | at floor |
|---|---|---|---|
| Vector | +0.0566 | 0.421 | 0.45 |
| BM25 | +2.4848 | — | — |
| RRF | +0.0035 | −0.044 (at default) | 0.25 |
| **Cross-encoder** | **+5.054** | **0.632** | **−3.0** |

On absent questions: correct answers average **−1.000**, absent questions'
top-1 averages **−8.483** with a maximum of **−3.314** — a margin of **+2.314**
between the best absent question and the average correct one. A floor at −3.0
would keep 24/38 answerable while rejecting **6/6** absent.

This is the first stage in the pipeline producing a score worth thresholding on,
and it is what every earlier threshold analysis said was missing. **No threshold
is implemented here** — `MIN_RETRIEVAL_SCORE` remains 0.25 and is a cosine
value that must never be compared against these logits.

### Cost

259 ms at depth 10 versus 21 ms for RRF — 12× slower, and that is the cheapest
configuration. For a live demo on a laptop this is the dominant cost in the
retrieval path.

## Experiment 5 — cross-encoder abstention calibration

Measured **2026-09-09**, `large` scale, same 44 questions, reranker top-20.
`python scripts/rerank_calibration.py`

An **abstention** gate, not a ranking one: accept if the top-1 cross-encoder
score ≥ threshold, otherwise refuse to answer. Ordering is never touched.
`MIN_RETRIEVAL_SCORE=0.25` is a cosine threshold, untouched and not used here.
**Nothing is applied to production.**

### Two ways to count an acceptance

Three answerable questions — **X1, X2, X3** — have *no relevant chunk retrieved
at all*. Accepting them cannot produce a grounded answer; it hands the LLM
irrelevant context and invites a confident wrong answer. So both accountings are
reported: **LENIENT** (accepted + answerable = success) and **STRICT** (also
requires that real evidence was retrieved).

### LENIENT

| thr | ansAcc | ansRej | absRej | absAcc | prec | recall | F1 | Youden J | FAR | FRR |
|---|---|---|---|---|---|---|---|---|---|---|
| −10.0 | 34 | 4 | 3 | 3 | 0.919 | 0.895 | 0.907 | 0.395 | 0.500 | 0.105 |
| −8.0 | 32 | 6 | 4 | 2 | 0.941 | 0.842 | 0.889 | 0.509 | 0.333 | 0.158 |
| −6.0 | 30 | 8 | 4 | 2 | 0.938 | 0.789 | 0.857 | 0.456 | 0.333 | 0.211 |
| −5.0 | 29 | 9 | 5 | 1 | 0.967 | 0.763 | 0.853 | 0.596 | 0.167 | 0.237 |
| −4.0 | 25 | 13 | 5 | 1 | 0.962 | 0.658 | 0.781 | 0.491 | 0.167 | 0.342 |
| −3.5 | 24 | 14 | 5 | 1 | 0.960 | 0.632 | 0.762 | 0.465 | 0.167 | 0.368 |
| **−3.3** | 24 | 14 | **6** | **0** | **1.000** | 0.632 | 0.774 | **0.632** | **0.000** | 0.368 |
| −3.0 | 24 | 14 | 6 | 0 | 1.000 | 0.632 | 0.774 | 0.632 | 0.000 | 0.368 |
| −2.5 | 24 | 14 | 6 | 0 | 1.000 | 0.632 | 0.774 | 0.632 | 0.000 | 0.368 |
| −2.0 | 23 | 15 | 6 | 0 | 1.000 | 0.605 | 0.754 | 0.605 | 0.000 | 0.395 |
| −1.5 | 23 | 15 | 6 | 0 | 1.000 | 0.605 | 0.754 | 0.605 | 0.000 | 0.395 |
| −1.0 | 22 | 16 | 6 | 0 | 1.000 | 0.579 | 0.733 | 0.579 | 0.000 | 0.421 |

### STRICT

| thr | ansAcc | ansRej | absRej | absAcc | prec | recall | F1 | Youden J | FAR | FRR |
|---|---|---|---|---|---|---|---|---|---|---|
| −10.0 | 31 | 4 | 3 | 6 | 0.838 | 0.886 | 0.861 | 0.219 | 0.667 | 0.114 |
| −8.0 | 30 | 5 | 5 | 4 | 0.882 | 0.857 | 0.870 | 0.413 | 0.444 | 0.143 |
| −6.0 | 30 | 5 | 7 | 2 | 0.938 | 0.857 | 0.896 | 0.635 | 0.222 | 0.143 |
| **−5.0** | 29 | 6 | 8 | 1 | 0.967 | 0.829 | 0.892 | **0.717** | 0.111 | 0.171 |
| −4.0 | 25 | 10 | 8 | 1 | 0.962 | 0.714 | 0.820 | 0.603 | 0.111 | 0.286 |
| −3.5 | 24 | 11 | 8 | 1 | 0.960 | 0.686 | 0.800 | 0.575 | 0.111 | 0.314 |
| −3.3 | 24 | 11 | 9 | 0 | 1.000 | 0.686 | 0.814 | 0.686 | 0.000 | 0.314 |
| −3.0 | 24 | 11 | 9 | 0 | 1.000 | 0.686 | 0.814 | 0.686 | 0.000 | 0.314 |
| −2.5 | 24 | 11 | 9 | 0 | 1.000 | 0.686 | 0.814 | 0.686 | 0.000 | 0.314 |
| −2.0 | 23 | 12 | 9 | 0 | 1.000 | 0.657 | 0.793 | 0.657 | 0.000 | 0.343 |
| −1.0 | 22 | 13 | 9 | 0 | 1.000 | 0.629 | 0.772 | 0.629 | 0.000 | 0.371 |

The two accountings **disagree about the optimum** (−3.3 vs −5.0), so which one
is used is a decision, not a detail.

### Acceptance by category

| thr | direct | paraphrase | contextual | multi_msg | exact_term | credential | absent |
|---|---|---|---|---|---|---|---|
| −10.0 | 8/8 | **4/8** | 6/6 | 6/6 | 7/7 | 3/3 | 3/6 |
| −6.0 | 8/8 | **4/8** | 6/6 | 5/6 | 7/7 | **0/3** | 2/6 |
| −5.0 | 8/8 | 4/8 | 6/6 | 4/6 | 7/7 | 0/3 | 1/6 |
| −3.3 | 7/8 | 3/8 | 5/6 | 3/6 | 6/7 | 0/3 | **0/6** |
| −1.0 | 7/8 | 3/8 | 4/6 | 3/6 | 5/7 | 0/3 | 0/6 |

**`paraphrase` never exceeds 4/8, even at −10.** Half of those questions score
below −10 no matter how permissive the gate.

### Watched questions

| Q | top-1 score | evidence rank | behaviour |
|---|---|---|---|
| **E6** `TCS` | +0.38 | 1 | accepted at every threshold ✅ |
| **E1** | −4.35 | 2 | abstained below −4.0 ❌ |
| **P5** | −10.81 | 4 | **abstained at every threshold** ❌ |
| **X1** | −7.20 | none | abstained from −6.0 ✅ *(correctly)* |
| X2 | −9.71 | none | abstained from −8.0 ✅ |
| X3 | −6.46 | none | abstained from −6.0 ✅ |

**The credential questions are correctly refused.** From −6.0 down, all three
abstain. Retrieval cannot reach the credential, but the system can at least
decline rather than fabricate — the grounded behaviour the project requires.

### The separability ceiling

Absent top-1 scores span **−10.94 … −3.31**. Answerable-with-evidence spans
**−11.28 … +8.77**. The distributions overlap, and **11 of 35 answerable
questions whose evidence WAS retrieved score below the worst absent question**:

| Q | category | top-1 | evidence at rank |
|---|---|---|---|
| P8 | paraphrase | **−11.28** | **1** |
| P4 | paraphrase | −11.22 | 3 |
| P5 | paraphrase | −10.81 | 4 |
| P2 | paraphrase | −10.08 | 1 |
| M3 | multi_message | −9.14 | 1 |
| M1 | multi_message | −5.47 | 2 |
| C3 | contextual | −4.92 | 1 |
| M6 | multi_message | −4.88 | 2 |
| D7 | direct | −4.85 | 1 |
| E1 | exact_term | −4.35 | 2 |
| P7 | paraphrase | −3.73 | 1 |

P8 has the **correct chunk at rank 1** scoring −11.28 — eight points below a
question with no answer at all. Any threshold accepting P8 must accept all six
absent questions.

**Ceiling: at zero false acceptance, at most 24 of 38 answerable questions can
be accepted.** The remaining 37% are structurally unreachable.

### Verdict — is the score calibrated enough for a reliable abstention decision?

**Partially. It is by far the best signal in the pipeline, and it is not
sufficient on its own.**

For it: at −3.3 the gate reaches **zero false acceptances**, precision 1.000, and
correctly refuses all three credential questions. Youden 0.632 (lenient) /
0.717 (strict) against vector's best 0.421 and RRF's −0.044.

Against it: that costs a **36.8% false-rejection rate** — 14 of 38 answerable
questions refused, including ones whose evidence was retrieved at rank 1. The
cause is that cross-encoder logits are calibrated *within* a query for ranking,
not *across* queries. Absolute magnitude tracks phrasing as much as evidence
quality, which is why `paraphrase` collapses.

A single global threshold is therefore the wrong shape for this problem. Options
worth testing before adopting one: per-question normalisation (top-1 margin over
the rest of the candidates rather than the absolute score), score calibration
against a held-out set, or letting the grounded prompt keep making the
abstention call with the cross-encoder score as one input among several.

## Experiment 6 — margin vs absolute score as an abstention signal

Measured **2026-09-09**, `large` scale, same 44 questions, reranker top-20.
`python scripts/margin_experiment.py`

**Question:** does within-query score *separation* beat the absolute
cross-encoder logit as an abstention signal? Experiment 5 found the absolute
score only partially calibrated because logits are calibrated within a query for
ranking, not across queries. A margin is within-query by construction, so it
should be immune to that.

Thresholds are derived from the data (every midpoint between observed values),
not assumed. Scoring goes through the same `calibration.classify` as experiment
5, so the comparison is exact. **STRICT is the production-relevant accounting.**
Retrieval is unchanged: R@1 68.4%, R@3 86.8%, R@5 92.1%, MRR 0.785,
`exact_term` R@1 57.1%, `credential` R@1 0.0%.

### Do the populations separate?

| Signal | answerable-with-evidence | absent | worst absent | positives below it | clean? | ceiling |
|---|---|---|---|---|---|---|
| abs top-1 | −11.28 … 8.78 (med −0.15) | −10.94 … −3.31 (med −10.28) | −3.31 | **11/35** | ❌ | 24/35 |
| margin 1−2 | 0.01 … 12.91 (med 3.45) | 0.01 … 1.46 (med 0.62) | 1.46 | **12/35** | ❌ | 23/35 |
| margin 1−3 | 0.05 … 17.98 (med 7.40) | 0.17 … 5.27 (med 0.75) | 5.27 | **10/35** | ❌ | 25/35 |

**No signal separates cleanly.** All three interleave. Margins do separate the
*medians* far more sharply (3.45 vs 0.62 for m12; 7.40 vs 0.75 for m13) than the
absolute score does, but the tails still overlap and the tails are what a
threshold has to live in.

### Head to head — best achievable

**STRICT (production-relevant)**

| Signal | threshold | Youden J | precision | recall | F1 | FAR | FRR |
|---|---|---|---|---|---|---|---|
| **abs top-1** | −5.077 | **0.717** | 0.967 | **0.829** | **0.892** | 0.111 | **0.171** |
| margin 1−2 | 2.327 | 0.600 | **1.000** | 0.600 | 0.750 | **0.000** | 0.400 |
| margin 1−3 | 5.326 | 0.714 | **1.000** | 0.714 | 0.833 | **0.000** | 0.286 |

**LENIENT (for comparison only)**

| Signal | threshold | Youden J | precision | recall | F1 | FAR | FRR |
|---|---|---|---|---|---|---|---|
| abs top-1 | −2.833 | 0.632 | 1.000 | 0.632 | 0.774 | 0.000 | 0.368 |
| margin 1−2 | 1.517 | 0.632 | 1.000 | 0.632 | 0.774 | 0.000 | 0.368 |
| margin 1−3 | 5.326 | **0.658** | 1.000 | 0.658 | 0.794 | 0.000 | 0.342 |

The absolute score wins its own optimum on J, recall, F1 and FRR — but it gets
there by accepting one absent question (FAR 0.111). At the **zero-false-accept**
operating point, which is the one that matters for a grounded system, the
comparison flips slightly:

| Signal at FAR = 0 | accepted | J | FRR |
|---|---|---|---|
| abs top-1 (−3.3) | 24/35 | 0.686 | 0.314 |
| **margin 1−3 (5.33)** | **25/35** | **0.714** | **0.286** |
| margin 1−2 (2.33) | 21/35 | 0.600 | 0.400 |

**One question's difference.** On 35 samples that is not a result to act on.

### Category acceptance at each signal's best STRICT threshold

| Category | abs top-1 | margin 1−2 | margin 1−3 |
|---|---|---|---|
| direct | 8/8 | 7/8 | 8/8 |
| **paraphrase** | **4/8** | 4/8 | **2/8** |
| contextual | 6/6 | 4/6 | 5/6 |
| multi_message | 4/6 | 2/6 | 3/6 |
| **exact_term** | **7/7** | **4/7** | **7/7** |
| credential | 0/3 | 0/3 | 0/3 |
| absent | 1/6 | **0/6** | **0/6** |

margin 1−2 collapses `exact_term` to 4/7 for a structural reason: overlapping
chunks from the same conversation score almost identically, so the top-2 gap
vanishes even when the top hit is right (E1 0.60, E2 0.53, E3 0.78). Using
top-3 sidesteps that. margin 1−3 in turn halves `paraphrase`.

### Watched questions (at each signal's best STRICT threshold)

| Q | evidence | rank | abs top-1 | margin 1−2 | margin 1−3 |
|---|---|---|---|---|---|
| X1 | **NO** | – | abstain (−7.20) | abstain (2.29) | abstain (2.66) |
| X2 | **NO** | – | abstain (−9.71) | abstain (0.38) | abstain (0.73) |
| X3 | **NO** | – | abstain (−6.46) | abstain (0.53) | abstain (1.62) |
| P5 | yes | 4 | abstain (−10.81) | abstain (0.23) | abstain (0.27) |
| P8 | yes | **1** | abstain (−11.28) | abstain (0.04) | abstain (0.07) |
| E1 | yes | 2 | **ACCEPT** (−4.35) | abstain (0.59) | **ACCEPT** (6.55) |
| E6 | yes | 1 | **ACCEPT** (0.38) | **ACCEPT** (10.73) | **ACCEPT** (10.94) |

Two things worth flagging:

**X1 nearly defeats margin 1−2.** Its margin is **2.29 — higher than every
absent question's (max 1.46)**. The signal rates the credential question as more
answerable than questions that genuinely have no answer, despite no evidence
being retrieved. It abstains only because the optimum lands at 2.327, 0.04 above
it. That is luck, not calibration.

**P8 defeats every signal.** Its correct chunk is at rank 1, yet top-1 = −11.28
and both margins are ~0.05: the top three candidates are indistinguishable.
Where the cross-encoder is uniformly wrong about a query, neither its level nor
its spread carries information.

### Answer to the question

**No. Within-query separation is not a better abstention signal than the
absolute score.**

- Neither separates cleanly; both leave 10–12 of 35 answerable-with-evidence
  questions below the worst absent question.
- margin 1−2 is **clearly worse** — lower J, lower recall, and it destroys
  `exact_term` through chunk overlap.
- margin 1−3 is **equivalent within noise** — better by one question at zero
  false acceptance (25 vs 24), worse on J/recall/F1 at each signal's own
  optimum, and it halves `paraphrase`.
- Margins do not fix the underlying failure. The same questions (P2, P4, P5, P8)
  fail under every signal, because the cross-encoder scores those queries badly
  in *both* level and spread.

The hypothesis that a within-query signal would sidestep cross-query calibration
was reasonable and is not supported. The limit is not the choice of statistic —
it is that the cross-encoder is simply wrong about a subset of paraphrase and
multi-message queries, and no function of its own output can detect that.

## Increment 7 — end-to-end grounded answering

Measured **2026-09-09**, `large` scale. `python scripts/run_e2e.py`

Full stack: `vector + BM25 → RRF → cross-encoder → top-5 chunks → Gemini →
grounded answer + citations`. Pure composition — no retrieval stage was
modified. Abstention is left entirely to the grounded prompt; **no
cross-encoder score or margin threshold was introduced**, per experiments 5–6.

### ⚠ The full 44-question run could not be completed

Gemini's free tier caps `GenerateRequestsPerDayPerProjectPerModel` at **20
requests per day, per model**. A 44-question run needs 44. The first attempt
exhausted `gemini-3.7-flash` and **all 44 questions failed with HTTP 429** — the
harness recorded them as failures rather than scoring them as abstentions, which
is exactly what its failure accounting exists for.

The results below are a **17-question subset** on `gemini-3.6-flash`, chosen to
cover all three groups and every watched question. They are real, but they are
not the full benchmark.

### Results (17-question subset, STRICT)

| Group | n | Result |
|---|---|---|
| **answerable, evidence retrieved** | 8 | **8/8 grounded successes (100%)** |
| **answerable, evidence NOT retrieved** | 3 | **3/3 correct abstentions, 0 fabrications** |
| **no answer exists** | 6 | **6/6 correct abstentions, 0 fabrications** |

| Metric | Value |
|---|---|
| Grounded answer accuracy | **100%** (8/8 answered) |
| Grounded-answer rate | 100% (8/8 with evidence) |
| Correct abstention rate | **100%** (9/9 should-refuse) |
| **Fabrication rate** | **0%** (0/9) |
| Unsupported-answer rate | **0%** |
| Citation correctness | **100%** (8/8 cite genuine supporting evidence) |
| End-to-end latency | mean 11.7 s, median 6.7 s, p95 39.2 s |

Retrieval over this subset: Recall@1 54.5%, Recall@3 63.6% — lower than the
68.4%/86.8% headline because the subset is deliberately weighted towards the
hard cases, not because retrieval changed.

### The watched questions

| Q | group | rank | outcome |
|---|---|---|---|
| **X1** | no evidence | – | **ABSTAINED** — *"I couldn't find anything about that in your chats."* |
| **X2** | no evidence | – | **ABSTAINED** |
| **X3** | no evidence | – | **ABSTAINED** |
| **P5** | with evidence | **4** | **GROUNDED**, 2 valid citations |
| **P8** | with evidence | 1 | **GROUNDED**, 1 valid citation |
| **E1** | with evidence | 2 | **GROUNDED**, 3 valid citations |
| **E6** | with evidence | 1 | **GROUNDED**, 1 valid citation |

Two results matter most:

**The credential questions are refused, not guessed.** X1/X2/X3 all abstain and
none states the synthetic credential. Earlier, at a loosened retrieval floor,
the model volunteered *"Likely: the password is …"* by inferring a nearby
handle. Rule 6 of the system prompt now forbids exactly that, and the behaviour
holds. The Gmail case remains a **retrieval** limitation — but it no longer
produces a fabricated answer.

**P5 answers correctly from evidence at rank 4.** The cross-encoder scored that
question −10.81, below every absent question; a score threshold would have
refused it. Leaving abstention to the prompt recovers it. That is direct
evidence for the decision not to add a threshold.

### Caveats

- 17 of 44 questions. `direct`, `contextual` and `multi_message` have one
  question each here; those rates are illustrative, not measured.
- Run on `gemini-3.6-flash` because `gemini-3.7-flash`'s daily quota was spent.
- Latency is dominated by the hosted LLM (p95 39 s) and varies with API load.
- LLM output is not bit-deterministic even at temperature 0; retrieval is.

Completing the full 44 needs a paid tier, or three runs across three days.

## Experiment 7 — retrieval strictness (Strict / Balanced / Permissive)

Measured **2026-09-10**, `large` scale, same 44 questions.
`python scripts/strictness_benchmark.py [--e2e]`

A user-facing control for "look harder", for questions about credentials, rare
names and codes where the right chunk ranks poorly. It moves **candidate depth**
— counts of chunks — and never a score. The four stages produce four
incomparable scales (cosine, BM25 sums, RRF fusion values, cross-encoder
logits), so a single numeric confidence knob would be meaningless; experiments 5
and 6 showed no cross-encoder threshold is calibrated enough to gate on either.

| mode | RRF pool | reranked | evidence → LLM |
|---|---|---|---|
| Strict | 50 | 20 | 3 |
| **Balanced (default)** | **50** | **20** | **5** |
| Permissive | **10** | 10 | 15 |

Balanced is derived from the live config, so it *is* the production default by
construction rather than a copy that can drift.

### Retrieval (all 44 questions, no LLM)

| mode | R@1 | R@3 | R@5 | R@10 | MRR | exact | X1 | X2 | X3 | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| Strict | 68.4% | 86.8% | 92.1% | 92.1% | 0.785 | 57.1% | – | – | – | 597 ms |
| Balanced | 68.4% | 86.8% | 92.1% | 92.1% | 0.785 | 57.1% | – | – | – | 570 ms |
| **Permissive** | 68.4% | 86.8% | 92.1% | **94.7%** | **0.792** | 57.1% | **14** | **8** | **13** | **449 ms** |

### Does the credential evidence reach the model?

| mode | evidence chunks | X1 | X2 | X3 |
|---|---|---|---|---|
| Strict | 3 | not reached | not reached | not reached |
| Balanced | 5 | not reached | not reached | not reached |
| **Permissive** | **15** | **rank 14** | **rank 8** | **rank 13** |

**Permissive is the first configuration that gets the credential evidence in
front of the model.** X1/X2/X3 were unreachable at every RRF parameterisation
tested in experiment 3 and at every rerank depth in experiment 4.

### Why Permissive fuses a *shallower* pool

The obvious design — widen every pool — was tried first and **made the credential
case worse**:

| variant | rrf | rerank | evidence | X1 | X2 | X3 | MRR | ms |
|---|---|---|---|---|---|---|---|---|
| balanced (ref) | 50 | 20 | 5 | – | – | – | 0.785 | 577 |
| wider evidence | 50 | 20 | 15 | – | 12 | – | 0.787 | 585 |
| deep + wide | 100 | 50 | 15 | – | – | – | 0.785 | 1170 |
| deep + very wide | 100 | 50 | 30 | 26 | 17 | – | 0.788 | 1170 |
| max depth | 144 | 144 | 30 | 28 | – | – | 0.786 | 2802 |
| **very shallow rrf** | **10** | **10** | **15** | **14** | **8** | **13** | **0.792** | **453** |

The cause is experiment 2's agreement-dominance result. RRF scores agreement, so
a deeper pool supplies *more* chunks that both retrievers return, and every one
of them outranks a chunk only one retriever can see. The credential chunk is
precisely that — visible to the dense retriever alone. Deepening the pool buries
it; shrinking the pool removes its competition.

So permissiveness is **"how much evidence reaches the model"**, and the pool
depth is set to serve that rather than raised for its own sake. Widening the
evidence window alone is not enough (row 2 reaches only X2).

### End to end (live LLM, 4 questions × 3 modes)

| mode | grounded acc | correct abstention | fabrication | **credential stated** | ms |
|---|---|---|---|---|---|
| Strict | 0/1 | 2/3 ⚠ | **0/3** | **never** | 12197 |
| Balanced | **1/1** | **3/3** | **0/3** | **never** | 2883 |
| Permissive | 0/0 | 1/2 | **0/2** | **never** | 11814 |

**The safety result is unambiguous and is the point of the experiment: zero
fabrications in every mode, and the credential is never stated in any mode —
including Permissive, where the token is demonstrably in the context.** X1 and
X3 abstain under all three settings. Permissive changes what the model may
*read*, never what it may *claim*.

Everything else in that table is too small and too noisy to conclude from:

- **Strict's 2/3 abstention is an artifact**: X3's LLM call failed after retries
  and was recorded as a failure, not an abstention. The benchmark now prints
  `!! LLM FAILED` so this cannot be misread again.
- Permissive's `should_refuse` is 2 rather than 3 because the credential
  evidence *does* reach the context, moving that question out of the
  "no evidence retrieved" group. Its abstention there is correct behaviour, and
  the strict accounting scores it as a miss.
- Grounded accuracy is n=1 per mode. Under Permissive, D1 answered without
  stating the gold fact — plausibly context dilution from 15 chunks, but a
  single observation is an anecdote.

### The default is unchanged

Balanced remains the default. Permissive wins on Recall@10, MRR, latency and
credential reach, but the end-to-end evidence for it is one question per mode
with a failed call in it, and the one hint of a downside — dilution at 15 chunks
— sits exactly where the evidence is thinnest. Changing the shipped default on
that would be unjustified. Permissive is offered to the user, per question,
which is where the trade-off belongs.

**Future experiment (not implemented):** run the full 44-question end-to-end
benchmark across all three modes on a paid tier — 132 calls — to establish
whether Permissive's context dilution is real, and whether it is worth making
the default for credential-shaped queries specifically.

## What this baseline is for

Any Phase 2 change (BM25, hybrid, reranking, semantic chunking) must be measured
at **`large` scale** against these numbers. The bar to beat:

- **Recall@1 57.9%**, MRR **0.677**, latency **15.9 ms**
- `exact_term` Recall@1 **0.0%** — the main target
- `credential` X1 at **rank 6** — must reach rank 1
- separation **+0.0566** — the number reranking should move most
- P5, E1, E6 currently unretrievable within k=10

Quote `sample` numbers only when comparing to the pre-existing baseline. A Phase
2 result reported at `sample` scale is not evidence.
