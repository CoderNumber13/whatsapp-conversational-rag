"""RRF parameter-sensitivity experiment.

Measures how Reciprocal Rank Fusion responds to its two knobs, ``RRF_K`` and
``RRF_CANDIDATES`` (candidate depth D), holding everything else fixed.

Controlled by construction: the corpus is ingested once, the vector and BM25
searchers are built once, and only the fusion parameters vary between
configurations. Each configuration therefore fuses *identical* component
rankings, so every difference observed is attributable to K and D alone.

This is an experiment. It does not change the production default.


The agreement-dominance condition
---------------------------------
A chunk found by ONE retriever at rank ``r`` scores ``1/(K+r)``.
A chunk found by BOTH at ranks ``a`` and ``b`` scores ``1/(K+a) + 1/(K+b)``.

The hardest case for the single-retriever chunk is a doubly-retrieved chunk
sitting at the very bottom of both lists, ``a = b = D``, scoring ``2/(K+D)``.
So a single-retriever chunk at rank ``r`` can outrank *some* doubly-retrieved
chunk only when::

    1/(K + r) > 2/(K + D)
    K + D     > 2K + 2r
    r         < (D - K) / 2                                          (1)

Two readings of (1):

* **Per-chunk reach.** A chunk visible to only one retriever must rank better
  than ``(D-K)/2`` to have any chance. At K=60, D=50 that bound is negative:
  *no* rank is good enough, so agreement dominates absolutely.
* **Regime boundary.** Setting r = 1 gives ``D > K + 2``: unless the candidate
  depth exceeds K by more than 2, no single-retriever chunk can ever outrank a
  doubly-retrieved one, whatever its rank.

Note (1) bounds the *possibility*, not the outcome: beating the worst
doubly-retrieved chunk is not the same as beating the ones ranked well by both.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from src.config import Config
from src.evaluation.corpus import generate_large_corpus
from src.evaluation.dataset import QUESTIONS, EvalQuestion, build_corpus, resolve_expectations
from src.evaluation.metrics import QuestionResult, Summary, by_category, summarize
from src.evaluation.runner import SAMPLE_DIR, evaluate_questions

DEFAULT_KS = (5, 10, 20, 30, 60, 120)
DEFAULT_DEPTHS = (10, 20, 50, 100, 144)


def single_can_outrank_double(k_rrf: int, depth: int) -> bool:
    """Is there ANY rank at which a one-retriever chunk beats a two-retriever
    chunk? True iff D > K + 2 — condition (1) with r = 1."""
    return depth > k_rrf + 2


def max_rank_that_can_win(k_rrf: int, depth: int) -> int:
    """Best rank a single-retriever chunk must achieve to have a chance:
    the largest integer r satisfying r < (D-K)/2. Returns 0 when none does."""
    return max(0, math.ceil((depth - k_rrf) / 2.0) - 1)


@dataclass
class SweepPoint:
    k_rrf: int
    depth: int
    summary: Summary
    per_category: dict[str, Summary]
    ranks: dict[str, Optional[int]]          # qid -> rank of first relevant hit
    dominance: bool                          # agreement strictly dominates rank?
    winnable_rank: int                       # per condition (1)


def run_rrf_sweep(
    ks: Sequence[int] = DEFAULT_KS,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    *,
    questions: Sequence[EvalQuestion] = tuple(QUESTIONS),
    k: int = 10,
    scale: str = "large",
    embedding_model: Optional[str] = None,
    watch: Sequence[str] = ("X1", "X2", "X3", "P5", "E1", "E6"),
) -> tuple[list[SweepPoint], dict]:
    tmp = Path(tempfile.mkdtemp(prefix="convmem-sweep-"))
    env = {
        "DB_PATH": str(tmp / "sweep.db"),
        "INDEX_DIR": str(tmp / "index"),
        "UPLOADS_DIR": str(tmp / "uploads"),
        "LLM_PROVIDER": "mock",
        "MIN_RETRIEVAL_SCORE": "-1",
    }
    if embedding_model:
        env["EMBEDDING_MODEL"] = embedding_model
    old = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        cfg = Config.reload()
        from src.pipeline.rag_pipeline import RagPipeline
        from src.retrieval.hybrid_search import RRFHybridSearch
        from src.retrieval.keyword_search import BM25Search
        from src.retrieval.retriever import Retriever
        from src.retrieval.vector_search import VectorSearch

        files = (generate_large_corpus(tmp / "corpus", SAMPLE_DIR) if scale == "large"
                 else build_corpus(tmp / "corpus", SAMPLE_DIR))
        pipe = RagPipeline(cfg)
        pipe.ingest(files, me_names=["Me"])

        messages = [m for cid in pipe.db.conversation_ids()
                    for m in pipe.db.get_conversation_messages(cid)]
        expected = resolve_expectations(questions, messages)

        # built ONCE and shared by every configuration
        vector = VectorSearch.open(pipe.db, cfg)
        bm25 = BM25Search.open(pipe.db, cfg)
        Retriever(pipe.db, vector, cfg).retrieve("warmup", k=1)  # load the model

        points: list[SweepPoint] = []
        for depth in depths:
            for k_rrf in ks:
                fused = RRFHybridSearch(
                    [("vector", vector), ("bm25", bm25)], cfg,
                    k_rrf=k_rrf, candidates=depth,
                )
                results = evaluate_questions(
                    Retriever(pipe.db, fused, cfg), questions, expected, k)
                by_qid = {r.qid: r.first_relevant_rank for r in results}
                points.append(SweepPoint(
                    k_rrf=k_rrf, depth=depth,
                    summary=summarize(results), per_category=by_category(results),
                    ranks={q: by_qid.get(q) for q in watch},
                    dominance=not single_can_outrank_double(k_rrf, depth),
                    winnable_rank=max_rank_that_can_win(k_rrf, depth),
                ))

        meta = {
            "scale": scale, "messages": len(messages),
            "conversations": len(pipe.db.conversation_ids()),
            "chunks": pipe.db.count_chunks(), "probe_k": k,
            "embedding_model": cfg.embedding_model,
            "questions": len(questions),
        }
        pipe.close()
        return points, meta
    finally:
        for key, val in old.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        Config.reload()
        shutil.rmtree(tmp, ignore_errors=True)
