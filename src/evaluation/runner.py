"""Run the retrieval benchmark against the current retriever.

Retrieval only — no LLM is called, so the numbers describe the retriever alone.
Everything is built in a throwaway directory: the benchmark never reads or
writes the user's private DB or index.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from src.config import Config
from src.evaluation.corpus import generate_large_corpus
from src.evaluation.dataset import QUESTIONS, EvalQuestion, build_corpus, resolve_expectations
from src.evaluation.metrics import QuestionResult, Summary, by_category, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "synthetic_chats"

PROBE_K = 10  # deepest K we report Recall for


@dataclass
class EvalRun:
    results: list[QuestionResult]
    summary: Summary
    per_category: dict[str, Summary]
    corpus: dict


def evaluate_questions(
    retriever, questions: Sequence[EvalQuestion], expected: dict[str, set[str]], k: int
) -> list[QuestionResult]:
    """Score one retriever over the benchmark.

    Shared by ``run_eval`` and the RRF sweep so both measure identically — a
    parameter sweep that scored differently from the baseline would not be
    comparable with it.
    """
    results: list[QuestionResult] = []
    for q in questions:
        want = expected[q.qid]
        t0 = time.perf_counter()
        hits = retriever.retrieve(q.question, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        first_rank: Optional[int] = None
        correct_score: Optional[float] = None
        top_irrelevant: Optional[float] = None
        for i, h in enumerate(hits, start=1):
            relevant = bool(want & set(h.chunk.message_ids))
            if relevant and first_rank is None:
                first_rank, correct_score = i, h.score
            if not relevant and top_irrelevant is None:
                top_irrelevant = h.score

        results.append(QuestionResult(
            qid=q.qid, category=q.category, question=q.question,
            is_absent=q.is_absent, n_expected=len(want),
            first_relevant_rank=first_rank, correct_score=correct_score,
            top_irrelevant_score=top_irrelevant,
            top_score=(hits[0].score if hits else 0.0),
            latency_ms=latency_ms, n_retrieved=len(hits),
        ))
    return results


def run_eval(
    questions: Sequence[EvalQuestion] = tuple(QUESTIONS),
    *,
    k: int = PROBE_K,
    embedding_model: Optional[str] = None,
    workdir: Optional[Path] = None,
    scale: str = "sample",
    retriever: str = "vector",
    rerank_depth: Optional[int] = None,
) -> EvalRun:
    """`scale="sample"` is the 7-chunk tracked corpus; `scale="large"` adds
    production-scale filler and distractors around the same gold answers.

    `retriever` selects the searcher behind the identical retrieval path:
    "vector" (FAISS, the baseline) or "bm25" (lexical). Both satisfy
    ChunkSearcher, so nothing else in the pipeline changes.
    """
    if scale not in ("sample", "large"):
        raise ValueError(f"unknown scale {scale!r} (want 'sample' or 'large')")
    if retriever not in ("vector", "bm25", "rrf", "rerank"):
        raise ValueError(f"unknown retriever {retriever!r} "
                         "(want 'vector', 'bm25', 'rrf' or 'rerank')")
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="convmem-eval-"))
    owned = workdir is None
    try:
        import os

        env = {
            "DB_PATH": str(tmp / "eval.db"),
            "INDEX_DIR": str(tmp / "index"),
            "UPLOADS_DIR": str(tmp / "uploads"),
            "LLM_PROVIDER": "mock",          # retrieval only; never calls an API
            "MIN_RETRIEVAL_SCORE": "-1",     # measure ranking, not the abstain gate
        }
        if embedding_model:
            env["EMBEDDING_MODEL"] = embedding_model
        old = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            cfg = Config.reload()
            from src.pipeline.rag_pipeline import RagPipeline

            files = (
                generate_large_corpus(tmp / "corpus", SAMPLE_DIR)
                if scale == "large"
                else build_corpus(tmp / "corpus", SAMPLE_DIR)
            )
            pipe = RagPipeline(cfg)
            pipe.ingest(files, me_names=["Me"])

            messages = [
                m for cid in pipe.db.conversation_ids()
                for m in pipe.db.get_conversation_messages(cid)
            ]
            expected = resolve_expectations(questions, messages)

            if retriever != "vector":
                from src.retrieval.retriever import Retriever

                # same Retriever, same filters, same context expansion — only
                # the searcher differs, so the comparison isolates ranking
                if retriever == "bm25":
                    from src.retrieval.keyword_search import BM25Search

                    searcher = BM25Search.open(pipe.db, cfg)
                else:
                    from src.retrieval.hybrid_search import RRFHybridSearch

                    searcher = RRFHybridSearch.open(pipe.db, cfg)
                    if retriever == "rerank":
                        from src.retrieval.reranker import (
                            CrossEncoderReranker, RerankedSearch,
                        )

                        ce = CrossEncoderReranker(config=cfg)
                        ce.warmup()
                        searcher = RerankedSearch(
                            searcher, ce, cfg, candidates=rerank_depth)
                pipe._retriever = Retriever(pipe.db, searcher, cfg)

            # warm up so the first question doesn't absorb model/index load
            pipe.retriever.retrieve("warmup", k=1)

            results = evaluate_questions(pipe.retriever, questions, expected, k)

            corpus = {
                "scale": scale,
                "retriever": retriever,
                "rerank_candidates": (
                    (rerank_depth or cfg.rerank_candidates)
                    if retriever == "rerank" else None),
                "rerank_model": cfg.rerank_model if retriever == "rerank" else None,
                "messages": len(messages),
                "conversations": len(pipe.db.conversation_ids()),
                "chunks": pipe.db.count_chunks(),
                "embedding_model": cfg.embedding_model,
                "chunk_size_messages": cfg.chunk_size_messages,
                "chunk_overlap_messages": cfg.chunk_overlap_messages,
                "probe_k": k,
            }
            pipe.close()
            return EvalRun(results, summarize(results), by_category(results), corpus)
        finally:
            for key, val in old.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
            Config.reload()
    finally:
        if owned:
            shutil.rmtree(tmp, ignore_errors=True)
