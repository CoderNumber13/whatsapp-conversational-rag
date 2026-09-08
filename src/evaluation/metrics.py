"""Retrieval metrics for the benchmark.

A retrieved chunk counts as RELEVANT for a question when it contains at least one
of that question's expected messages. Recall@K is therefore "did any expected
message appear inside the top-K chunks" — the right question for a RAG pipeline,
where a chunk is the unit handed to the LLM.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class QuestionResult:
    qid: str
    category: str
    question: str
    is_absent: bool
    n_expected: int
    # 1-based rank of the first relevant chunk; None if not in the top-K probed
    first_relevant_rank: Optional[int]
    # score of the best-ranked relevant chunk
    correct_score: Optional[float]
    # score of the best-ranked chunk that is NOT relevant
    top_irrelevant_score: Optional[float]
    top_score: float
    latency_ms: float
    n_retrieved: int

    def hit_at(self, k: int) -> bool:
        r = self.first_relevant_rank
        return r is not None and r <= k

    @property
    def separation(self) -> Optional[float]:
        """correct - best irrelevant. Negative means an irrelevant chunk
        outranked the answer by score."""
        if self.correct_score is None or self.top_irrelevant_score is None:
            return None
        return self.correct_score - self.top_irrelevant_score


def _mean(xs: Sequence[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _pct(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


@dataclass
class Summary:
    n: int
    n_answerable: int
    n_absent: int
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    latency_ms_mean: Optional[float] = None
    latency_ms_median: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    correct_score_mean: Optional[float] = None
    top_irrelevant_score_mean: Optional[float] = None
    separation_mean: Optional[float] = None
    absent_top_score_mean: Optional[float] = None
    absent_top_score_max: Optional[float] = None


def summarize(results: Sequence[QuestionResult], ks: Sequence[int] = (1, 3, 5, 10)) -> Summary:
    answerable = [r for r in results if not r.is_absent]
    absent = [r for r in results if r.is_absent]

    recall = {
        k: (sum(r.hit_at(k) for r in answerable) / len(answerable) if answerable else 0.0)
        for k in ks
    }
    rr = [1.0 / r.first_relevant_rank for r in answerable if r.first_relevant_rank]
    lat = [r.latency_ms for r in results]

    return Summary(
        n=len(results),
        n_answerable=len(answerable),
        n_absent=len(absent),
        recall=recall,
        mrr=(sum(rr) / len(answerable)) if answerable else 0.0,
        latency_ms_mean=_mean(lat),
        latency_ms_median=(statistics.median(lat) if lat else None),
        latency_ms_p95=_pct(lat, 0.95),
        correct_score_mean=_mean([r.correct_score for r in answerable
                                  if r.correct_score is not None]),
        top_irrelevant_score_mean=_mean([r.top_irrelevant_score for r in answerable
                                         if r.top_irrelevant_score is not None]),
        separation_mean=_mean([s for s in (r.separation for r in answerable)
                               if s is not None]),
        absent_top_score_mean=_mean([r.top_score for r in absent]),
        absent_top_score_max=(max(r.top_score for r in absent) if absent else None),
    )


@dataclass
class ThresholdPoint:
    threshold: float
    answerable_kept: int      # answerable questions whose top-1 clears the floor
    answerable_total: int
    absent_rejected: int      # absent questions correctly refused before the LLM
    absent_total: int

    @property
    def youden(self) -> float:
        """kept-rate + rejected-rate - 1. Positive means the score separates the
        two populations at all; <= 0 means it is no better than a coin flip."""
        a = self.answerable_kept / self.answerable_total if self.answerable_total else 0.0
        b = self.absent_rejected / self.absent_total if self.absent_total else 0.0
        return a + b - 1.0


def threshold_analysis(
    results: Sequence[QuestionResult], grid: Optional[Sequence[float]] = None
) -> list[ThresholdPoint]:
    """How well an absolute score floor (MIN_RETRIEVAL_SCORE) can ever separate
    answerable questions from absent ones on this corpus.

    The abstain gate compares the TOP-1 score against the floor, so that is what
    is measured here — not the score of the correct chunk.
    """
    answerable = [r for r in results if not r.is_absent]
    absent = [r for r in results if r.is_absent]
    if grid is None:
        grid = [round(x / 100, 2) for x in range(0, 101, 5)]
    return [
        ThresholdPoint(
            threshold=t,
            answerable_kept=sum(r.top_score >= t for r in answerable),
            answerable_total=len(answerable),
            absent_rejected=sum(r.top_score < t for r in absent),
            absent_total=len(absent),
        )
        for t in grid
    ]


def by_category(
    results: Sequence[QuestionResult], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, Summary]:
    cats: dict[str, list[QuestionResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    return {c: summarize(rs, ks) for c, rs in cats.items()}
