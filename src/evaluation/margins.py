"""Is within-query score *separation* a better abstention signal than the
absolute cross-encoder score?

Calibration (experiment 5) found the absolute logit only partially usable: it
reaches zero false acceptances at −3.3 but refuses 37% of answerable questions,
because cross-encoder logits are calibrated *within* a query for ranking, not
*across* queries. Absolute magnitude tracks phrasing as much as evidence quality.

A margin — how far the top candidate beats the runner-up — is by construction
within-query, so it should be immune to that. Whether it actually separates
"found the answer" from "there is no answer" is an empirical question, and the
plausible failure mode is obvious: a question with no answer can still have one
candidate that looks much better than the rest, producing a large margin on
nothing.

Nothing here is applied. Scoring uses ``calibration.classify`` so results are
directly comparable with the absolute-score baseline.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional, Sequence

from src.evaluation.calibration import ConfusionCounts, classify

SIGNALS = ("abs_top1", "margin_12", "margin_13")


@dataclass
class MarginObservation:
    qid: str
    category: str
    is_absent: bool
    top1: Optional[float]
    top2: Optional[float]
    top3: Optional[float]
    has_evidence: bool
    evidence_rank: Optional[int]

    @property
    def margin_12(self) -> Optional[float]:
        if self.top1 is None or self.top2 is None:
            return None
        return self.top1 - self.top2

    @property
    def margin_13(self) -> Optional[float]:
        if self.top1 is None or self.top3 is None:
            return None
        return self.top1 - self.top3

    @property
    def abs_top1(self) -> Optional[float]:
        return self.top1

    def signal(self, name: str) -> Optional[float]:
        if name not in SIGNALS:
            raise ValueError(f"unknown signal {name!r} (want one of {SIGNALS})")
        return getattr(self, name)


def observe(
    retriever, questions: Sequence, expected: dict[str, set[str]], k: int = 10
) -> list[MarginObservation]:
    """Record the top-3 reranked scores and evidence position for each question.

    Ordering is untouched — this only reads what the retriever already returned.
    """
    out: list[MarginObservation] = []
    for q in questions:
        want = expected[q.qid]
        hits = retriever.retrieve(q.question, k=max(k, 3))
        scores = [h.score for h in hits]
        rank = next((i for i, h in enumerate(hits, 1)
                     if want & set(h.chunk.message_ids)), None)
        out.append(MarginObservation(
            qid=q.qid, category=q.category, is_absent=q.is_absent,
            top1=scores[0] if len(scores) > 0 else None,
            top2=scores[1] if len(scores) > 1 else None,
            top3=scores[2] if len(scores) > 2 else None,
            has_evidence=rank is not None, evidence_rank=rank,
        ))
    return out


@dataclass
class SweepPoint:
    signal: str
    threshold: float
    lenient: ConfusionCounts
    strict: ConfusionCounts
    accepted_qids: frozenset


def sweep(
    observations: Sequence[MarginObservation],
    signal: str,
    thresholds: Sequence[float],
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for t in thresholds:
        lenient, strict = ConfusionCounts(), ConfusionCounts()
        accepted: set[str] = set()
        for o in observations:
            value = o.signal(signal)
            # A question with no candidates at all cannot be accepted.
            is_accepted = value is not None and value >= t
            if is_accepted:
                accepted.add(o.qid)
            len_cell, strict_cell = classify(
                is_absent=o.is_absent, has_evidence=o.has_evidence,
                accepted=is_accepted)
            setattr(lenient, len_cell, getattr(lenient, len_cell) + 1)
            setattr(strict, strict_cell, getattr(strict, strict_cell) + 1)
        points.append(SweepPoint(signal, t, lenient, strict, frozenset(accepted)))
    return points


def candidate_thresholds(
    observations: Sequence[MarginObservation], signal: str, *, limit: int = 60
) -> list[float]:
    """Decision boundaries implied by the data, not guessed.

    Every distinct ranking of the observations by this signal changes only at a
    midpoint between two observed values, so those midpoints (plus the extremes)
    are the complete set of thresholds worth testing.
    """
    vals = sorted({v for v in (o.signal(signal) for o in observations)
                   if v is not None})
    if not vals:
        return []
    mids = [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]
    grid = [vals[0] - 0.5, *mids, vals[-1] + 0.5]
    if len(grid) <= limit:
        return grid
    step = len(grid) / limit
    return [grid[int(i * step)] for i in range(limit)]


@dataclass
class Distribution:
    n: int
    lo: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    hi: Optional[float] = None
    mean: Optional[float] = None


def distribution(values: Sequence[float]) -> Distribution:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return Distribution(n=0)

    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = p * (len(vals) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)

    return Distribution(
        n=len(vals), lo=vals[0], q1=q(0.25), median=q(0.5), q3=q(0.75),
        hi=vals[-1], mean=statistics.fmean(vals),
    )


def overlap(
    observations: Sequence[MarginObservation], signal: str
) -> dict:
    """How badly the two populations interleave under this signal.

    ``clean`` is true only when every answerable-with-evidence question scores
    above every absent one — the condition for a threshold to separate them
    perfectly.
    """
    pos = [o.signal(signal) for o in observations
           if not o.is_absent and o.has_evidence]
    neg = [o.signal(signal) for o in observations if o.is_absent]
    pos = [v for v in pos if v is not None]
    neg = [v for v in neg if v is not None]
    if not pos or not neg:
        return {"clean": None, "positives": len(pos), "negatives": len(neg)}

    worst_neg, best_neg = max(neg), min(neg)
    below = [v for v in pos if v < worst_neg]
    # ceiling: accepting every positive above the worst negative, with no
    # negative accepted, is the best any single threshold can do
    return {
        "clean": len(below) == 0,
        "positives": len(pos),
        "negatives": len(neg),
        "worst_negative": worst_neg,
        "best_negative": best_neg,
        "positives_below_worst_negative": len(below),
        "max_positives_at_zero_false_accept": len(pos) - len(below),
        "positive": distribution(pos),
        "negative": distribution(neg),
    }
