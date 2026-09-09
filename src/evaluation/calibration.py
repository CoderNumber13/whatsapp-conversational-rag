"""Abstention-threshold calibration for cross-encoder scores.

Answers one question: is the cross-encoder's relevance score calibrated well
enough to decide "I don't have enough evidence to answer"?

This is an ABSTENTION gate, not a ranking one. Candidate ordering is never
touched; the decision is made on the top-1 score *after* reranking:

    accept  if  top1_score >= threshold        -> hand context to the LLM
    abstain otherwise

``MIN_RETRIEVAL_SCORE`` is a cosine threshold for the dense retriever and is
untouched and unrelated. Cross-encoder scores are unbounded logits.


Two ways to count a correct acceptance
--------------------------------------
The obvious accounting — an answerable question that is accepted is a success —
hides the failure mode that matters most for a grounded RAG system. A question
can be answerable, and accepted, while the evidence that answers it was never
retrieved. The gate then passes a pile of irrelevant chunks to the LLM and
invites a confident wrong answer, which is strictly worse than abstaining.

So both are computed:

* ``LENIENT``  — accepted and the question is answerable.
* ``STRICT``   — accepted, answerable, **and** at least one genuinely relevant
  chunk was retrieved, so the LLM has something real to ground on.

X1 and X3 are precisely the difference: answerable questions whose evidence no
retriever ever surfaces. Under LENIENT accounting, accepting them looks like a
win. Under STRICT it is a false acceptance, which is what it actually is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.evaluation.metrics import QuestionResult

DEFAULT_THRESHOLDS = (
    -10.0, -8.0, -6.0, -5.0, -4.0, -3.5, -3.3, -3.0, -2.5, -2.0, -1.5, -1.0,
)


def _safe(num: float, den: float) -> Optional[float]:
    return (num / den) if den else None


@dataclass
class ConfusionCounts:
    """Positive class = "accepted"."""

    tp: int = 0   # accepted, and should have been
    fn: int = 0   # abstained, but could have answered
    tn: int = 0   # abstained on a question with no answer — correct
    fp: int = 0   # accepted a question it should have refused

    @property
    def precision(self) -> Optional[float]:
        return _safe(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> Optional[float]:
        """Sensitivity: of the questions that should be answered, how many are."""
        return _safe(self.tp, self.tp + self.fn)

    @property
    def specificity(self) -> Optional[float]:
        return _safe(self.tn, self.tn + self.fp)

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p and r and (p + r)) else (0.0 if p is not None else None)

    @property
    def youden_j(self) -> Optional[float]:
        s, sp = self.recall, self.specificity
        return (s + sp - 1.0) if (s is not None and sp is not None) else None

    @property
    def false_acceptance_rate(self) -> Optional[float]:
        """Share of questions that should be refused but are accepted (1 - specificity)."""
        return _safe(self.fp, self.fp + self.tn)

    @property
    def false_rejection_rate(self) -> Optional[float]:
        """Share of answerable questions wrongly abstained on (1 - recall)."""
        return _safe(self.fn, self.fn + self.tp)


@dataclass
class CalibrationPoint:
    threshold: float
    lenient: ConfusionCounts
    strict: ConfusionCounts
    # category -> (accepted, total)
    per_category: dict[str, tuple[int, int]] = field(default_factory=dict)
    # qid -> (accepted, top1 score, rank of first relevant hit)
    watched: dict[str, tuple[bool, float, Optional[int]]] = field(default_factory=dict)

    @property
    def answerable_accepted(self) -> int:
        return self.lenient.tp

    @property
    def answerable_rejected(self) -> int:
        return self.lenient.fn

    @property
    def absent_rejected(self) -> int:
        return self.lenient.tn

    @property
    def absent_accepted(self) -> int:
        return self.lenient.fp


def classify(*, is_absent: bool, has_evidence: bool, accepted: bool) -> tuple[str, str]:
    """Which confusion cell one decision falls in, as ``(lenient, strict)``.

    Extracted so every abstention experiment — absolute score, margin, anything
    later — scores decisions through one implementation. Two experiments that
    counted differently would not be comparable.
    """
    if is_absent:
        # no answer exists, so accepting is wrong under either accounting
        cell = "fp" if accepted else "tn"
        return cell, cell
    if accepted:
        # strict: an acceptance only counts if real evidence was retrieved
        return "tp", ("tp" if has_evidence else "fp")
    # abstaining when nothing relevant was retrieved is the right call
    return "fn", ("fn" if has_evidence else "tn")


def calibrate(
    results: Sequence[QuestionResult],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    *,
    watch: Sequence[str] = ("X1", "X2", "X3", "P5", "E1", "E6"),
) -> list[CalibrationPoint]:
    """Sweep abstention thresholds over an already-ranked result set.

    ``results`` must come from a reranked run: ``top_score`` is the value gated
    on, so it has to be a cross-encoder score.
    """
    points: list[CalibrationPoint] = []
    for t in thresholds:
        lenient, strict = ConfusionCounts(), ConfusionCounts()
        per_cat: dict[str, list[int]] = {}
        watched: dict[str, tuple[bool, float, Optional[int]]] = {}

        for r in results:
            accepted = r.top_score >= t
            has_evidence = r.first_relevant_rank is not None

            len_cell, strict_cell = classify(
                is_absent=r.is_absent, has_evidence=has_evidence, accepted=accepted)
            setattr(lenient, len_cell, getattr(lenient, len_cell) + 1)
            setattr(strict, strict_cell, getattr(strict, strict_cell) + 1)

            slot = per_cat.setdefault(r.category, [0, 0])
            slot[0] += int(accepted)
            slot[1] += 1
            if r.qid in watch:
                watched[r.qid] = (accepted, r.top_score, r.first_relevant_rank)

        points.append(CalibrationPoint(
            threshold=t, lenient=lenient, strict=strict,
            per_category={k: (v[0], v[1]) for k, v in per_cat.items()},
            watched=watched,
        ))
    return points


def best_by(points: Sequence[CalibrationPoint], metric: str = "youden_j",
            accounting: str = "strict") -> CalibrationPoint:
    def key(p: CalibrationPoint):
        val = getattr(getattr(p, accounting), metric)
        return (val if val is not None else float("-inf"), -p.threshold)

    return max(points, key=key)
