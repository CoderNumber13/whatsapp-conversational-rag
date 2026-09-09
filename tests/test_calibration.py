"""Abstention-threshold calibration maths.

Built on hand-constructed QuestionResults so every expected count can be worked
out by hand. Nothing here runs a model.
"""

from __future__ import annotations

import pytest

from src.evaluation.calibration import (
    DEFAULT_THRESHOLDS,
    ConfusionCounts,
    best_by,
    calibrate,
)
from src.evaluation.metrics import QuestionResult


def _q(qid, *, absent=False, top=0.0, rank=1, category="direct"):
    return QuestionResult(
        qid=qid, category=category, question="q", is_absent=absent, n_expected=1,
        first_relevant_rank=rank, correct_score=None, top_irrelevant_score=None,
        top_score=top, latency_ms=1.0, n_retrieved=10,
    )


# --- confusion maths --------------------------------------------------

def test_metrics_match_their_definitions():
    c = ConfusionCounts(tp=8, fn=2, tn=3, fp=1)
    assert c.precision == pytest.approx(8 / 9)
    assert c.recall == pytest.approx(8 / 10)
    assert c.specificity == pytest.approx(3 / 4)
    assert c.f1 == pytest.approx(2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8))
    assert c.youden_j == pytest.approx(0.8 + 0.75 - 1.0)
    assert c.false_acceptance_rate == pytest.approx(1 / 4)
    assert c.false_rejection_rate == pytest.approx(2 / 10)


def test_far_and_frr_are_the_complements_of_specificity_and_recall():
    c = ConfusionCounts(tp=5, fn=5, tn=7, fp=3)
    assert c.false_acceptance_rate == pytest.approx(1 - c.specificity)
    assert c.false_rejection_rate == pytest.approx(1 - c.recall)


def test_perfect_gate_scores_one():
    c = ConfusionCounts(tp=10, fn=0, tn=6, fp=0)
    assert c.precision == 1.0 and c.recall == 1.0
    assert c.youden_j == pytest.approx(1.0)
    assert c.false_acceptance_rate == 0.0 and c.false_rejection_rate == 0.0


def test_degenerate_counts_do_not_divide_by_zero():
    empty = ConfusionCounts()
    assert empty.precision is None and empty.recall is None
    assert empty.specificity is None and empty.youden_j is None
    no_absent = ConfusionCounts(tp=3, fn=1)
    assert no_absent.specificity is None and no_absent.youden_j is None
    assert no_absent.recall == pytest.approx(0.75)


# --- the gate ---------------------------------------------------------

def test_accepts_at_or_above_the_threshold():
    results = [_q("a", top=-3.0), _q("b", top=-3.01)]
    p = calibrate(results, [-3.0])[0]
    assert (p.lenient.tp, p.lenient.fn) == (1, 1), "boundary is inclusive"


def test_answerable_and_absent_are_counted_into_the_right_cells():
    results = [
        _q("a1", top=1.0), _q("a2", top=-9.0),               # answerable
        _q("z1", absent=True, rank=None, top=1.0),           # absent, accepted
        _q("z2", absent=True, rank=None, top=-9.0),          # absent, rejected
    ]
    p = calibrate(results, [0.0])[0]
    assert (p.answerable_accepted, p.answerable_rejected) == (1, 1)
    assert (p.absent_accepted, p.absent_rejected) == (1, 1)
    assert p.lenient.tp == 1 and p.lenient.fn == 1
    assert p.lenient.fp == 1 and p.lenient.tn == 1


def test_acceptance_is_monotone_in_the_threshold():
    results = [_q(f"a{i}", top=float(i)) for i in range(-10, 10)]
    accepted = [calibrate(results, [t])[0].answerable_accepted
                for t in sorted(DEFAULT_THRESHOLDS)]
    assert accepted == sorted(accepted, reverse=True), \
        "raising the threshold can never accept more questions"


# --- strict vs lenient ------------------------------------------------

def test_strict_counts_an_accept_without_evidence_as_a_false_acceptance():
    """The X1 case: answerable, accepted, but nothing relevant was retrieved —
    so the LLM would answer from irrelevant context."""
    results = [_q("x1", top=5.0, rank=None)]
    p = calibrate(results, [0.0])[0]
    assert p.lenient.tp == 1 and p.lenient.fp == 0
    assert p.strict.tp == 0 and p.strict.fp == 1


def test_strict_counts_abstaining_without_evidence_as_a_correct_refusal():
    results = [_q("x1", top=-9.0, rank=None)]
    p = calibrate(results, [0.0])[0]
    assert p.lenient.fn == 1 and p.lenient.tn == 0
    assert p.strict.fn == 0 and p.strict.tn == 1


def test_the_two_accountings_agree_when_evidence_is_always_retrieved():
    results = [_q("a", top=2.0), _q("b", top=-8.0),
               _q("z", absent=True, rank=None, top=-9.0)]
    p = calibrate(results, [0.0])[0]
    assert (p.lenient.tp, p.lenient.fn, p.lenient.fp) == \
        (p.strict.tp, p.strict.fn, p.strict.fp)


def test_absent_questions_are_scored_identically_under_both_accountings():
    results = [_q("z1", absent=True, rank=None, top=5.0),
               _q("z2", absent=True, rank=None, top=-5.0)]
    p = calibrate(results, [0.0])[0]
    assert (p.lenient.fp, p.lenient.tn) == (p.strict.fp, p.strict.tn) == (1, 1)


# --- per-category and watched -----------------------------------------

def test_per_category_counts_accepted_over_total():
    results = [
        _q("d1", top=1.0, category="direct"), _q("d2", top=-9.0, category="direct"),
        _q("e1", top=1.0, category="exact_term"),
    ]
    p = calibrate(results, [0.0])[0]
    assert p.per_category["direct"] == (1, 2)
    assert p.per_category["exact_term"] == (1, 1)


def test_watched_questions_record_decision_score_and_evidence_rank():
    results = [_q("X1", top=-7.2, rank=None), _q("E6", top=0.38, rank=1)]
    p = calibrate(results, [-3.0], watch=("X1", "E6"))[0]
    assert p.watched["X1"] == (False, -7.2, None)
    assert p.watched["E6"] == (True, 0.38, 1)


def test_unwatched_questions_are_not_recorded():
    p = calibrate([_q("D1", top=1.0)], [0.0], watch=("X1",))[0]
    assert p.watched == {}


# --- selection --------------------------------------------------------

def test_best_by_picks_the_highest_youden_and_can_differ_per_accounting():
    """The real finding: strict and lenient accounting disagree about the
    optimum, so which one is used is a decision, not a detail."""
    results = [
        # answerable, evidence retrieved, scored high
        _q("ok1", top=5.0, rank=1), _q("ok2", top=4.0, rank=2),
        # answerable, NO evidence retrieved, scored mid (the X1/X3 shape)
        _q("noev1", top=-2.0, rank=None), _q("noev2", top=-2.5, rank=None),
        # absent, scored low
        _q("ab1", absent=True, rank=None, top=-8.0),
        _q("ab2", absent=True, rank=None, top=-9.0),
    ]
    pts = calibrate(results, [-5.0, -1.0])

    # LENIENT sees the permissive floor as perfect: all 4 answerable accepted,
    # both absent rejected.
    assert best_by(pts, "youden_j", "lenient").threshold == -5.0
    # STRICT sees that two of those acceptances have no evidence behind them,
    # and prefers the stricter floor that refuses them.
    assert best_by(pts, "youden_j", "strict").threshold == -1.0

    lo = next(p for p in pts if p.threshold == -5.0)
    assert lo.lenient.youden_j == pytest.approx(1.0)
    assert lo.strict.youden_j == pytest.approx(0.5)


def test_calibrate_returns_one_point_per_threshold_in_order():
    pts = calibrate([_q("a", top=0.0)], DEFAULT_THRESHOLDS)
    assert [p.threshold for p in pts] == list(DEFAULT_THRESHOLDS)


def test_default_thresholds_cover_the_requested_range():
    for wanted in (-10, -8, -6, -5, -4, -3.5, -3.3, -3.0, -2.5, -2.0, -1.5, -1.0):
        assert float(wanted) in DEFAULT_THRESHOLDS


def test_calibration_never_reorders_anything():
    """It is an abstention gate: it reads top_score and changes no ranking."""
    results = [_q("a", top=1.0, rank=3), _q("b", top=-9.0, rank=1)]
    before = [(r.qid, r.first_relevant_rank, r.top_score) for r in results]
    calibrate(results, DEFAULT_THRESHOLDS)
    assert [(r.qid, r.first_relevant_rank, r.top_score) for r in results] == before
