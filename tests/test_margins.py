"""Margin-based abstention signals.

Hand-built observations only; nothing here runs a model.
"""

from __future__ import annotations

import pytest

from src.evaluation.calibration import ConfusionCounts, calibrate, classify
from src.evaluation.margins import (
    SIGNALS,
    MarginObservation,
    candidate_thresholds,
    distribution,
    overlap,
    sweep,
)
from src.evaluation.metrics import QuestionResult


def _o(qid, top1, top2=None, top3=None, *, absent=False, rank=1,
       category="direct"):
    return MarginObservation(
        qid=qid, category=category, is_absent=absent,
        top1=top1, top2=top2, top3=top3,
        has_evidence=rank is not None, evidence_rank=rank,
    )


# --- signal arithmetic ------------------------------------------------

def test_margins_are_differences_from_the_top_score():
    o = _o("a", 5.0, 1.0, -2.0)
    assert o.abs_top1 == 5.0
    assert o.margin_12 == pytest.approx(4.0)
    assert o.margin_13 == pytest.approx(7.0)


def test_margins_are_none_when_there_are_too_few_candidates():
    assert _o("a", 5.0).margin_12 is None
    assert _o("a", 5.0, 1.0).margin_13 is None
    assert _o("a", 5.0, 1.0).margin_12 == pytest.approx(4.0)
    empty = _o("a", None)
    assert empty.abs_top1 is None and empty.margin_12 is None


def test_signal_lookup_rejects_unknown_names():
    o = _o("a", 1.0, 0.0, -1.0)
    assert o.signal("margin_12") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown signal"):
        o.signal("margin_99")


def test_all_declared_signals_are_readable():
    o = _o("a", 3.0, 2.0, 1.0)
    assert all(o.signal(s) is not None for s in SIGNALS)


# --- the gate ---------------------------------------------------------

def test_acceptance_is_inclusive_at_the_threshold():
    obs = [_o("a", 5.0, 3.0), _o("b", 5.0, 3.01)]   # margins 2.0 and 1.99
    p = sweep(obs, "margin_12", [2.0])[0]
    assert p.accepted_qids == {"a"}


def test_a_question_with_no_candidates_is_never_accepted():
    """A missing signal must abstain rather than crash or accept."""
    obs = [_o("none", None, rank=None), _o("ok", 9.0, 0.0)]
    p = sweep(obs, "margin_12", [-999.0])[0]
    assert "none" not in p.accepted_qids
    assert "ok" in p.accepted_qids


def test_sweep_scores_through_the_same_classifier_as_calibration():
    """Margin and absolute-score experiments must be directly comparable, which
    requires identical accounting — so both go through classify()."""
    cases = [
        ("ans_ev", False, True, 5.0),
        ("ans_noev", False, False, 4.0),
        ("absent", True, False, 3.0),
        ("ans_ev_low", False, True, -9.0),
    ]
    obs = [_o(qid, top1, 0.0, rank=(1 if ev else None), absent=ab)
           for qid, ab, ev, top1 in cases]
    results = [QuestionResult(
        qid=qid, category="direct", question="q", is_absent=ab, n_expected=1,
        first_relevant_rank=(1 if ev else None), correct_score=None,
        top_irrelevant_score=None, top_score=top1, latency_ms=1.0, n_retrieved=3)
        for qid, ab, ev, top1 in cases]

    mp = sweep(obs, "abs_top1", [0.0])[0]
    cp = calibrate(results, [0.0])[0]
    for cell in ("tp", "fn", "tn", "fp"):
        assert getattr(mp.strict, cell) == getattr(cp.strict, cell)
        assert getattr(mp.lenient, cell) == getattr(cp.lenient, cell)


def test_answerable_without_evidence_is_a_false_accept_under_strict():
    """X1's shape: answerable, big margin, but nothing relevant retrieved."""
    obs = [_o("x1", 0.0, -5.0, rank=None)]      # margin 5.0
    p = sweep(obs, "margin_12", [1.0])[0]
    assert p.accepted_qids == {"x1"}
    assert p.lenient.tp == 1 and p.lenient.fp == 0
    assert p.strict.tp == 0 and p.strict.fp == 1


# --- threshold grid ---------------------------------------------------

def test_candidate_thresholds_are_midpoints_between_observed_values():
    obs = [_o("a", 1.0, 0.0), _o("b", 4.0, 0.0), _o("c", 8.0, 0.0)]
    grid = candidate_thresholds(obs, "margin_12")   # margins 1, 4, 8
    assert 2.5 in grid and 6.0 in grid
    assert min(grid) < 1.0 and max(grid) > 8.0


def test_candidate_thresholds_separate_every_distinct_outcome():
    obs = [_o(str(i), float(i), 0.0) for i in range(1, 6)]
    grid = candidate_thresholds(obs, "margin_12")
    sizes = {len(p.accepted_qids) for p in sweep(obs, "margin_12", grid)}
    assert sizes == {0, 1, 2, 3, 4, 5}, "grid must reach every acceptance count"


def test_candidate_thresholds_are_capped():
    obs = [_o(str(i), float(i) / 10, 0.0) for i in range(300)]
    assert len(candidate_thresholds(obs, "margin_12", limit=20)) <= 20


def test_candidate_thresholds_of_an_empty_signal_is_empty():
    assert candidate_thresholds([_o("a", 1.0)], "margin_12") == []


# --- distributions and overlap ---------------------------------------

def test_distribution_reports_order_statistics():
    d = distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (d.n, d.lo, d.median, d.hi) == (5, 1.0, 3.0, 5.0)
    assert d.mean == pytest.approx(3.0)
    assert d.q1 == pytest.approx(2.0) and d.q3 == pytest.approx(4.0)


def test_distribution_of_nothing_is_empty_not_an_error():
    assert distribution([]).n == 0
    assert distribution([]).median is None


def test_overlap_detects_clean_separation():
    obs = [_o("p1", 9.0, 0.0), _o("p2", 8.0, 0.0),          # positives, margins 9/8
           _o("n1", 1.0, 0.0, absent=True, rank=None)]      # negative, margin 1
    ov = overlap(obs, "margin_12")
    assert ov["clean"] is True
    assert ov["positives_below_worst_negative"] == 0
    assert ov["max_positives_at_zero_false_accept"] == 2


def test_overlap_detects_interleaving_and_reports_the_ceiling():
    obs = [_o("p1", 9.0, 0.0), _o("p2", 0.5, 0.0),          # margins 9 and 0.5
           _o("n1", 1.0, 0.0, absent=True, rank=None)]      # margin 1
    ov = overlap(obs, "margin_12")
    assert ov["clean"] is False
    assert ov["positives_below_worst_negative"] == 1
    assert ov["max_positives_at_zero_false_accept"] == 1


def test_overlap_ignores_answerable_questions_without_evidence():
    """They are not positives: there is nothing to keep."""
    obs = [_o("p1", 9.0, 0.0),
           _o("noev", 8.0, 0.0, rank=None),
           _o("n1", 1.0, 0.0, absent=True, rank=None)]
    ov = overlap(obs, "margin_12")
    assert ov["positives"] == 1


def test_overlap_with_one_population_missing_is_undefined_not_wrong():
    ov = overlap([_o("p1", 9.0, 0.0)], "margin_12")
    assert ov["clean"] is None


# --- the ceiling is a real bound --------------------------------------

def test_zero_false_accept_ceiling_is_actually_achievable_and_not_exceeded():
    obs = [_o("p1", 9.0, 0.0), _o("p2", 5.0, 0.0), _o("p3", 0.5, 0.0),
           _o("n1", 1.0, 0.0, absent=True, rank=None),
           _o("n2", 0.2, 0.0, absent=True, rank=None)]
    ov = overlap(obs, "margin_12")
    ceiling = ov["max_positives_at_zero_false_accept"]

    best = 0
    for p in sweep(obs, "margin_12", candidate_thresholds(obs, "margin_12")):
        if p.strict.fp == 0:
            best = max(best, p.strict.tp)
    assert best == ceiling
