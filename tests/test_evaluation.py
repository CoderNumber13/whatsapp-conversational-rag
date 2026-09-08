"""The retrieval benchmark: metric maths, dataset integrity, and a smoke run.

These do not assert scores — scores are the measurement, and pinning them would
turn every retrieval improvement into a test failure. They assert that the
harness computes what it claims and that the question set still matches the
synthetic corpus.
"""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from src.evaluation.dataset import (
    ABSENT,
    CATEGORIES,
    CREDENTIAL,
    QUESTIONS,
    EvalQuestion,
    resolve_expectations,
)
from src.evaluation.corpus import CREDENTIAL_TOKEN, generate_large_corpus
from src.evaluation.metrics import QuestionResult, summarize, threshold_analysis
from src.evaluation.runner import SAMPLE_DIR, run_eval


def _r(qid, rank, *, absent=False, top=0.5, correct=None, irrel=None, ms=1.0):
    return QuestionResult(
        qid=qid, category="direct", question="q", is_absent=absent, n_expected=1,
        first_relevant_rank=rank, correct_score=correct, top_irrelevant_score=irrel,
        top_score=top, latency_ms=ms, n_retrieved=10,
    )


# --- metrics ----------------------------------------------------------

def test_recall_counts_a_hit_only_within_k():
    r = _r("a", 4)
    assert not r.hit_at(1) and not r.hit_at(3)
    assert r.hit_at(4) and r.hit_at(10)


def test_recall_and_mrr_over_a_known_set():
    results = [_r("a", 1), _r("b", 2), _r("c", None), _r("d", 5)]
    s = summarize(results, ks=(1, 3, 5, 10))
    assert s.n_answerable == 4
    assert s.recall[1] == pytest.approx(0.25)   # only 'a'
    assert s.recall[3] == pytest.approx(0.50)   # a, b
    assert s.recall[5] == pytest.approx(0.75)   # a, b, d
    assert s.recall[10] == pytest.approx(0.75)  # 'c' never retrieved
    # (1/1 + 1/2 + 0 + 1/5) / 4
    assert s.mrr == pytest.approx((1 + 0.5 + 0 + 0.2) / 4)


def test_absent_questions_are_excluded_from_recall():
    s = summarize([_r("a", 1), _r("z", None, absent=True, top=0.9)])
    assert s.n_answerable == 1 and s.n_absent == 1
    assert s.recall[1] == pytest.approx(1.0)      # the absent one must not dilute it
    assert s.absent_top_score_max == pytest.approx(0.9)


def test_separation_is_negative_when_an_irrelevant_chunk_outranks_the_answer():
    assert _r("a", 2, correct=0.30, irrel=0.42).separation == pytest.approx(-0.12)
    assert _r("b", 1, correct=0.50, irrel=0.20).separation == pytest.approx(0.30)
    assert _r("c", None).separation is None


def test_threshold_analysis_counts_both_populations():
    results = [
        _r("a", 1, top=0.40), _r("b", 1, top=0.20),          # answerable
        _r("y", None, absent=True, top=0.10),
        _r("z", None, absent=True, top=0.45),                # absent, scores high
    ]
    pts = {p.threshold: p for p in threshold_analysis(results, grid=[0.15, 0.30, 0.50])}
    assert (pts[0.15].answerable_kept, pts[0.15].absent_rejected) == (2, 1)
    assert (pts[0.30].answerable_kept, pts[0.30].absent_rejected) == (1, 1)
    assert (pts[0.50].answerable_kept, pts[0.50].absent_rejected) == (0, 2)
    # a floor that keeps everything separates nothing
    assert pts[0.15].youden == pytest.approx(2 / 2 + 1 / 2 - 1)


# --- dataset integrity ------------------------------------------------

def test_question_ids_are_unique():
    ids = [q.qid for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_question_set_covers_every_category_and_is_large_enough():
    assert 30 <= len(QUESTIONS) <= 50, "the brief asks for roughly 30-50 questions"
    present = {q.category for q in QUESTIONS}
    assert present == set(CATEGORIES), f"missing categories: {set(CATEGORIES) - present}"


def test_absent_questions_expect_nothing_and_others_expect_something():
    for q in QUESTIONS:
        if q.category == ABSENT:
            assert q.is_absent, f"{q.qid} is an ABSENT case but names expected messages"
        else:
            assert q.expect, f"{q.qid} is answerable but names no expected message"


def test_no_real_credential_in_the_benchmark():
    """The credential cases must stay synthetic."""
    for q in QUESTIONS:
        if q.category == CREDENTIAL:
            assert all(
                "sample" in e or "example.com" in e for e in q.expect
            ), f"{q.qid} expectation does not look synthetic"


def test_resolve_expectations_rejects_ambiguous_or_missing_substrings():
    class M:
        def __init__(self, mid, text):
            self.message_id, self.text = mid, text

    msgs = [M("m1", "hello there"), M("m2", "hello again")]

    with pytest.raises(ValueError, match="matched 2 messages"):
        resolve_expectations([EvalQuestion("Q", "direct", "?", ("hello",))], msgs)
    with pytest.raises(ValueError, match="matched 0 messages"):
        resolve_expectations([EvalQuestion("Q", "direct", "?", ("nope",))], msgs)

    ok = resolve_expectations([EvalQuestion("Q", "direct", "?", ("again",))], msgs)
    assert ok["Q"] == {"m2"}


# --- end-to-end -------------------------------------------------------

def test_benchmark_runs_and_expectations_still_match_the_corpus():
    """The guard against silent rot: every expected substring must still resolve
    to exactly one message in the synthetic corpus. run_eval raises otherwise."""
    run = run_eval(embedding_model="mock-64", k=5)

    assert len(run.results) == len(QUESTIONS)
    assert run.summary.n_answerable + run.summary.n_absent == len(QUESTIONS)
    assert run.corpus["chunks"] > 0
    for r in run.results:
        assert r.latency_ms >= 0
        assert r.n_retrieved <= 5
        if r.is_absent:
            assert r.first_relevant_rank is None


# --- production-scale corpus ------------------------------------------

def test_large_corpus_is_deterministic(tmp_path):
    """A benchmark corpus that changes between runs makes every comparison
    meaningless, so generation must be byte-for-byte reproducible."""
    a = generate_large_corpus(tmp_path / "a", SAMPLE_DIR)
    b = generate_large_corpus(tmp_path / "b", SAMPLE_DIR)
    assert [p.name for p in a] == [p.name for p in b]
    for pa, pb in zip(a, b):
        assert pa.read_text(encoding="utf-8") == pb.read_text(encoding="utf-8")


def test_large_corpus_is_deep_enough_for_recall_at_10(tmp_path):
    """The point of this corpus: k=10 must be a real top-10, not the whole set."""
    files = generate_large_corpus(tmp_path / "c", SAMPLE_DIR)
    total = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files)
    assert len(files) >= 12, "too few conversations to model a real export"
    assert total > 1000, "corpus too small for Recall@5/@10 to discriminate"


def test_large_corpus_keeps_the_sample_chats_verbatim(tmp_path):
    """Gold answers live in the tracked sample chats; copying them unchanged is
    what makes the two scales comparable."""
    files = generate_large_corpus(tmp_path / "c", SAMPLE_DIR)
    by_name = {f.name: f for f in files}
    for src in SAMPLE_DIR.glob("*.txt"):
        assert src.name in by_name
        assert by_name[src.name].read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_credential_is_buried_not_sitting_in_a_tiny_chat(tmp_path):
    """The small corpus fails to reproduce the bug because the credential owns a
    7-message conversation. Here it must be deep inside a long history."""
    files = generate_large_corpus(tmp_path / "c", SAMPLE_DIR)
    karan = next(f for f in files if "Karan" in f.name)
    lines = karan.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 60, "credential conversation is not production-length"
    idx = next(i for i, l in enumerate(lines) if CREDENTIAL_TOKEN in l)
    assert idx > 10, "credential is too close to the start to be buried"
    assert sum(CREDENTIAL_TOKEN in l for l in lines) == 1, "credential must appear once"


def test_password_distractors_exist_and_carry_no_credential(tmp_path):
    """The distractors are the mechanism being modelled: chunks that discuss
    credentials without containing one. Without them the failure vanishes."""
    files = generate_large_corpus(tmp_path / "c", SAMPLE_DIR)
    corpus = "\n".join(f.read_text(encoding="utf-8") for f in files)
    hits = [l for l in corpus.splitlines()
            if any(w in l.lower() for w in ("password", "login", "otp", "credential"))]
    assert len(hits) >= 10, "too few password distractors to create competition"
    assert not any(CREDENTIAL_TOKEN in l for l in hits), (
        "a distractor leaked the credential — it must never co-occur"
    )


def test_large_scale_benchmark_runs_end_to_end():
    """Expectations must still resolve uniquely against the bigger corpus."""
    run = run_eval(embedding_model="mock-64", k=10, scale="large")
    assert len(run.results) == len(QUESTIONS)
    assert run.corpus["scale"] == "large"
    assert run.corpus["chunks"] > 100, "not deep enough for k=10 to be meaningful"
