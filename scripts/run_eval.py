"""Print the retrieval baseline.

    python scripts/run_eval.py              # real embedder (the real baseline)
    python scripts/run_eval.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.dataset import CATEGORIES  # noqa: E402
from src.evaluation.metrics import threshold_analysis  # noqa: E402
from src.evaluation.runner import run_eval  # noqa: E402


def _f(x, spec=".4f", dash="  -   "):
    return format(x, spec) if x is not None else dash


def main() -> int:
    try:  # the corpus and headings contain non-cp1252 characters
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--embedding-model", default=None,
                    help="override EMBEDDING_MODEL (e.g. mock-64 for a fast smoke run)")
    ap.add_argument("--scale", choices=("sample", "large"), default="sample",
                    help="sample = tracked 7-chunk corpus; large = production scale")
    ap.add_argument("--retriever", choices=("vector", "bm25"), default="vector")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    run = run_eval(k=args.k, embedding_model=args.embedding_model, scale=args.scale,
                   retriever=args.retriever)
    s, c = run.summary, run.corpus

    print("=" * 78)
    print(f"RETRIEVAL EVAL — {c['retriever'].upper()} (no hybrid, no rerank)")
    print("=" * 78)
    print(f"corpus     : {c['scale']} scale - {c['messages']} messages / "
          f"{c['conversations']} conversations / {c['chunks']} chunks")
    print(f"embedder   : {c['embedding_model']}")
    print(f"chunking   : {c['chunk_size_messages']} msgs, "
          f"{c['chunk_overlap_messages']} overlap   probe k={c['probe_k']}")
    print(f"questions  : {s.n}  ({s.n_answerable} answerable, {s.n_absent} absent)")

    print("\n--- recall (answerable questions) ---")
    for k in sorted(s.recall):
        hits = round(s.recall[k] * s.n_answerable)
        print(f"  Recall@{k:<3d} {s.recall[k]*100:6.1f}%   ({hits}/{s.n_answerable})")
    print(f"  MRR       {s.mrr:6.3f}")

    print("\n--- latency (per query, retrieval only) ---")
    print(f"  mean {_f(s.latency_ms_mean, '.1f')} ms   "
          f"median {_f(s.latency_ms_median, '.1f')} ms   "
          f"p95 {_f(s.latency_ms_p95, '.1f')} ms")

    print("\n--- similarity scores ---")
    print(f"  correct chunk (mean)            {_f(s.correct_score_mean)}")
    print(f"  top irrelevant chunk (mean)     {_f(s.top_irrelevant_score_mean)}")
    print(f"  separation (correct - irrel.)   {_f(s.separation_mean)}")
    print(f"  ABSENT questions, top-1 (mean)  {_f(s.absent_top_score_mean)}")
    print(f"  ABSENT questions, top-1 (max)   {_f(s.absent_top_score_max)}")

    print("\n--- by category ---")
    print(f"  {'category':14s} {'n':>3s} {'R@1':>7s} {'R@3':>7s} {'R@5':>7s} "
          f"{'R@10':>7s} {'correct':>9s} {'irrel':>9s}")
    for cat in CATEGORIES:
        cs = run.per_category.get(cat)
        if not cs:
            continue
        if cs.n_answerable:
            r = [f"{cs.recall[k]*100:6.1f}%" for k in (1, 3, 5, 10)]
        else:
            r = ["    n/a"] * 4
        print(f"  {cat:14s} {cs.n:3d} {r[0]:>7s} {r[1]:>7s} {r[2]:>7s} {r[3]:>7s} "
              f"{_f(cs.correct_score_mean, '.4f', '     -   '):>9s} "
              f"{_f(cs.top_irrelevant_score_mean, '.4f', '     -   '):>9s}")

    print("\n--- per question ---")
    print(f"  {'qid':4s} {'cat':12s} {'rank':>5s} {'correct':>8s} {'irrel':>8s} "
          f"{'top1':>8s} {'ms':>6s}  question")
    for r in run.results:
        rank = "MISS" if (not r.is_absent and r.first_relevant_rank is None) else (
            "n/a" if r.is_absent else str(r.first_relevant_rank))
        print(f"  {r.qid:4s} {r.category:12s} {rank:>5s} "
              f"{_f(r.correct_score, '.4f', '    -   '):>8s} "
              f"{_f(r.top_irrelevant_score, '.4f', '    -   '):>8s} "
              f"{r.top_score:8.4f} {r.latency_ms:6.1f}  {r.question[:44]}")

    print("\n--- can an absolute score floor separate answerable from absent? ---")
    pts = threshold_analysis(run.results)
    print(f"  {'floor':>6s} {'answerable kept':>16s} {'absent rejected':>16s} {'Youden':>8s}")
    for p in pts:
        if p.threshold > 0.55:
            break
        star = "  <- MIN_RETRIEVAL_SCORE" if abs(p.threshold - 0.25) < 1e-9 else ""
        print(f"  {p.threshold:6.2f} {p.answerable_kept:>10d}/{p.answerable_total:<5d} "
              f"{p.absent_rejected:>10d}/{p.absent_total:<5d} {p.youden:8.3f}{star}")
    best = max(pts, key=lambda p: p.youden)
    print(f"  best separation at floor={best.threshold:.2f}: "
          f"keeps {best.answerable_kept}/{best.answerable_total} answerable, "
          f"rejects {best.absent_rejected}/{best.absent_total} absent "
          f"(Youden {best.youden:.3f})")

    misses = [r for r in run.results if not r.is_absent and r.first_relevant_rank is None]
    if misses:
        print(f"\n--- misses (not retrieved within k={c['probe_k']}) ---")
        for r in misses:
            print(f"  {r.qid} [{r.category}] {r.question}")

    if args.json:
        args.json.write_text(json.dumps({
            "corpus": c,
            "summary": vars(s),
            "per_category": {k: vars(v) for k, v in run.per_category.items()},
            "results": [vars(r) for r in run.results],
        }, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
