"""RRF parameter-sensitivity experiment.

    python scripts/rrf_sweep.py
    python scripts/rrf_sweep.py --json sweep.json

Experiment only — it does not change the production default (RRF_K=60, D=50).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.sweep import (  # noqa: E402
    DEFAULT_DEPTHS, DEFAULT_KS, max_rank_that_can_win, run_rrf_sweep,
    single_can_outrank_double,
)

BASELINES = {
    "vector": dict(r1=57.9, r3=73.7, r5=76.3, r10=92.1, mrr=0.682, ex=0.0, cr=0.0),
    "bm25": dict(r1=65.8, r3=81.6, r5=86.8, r10=89.5, mrr=0.749, ex=85.7, cr=0.0),
}


def _rank(v):
    return "-" if v is None else str(v)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    ap.add_argument("--embedding-model", default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    print("=" * 100)
    print("RRF PARAMETER SENSITIVITY — experiment only, production default unchanged")
    print("=" * 100)

    print("\n--- condition (1): can a ONE-retriever chunk ever outrank a TWO-retriever chunk? ---")
    print("  single@r = 1/(K+r)   double@(D,D) = 2/(K+D)   =>   r < (D-K)/2")
    print(f"  {'D\\K':>6s} " + " ".join(f"{k:>10d}" for k in args.ks))
    for d in args.depths:
        cells = []
        for k in args.ks:
            if single_can_outrank_double(k, d):
                cells.append(f"r<={max_rank_that_can_win(k, d):>3d}    ")
            else:
                cells.append("  NEVER   ")
        print(f"  {d:>6d} " + " ".join(f"{c:>10s}" for c in cells))
    print("  'NEVER' = agreement dominates absolutely (D <= K+2): no rank is good enough.")

    points, meta = run_rrf_sweep(
        ks=args.ks, depths=args.depths, embedding_model=args.embedding_model)

    print(f"\ncorpus: {meta['messages']} messages / {meta['conversations']} conversations "
          f"/ {meta['chunks']} chunks | {meta['questions']} questions | "
          f"probe k={meta['probe_k']}")
    print(f"embedder: {meta['embedding_model']}")

    print("\n--- sweep ---")
    hdr = (f"  {'D':>4s} {'K':>4s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'R@10':>6s} "
           f"{'MRR':>6s} {'exact':>6s} {'cred':>6s} "
           f"{'X1':>4s} {'P5':>4s} {'E1':>4s} {'E6':>4s} {'ms':>6s} {'dom':>4s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    prev_depth = None
    for p in points:
        if prev_depth is not None and p.depth != prev_depth:
            print("  " + "-" * (len(hdr) - 2))
        prev_depth = p.depth
        s = p.summary
        ex = p.per_category.get("exact_term")
        cr = p.per_category.get("credential")
        print(f"  {p.depth:>4d} {p.k_rrf:>4d} "
              f"{s.recall[1]*100:>5.1f}% {s.recall[3]*100:>5.1f}% "
              f"{s.recall[5]*100:>5.1f}% {s.recall[10]*100:>5.1f}% "
              f"{s.mrr:>6.3f} "
              f"{(ex.recall[1]*100 if ex else 0):>5.1f}% {(cr.recall[1]*100 if cr else 0):>5.1f}% "
              f"{_rank(p.ranks['X1']):>4s} {_rank(p.ranks['P5']):>4s} "
              f"{_rank(p.ranks['E1']):>4s} {_rank(p.ranks['E6']):>4s} "
              f"{s.latency_ms_mean:>6.1f} {('Y' if p.dominance else 'n'):>4s}")

    print("\n  dom=Y: agreement dominates rank absolutely (no single-retriever chunk can win)")

    print("\n--- reference baselines (unchanged) ---")
    for name, b in BASELINES.items():
        print(f"  {name:8s} R@1 {b['r1']:5.1f}%  R@3 {b['r3']:5.1f}%  R@5 {b['r5']:5.1f}%  "
              f"R@10 {b['r10']:5.1f}%  MRR {b['mrr']:.3f}  "
              f"exact {b['ex']:5.1f}%  cred {b['cr']:4.1f}%")

    print("\n--- per-category Recall@1 across K (at each depth) ---")
    cats = ["direct", "paraphrase", "contextual", "multi_message", "exact_term", "credential"]
    for d in args.depths:
        print(f"\n  depth D={d}")
        print(f"    {'K':>4s} " + " ".join(f"{c[:9]:>10s}" for c in cats))
        for p in [x for x in points if x.depth == d]:
            cells = []
            for c in cats:
                cs = p.per_category.get(c)
                cells.append(f"{cs.recall[1]*100:>9.1f}%" if cs and cs.n_answerable else "        -")
            print(f"    {p.k_rrf:>4d} " + " ".join(f"{c:>10s}" for c in cells))

    if args.json:
        args.json.write_text(json.dumps({
            "meta": meta,
            "points": [{
                "k_rrf": p.k_rrf, "depth": p.depth, "dominance": p.dominance,
                "winnable_rank": p.winnable_rank, "ranks": p.ranks,
                "summary": vars(p.summary),
                "per_category": {k: vars(v) for k, v in p.per_category.items()},
            } for p in points],
        }, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
