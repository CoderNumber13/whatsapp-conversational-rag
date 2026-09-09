"""Does within-query score separation beat the absolute cross-encoder score as
an abstention signal?

    python scripts/margin_experiment.py
    python scripts/margin_experiment.py --json margins.json

Experiment only. Nothing is applied and no production default changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.evaluation.corpus import generate_large_corpus  # noqa: E402
from src.evaluation.dataset import CATEGORIES, QUESTIONS, resolve_expectations  # noqa: E402
from src.evaluation.margins import (  # noqa: E402
    SIGNALS, candidate_thresholds, distribution, observe, overlap, sweep,
)
from src.evaluation.metrics import by_category, summarize  # noqa: E402
from src.evaluation.runner import SAMPLE_DIR, evaluate_questions  # noqa: E402

WATCH = ("X1", "X2", "X3", "P5", "P8", "E1", "E6")
LABEL = {"abs_top1": "absolute top-1 score (baseline, commit 1745171)",
         "margin_12": "margin  top1 - top2",
         "margin_13": "margin  top1 - top3"}


def _f(x, spec="6.3f"):
    return format(x, spec) if x is not None else "   -  "


def _dist_row(name, d):
    return (f"  {name:26s} n={d.n:<3d} min {_f(d.lo)} q1 {_f(d.q1)} "
            f"med {_f(d.median)} q3 {_f(d.q3)} max {_f(d.hi)} mean {_f(d.mean)}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-depth", type=int, default=None)
    ap.add_argument("--rows", type=int, default=14, help="threshold rows to print")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="convmem-margin-"))
    os.environ.update({
        "DB_PATH": str(tmp / "e.db"), "INDEX_DIR": str(tmp / "i"),
        "UPLOADS_DIR": str(tmp / "u"), "LLM_PROVIDER": "mock",
        "MIN_RETRIEVAL_SCORE": "-1",
    })
    try:
        cfg = Config.reload()
        from src.pipeline.rag_pipeline import RagPipeline
        from src.retrieval.hybrid_search import RRFHybridSearch
        from src.retrieval.reranker import CrossEncoderReranker, RerankedSearch
        from src.retrieval.retriever import Retriever

        pipe = RagPipeline(cfg)
        pipe.ingest(generate_large_corpus(tmp / "c", SAMPLE_DIR), me_names=["Me"])
        messages = [m for cid in pipe.db.conversation_ids()
                    for m in pipe.db.get_conversation_messages(cid)]
        expected = resolve_expectations(QUESTIONS, messages)

        ce = CrossEncoderReranker(config=cfg)
        ce.warmup()
        depth = args.rerank_depth or cfg.rerank_candidates
        retr = Retriever(
            pipe.db,
            RerankedSearch(RRFHybridSearch.open(pipe.db, cfg), ce, cfg,
                           candidates=depth),
            cfg)

        print("=" * 104)
        print("MARGIN vs ABSOLUTE SCORE as an abstention signal — experiment only")
        print("=" * 104)
        print(f"corpus   : {len(messages)} messages / {pipe.db.count_chunks()} chunks")
        print(f"reranker : {cfg.rerank_model}, top-{depth}")
        print("gate     : accept if SIGNAL >= threshold. Ranking untouched.")
        print(f"note     : MIN_RETRIEVAL_SCORE={Config().min_retrieval_score} unused "
              "(cosine threshold, unrelated)")

        # retrieval quality before any abstention
        results = evaluate_questions(retr, QUESTIONS, expected, 10)
        s, cats = summarize(results), by_category(results)
        print("\n--- retrieval quality BEFORE abstention (unchanged by this experiment) ---")
        print(f"  Recall@1 {s.recall[1]*100:.1f}%   Recall@3 {s.recall[3]*100:.1f}%   "
              f"Recall@5 {s.recall[5]*100:.1f}%   Recall@10 {s.recall[10]*100:.1f}%   "
              f"MRR {s.mrr:.3f}")
        print(f"  exact_term R@1 {cats['exact_term'].recall[1]*100:.1f}%   "
              f"credential R@1 {cats['credential'].recall[1]*100:.1f}%")

        obs = observe(retr, QUESTIONS, expected, 10)
        by_qid = {o.qid: o for o in obs}

        print("\n--- per-question signals ---")
        print(f"  {'qid':4s} {'cat':13s} {'top1':>8s} {'top2':>8s} {'top3':>8s} "
              f"{'m12':>7s} {'m13':>7s} {'evid':>5s} {'rank':>5s}")
        for o in obs:
            print(f"  {o.qid:4s} {o.category:13s} {_f(o.top1, '8.3f')} "
                  f"{_f(o.top2, '8.3f')} {_f(o.top3, '8.3f')} "
                  f"{_f(o.margin_12, '7.3f')} {_f(o.margin_13, '7.3f')} "
                  f"{('yes' if o.has_evidence else 'NO'):>5s} "
                  f"{str(o.evidence_rank or '-'):>5s}")

        print("\n--- do the two populations separate? ---")
        payload = {"signals": {}}
        for sig in SIGNALS:
            ov = overlap(obs, sig)
            print(f"\n  [{LABEL[sig]}]")
            print(_dist_row("answerable WITH evidence", ov["positive"]))
            print(_dist_row("absent (no answer)", ov["negative"]))
            print(f"    worst absent = {_f(ov['worst_negative'])}; "
                  f"{ov['positives_below_worst_negative']}/{ov['positives']} "
                  f"answerable-with-evidence fall below it")
            print(f"    CLEANLY SEPARABLE: {ov['clean']}   "
                  f"ceiling at zero false-accept: "
                  f"{ov['max_positives_at_zero_false_accept']}/{ov['positives']}")

        print("\n" + "=" * 104)
        print("STRICT accounting (production-relevant) — LENIENT shown after")
        print("=" * 104)

        best = {}
        for sig in SIGNALS:
            grid = candidate_thresholds(obs, sig)
            pts = sweep(obs, sig, grid)
            for acc in ("strict", "lenient"):
                b = max(pts, key=lambda p: (
                    getattr(getattr(p, acc), "youden_j") or float("-inf"),
                    -p.threshold))
                best[(sig, acc)] = b
            payload["signals"][sig] = {
                "overlap": {k: (vars(v) if hasattr(v, "__dict__") else v)
                            for k, v in overlap(obs, sig).items()},
                "best_strict": {
                    "threshold": best[(sig, "strict")].threshold,
                    **{k: getattr(best[(sig, "strict")].strict, k) for k in
                       ("tp", "fn", "tn", "fp", "precision", "recall", "f1",
                        "youden_j", "false_acceptance_rate",
                        "false_rejection_rate")}},
                "best_lenient": {
                    "threshold": best[(sig, "lenient")].threshold,
                    **{k: getattr(best[(sig, "lenient")].lenient, k) for k in
                       ("tp", "fn", "tn", "fp", "precision", "recall", "f1",
                        "youden_j", "false_acceptance_rate",
                        "false_rejection_rate")}},
            }

            for acc in ("strict", "lenient"):
                print(f"\n--- {LABEL[sig]} — {acc.upper()} ---")
                print(f"  {'thr':>7s} {'accept':>7s} {'reject':>7s} {'absRej':>7s} "
                      f"{'absAcc':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s} "
                      f"{'YoudenJ':>8s} {'FAR':>7s} {'FRR':>7s}")
                step = max(1, len(pts) // args.rows)
                shown = pts[::step]
                if best[(sig, acc)] not in shown:
                    shown = sorted(shown + [best[(sig, acc)]],
                                   key=lambda p: p.threshold)
                for p in shown:
                    c = getattr(p, acc)
                    star = "  <- best J" if p is best[(sig, acc)] else ""
                    print(f"  {p.threshold:>7.3f} {c.tp:>7d} {c.fn:>7d} {c.tn:>7d} "
                          f"{c.fp:>7d} {_f(c.precision):>7s} {_f(c.recall):>7s} "
                          f"{_f(c.f1):>7s} {_f(c.youden_j):>8s} "
                          f"{_f(c.false_acceptance_rate):>7s} "
                          f"{_f(c.false_rejection_rate):>7s}{star}")

        print("\n" + "=" * 104)
        print("HEAD TO HEAD — best achievable per signal")
        print("=" * 104)
        for acc in ("strict", "lenient"):
            print(f"\n  {acc.upper()}")
            print(f"    {'signal':<12s} {'thr':>8s} {'J':>7s} {'prec':>7s} "
                  f"{'recall':>7s} {'F1':>7s} {'FAR':>7s} {'FRR':>7s}")
            for sig in SIGNALS:
                b = best[(sig, acc)]
                c = getattr(b, acc)
                print(f"    {sig:<12s} {b.threshold:>8.3f} {_f(c.youden_j):>7s} "
                      f"{_f(c.precision):>7s} {_f(c.recall):>7s} {_f(c.f1):>7s} "
                      f"{_f(c.false_acceptance_rate):>7s} "
                      f"{_f(c.false_rejection_rate):>7s}")

        print("\n--- watched questions at each signal's best STRICT threshold ---")
        print(f"  {'qid':4s} {'evid':>5s} {'rank':>5s} " +
              " ".join(f"{sig:>22s}" for sig in SIGNALS))
        for q in WATCH:
            o = by_qid[q]
            cells = []
            for sig in SIGNALS:
                b = best[(sig, "strict")]
                v = o.signal(sig)
                cells.append(f"{'ACCEPT' if q in b.accepted_qids else 'abstain'} "
                             f"({_f(v, '6.2f')})")
            print(f"  {q:4s} {('yes' if o.has_evidence else 'NO'):>5s} "
                  f"{str(o.evidence_rank or '-'):>5s} " +
                  " ".join(f"{c:>22s}" for c in cells))

        print("\n--- category acceptance at each signal's best STRICT threshold ---")
        print(f"  {'category':14s} " + " ".join(f"{sig:>12s}" for sig in SIGNALS))
        for cat in CATEGORIES:
            in_cat = [o for o in obs if o.category == cat]
            if not in_cat:
                continue
            cells = []
            for sig in SIGNALS:
                b = best[(sig, "strict")]
                n = sum(1 for o in in_cat if o.qid in b.accepted_qids)
                cells.append(f"{n}/{len(in_cat)}")
            print(f"  {cat:14s} " + " ".join(f"{c:>12s}" for c in cells))

        pipe.close()
        if args.json:
            payload["observations"] = [{
                "qid": o.qid, "category": o.category, "is_absent": o.is_absent,
                "top1": o.top1, "top2": o.top2, "top3": o.top3,
                "margin_12": o.margin_12, "margin_13": o.margin_13,
                "has_evidence": o.has_evidence, "evidence_rank": o.evidence_rank,
            } for o in obs]
            args.json.write_text(json.dumps(payload, indent=2, default=str),
                                 encoding="utf-8")
            print(f"\nwrote {args.json}")
        return 0
    finally:
        Config.reload()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
