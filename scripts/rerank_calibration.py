"""Cross-encoder abstention-threshold calibration.

    python scripts/rerank_calibration.py
    python scripts/rerank_calibration.py --json calib.json

Experiment only. Nothing is applied; MIN_RETRIEVAL_SCORE is untouched and is a
cosine threshold unrelated to these scores.
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
from src.evaluation.calibration import DEFAULT_THRESHOLDS, best_by, calibrate  # noqa: E402
from src.evaluation.corpus import generate_large_corpus  # noqa: E402
from src.evaluation.dataset import CATEGORIES, QUESTIONS, resolve_expectations  # noqa: E402
from src.evaluation.runner import SAMPLE_DIR, evaluate_questions  # noqa: E402

WATCH = ("X1", "X2", "X3", "P5", "E1", "E6")


def _f(x, spec="6.3f"):
    return format(x, spec) if x is not None else "   -  "


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=list(DEFAULT_THRESHOLDS))
    ap.add_argument("--rerank-depth", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="convmem-calib-"))
    os.environ.update({
        "DB_PATH": str(tmp / "e.db"), "INDEX_DIR": str(tmp / "i"),
        "UPLOADS_DIR": str(tmp / "u"), "LLM_PROVIDER": "mock",
        "MIN_RETRIEVAL_SCORE": "-1",     # the cosine gate is out of the way here
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
        searcher = RerankedSearch(RRFHybridSearch.open(pipe.db, cfg), ce, cfg,
                                  candidates=depth)
        results = evaluate_questions(
            Retriever(pipe.db, searcher, cfg), QUESTIONS, expected, 10)

        print("=" * 104)
        print("CROSS-ENCODER ABSTENTION CALIBRATION — experiment only, nothing applied")
        print("=" * 104)
        print(f"corpus   : {len(messages)} messages / {pipe.db.count_chunks()} chunks")
        print(f"reranker : {cfg.rerank_model}, top-{depth}")
        print(f"gate     : accept if top-1 cross-encoder score >= threshold "
              f"(ranking untouched)")
        print(f"note     : MIN_RETRIEVAL_SCORE={Config().min_retrieval_score} is a "
              f"cosine threshold and is NOT used here")

        answerable = [r for r in results if not r.is_absent]
        absent = [r for r in results if r.is_absent]
        no_evidence = [r for r in answerable if r.first_relevant_rank is None]
        print(f"\nquestions: {len(results)} = {len(answerable)} answerable "
              f"+ {len(absent)} absent")
        print(f"  of the answerable, {len(no_evidence)} have NO relevant chunk "
              f"retrieved at all: {sorted(r.qid for r in no_evidence)}")
        print("  accepting those cannot produce a grounded answer, which is why "
              "STRICT accounting counts them as false acceptances.")

        points = calibrate(results, args.thresholds, watch=WATCH)

        for label, acc in (("LENIENT (accepted & answerable = success)", "lenient"),
                           ("STRICT (also requires evidence retrieved)", "strict")):
            print(f"\n--- {label} ---")
            print(f"  {'thr':>6s} {'ansAcc':>7s} {'ansRej':>7s} {'absRej':>7s} "
                  f"{'absAcc':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s} "
                  f"{'YoudenJ':>8s} {'FAR':>7s} {'FRR':>7s}")
            for p in points:
                c = getattr(p, acc)
                print(f"  {p.threshold:>6.1f} {c.tp:>7d} {c.fn:>7d} {c.tn:>7d} "
                      f"{c.fp:>7d} {_f(c.precision):>7s} {_f(c.recall):>7s} "
                      f"{_f(c.f1):>7s} {_f(c.youden_j):>8s} "
                      f"{_f(c.false_acceptance_rate):>7s} "
                      f"{_f(c.false_rejection_rate):>7s}")
            b = best_by(points, "youden_j", acc)
            bc = getattr(b, acc)
            print(f"  best Youden J = {_f(bc.youden_j)} at threshold {b.threshold}")

        print("\n--- acceptance by category (accepted / total) ---")
        cats = [c for c in CATEGORIES]
        print(f"  {'thr':>6s} " + " ".join(f"{c[:12]:>13s}" for c in cats))
        for p in points:
            cells = []
            for c in cats:
                a, n = p.per_category.get(c, (0, 0))
                cells.append(f"{a}/{n}")
            print(f"  {p.threshold:>6.1f} " + " ".join(f"{c:>13s}" for c in cells))

        print("\n--- watched questions (accept / top-1 score / rank of evidence) ---")
        print(f"  {'thr':>6s} " + " ".join(f"{q:>16s}" for q in WATCH))
        for p in points:
            cells = []
            for q in WATCH:
                acc, score, rank = p.watched[q]
                cells.append(f"{'ACC' if acc else 'abs'} {score:6.2f} "
                             f"r{rank if rank else '-'}")
            print(f"  {p.threshold:>6.1f} " + " ".join(f"{c:>16s}" for c in cells))

        pipe.close()
        if args.json:
            args.json.write_text(json.dumps([{
                "threshold": p.threshold,
                "lenient": {k: getattr(p.lenient, k) for k in
                            ("tp", "fn", "tn", "fp", "precision", "recall", "f1",
                             "youden_j", "false_acceptance_rate",
                             "false_rejection_rate")},
                "strict": {k: getattr(p.strict, k) for k in
                           ("tp", "fn", "tn", "fp", "precision", "recall", "f1",
                            "youden_j", "false_acceptance_rate",
                            "false_rejection_rate")},
                "per_category": p.per_category,
                "watched": {k: list(v) for k, v in p.watched.items()},
            } for p in points], indent=2, default=str), encoding="utf-8")
            print(f"\nwrote {args.json}")
        return 0
    finally:
        Config.reload()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
