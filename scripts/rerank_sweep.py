"""Cross-encoder reranking depth experiment.

    python scripts/rerank_sweep.py
    python scripts/rerank_sweep.py --depths 10 20 30 50 --json out.json

Ingests the corpus and builds vector + BM25 + RRF once, then varies only the
rerank candidate depth, so every configuration reranks the same RRF pool.
Experiment only — no production default is changed.
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
from src.evaluation.corpus import CREDENTIAL_TOKEN, generate_large_corpus  # noqa: E402
from src.evaluation.dataset import QUESTIONS, resolve_expectations  # noqa: E402
from src.evaluation.metrics import by_category, summarize  # noqa: E402
from src.evaluation.runner import SAMPLE_DIR, evaluate_questions  # noqa: E402

WATCH = ("X1", "X2", "X3", "P5", "E1", "E6")
RRF_BASELINE = dict(r1=65.8, r3=89.5, r5=92.1, r10=92.1, mrr=0.779, ex=42.9, cr=0.0)


def _rank(v):
    return "-" if v is None else str(v)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[10, 20, 30, 50])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="convmem-rerank-"))
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
        gold_cred = {c.chunk_id for c in pipe.db.all_chunks()
                     if CREDENTIAL_TOKEN in c.text}

        rrf = RRFHybridSearch.open(pipe.db, cfg)
        ce = CrossEncoderReranker(config=cfg)
        ce.warmup()
        Retriever(pipe.db, rrf, cfg).retrieve("warmup", k=1)

        print("=" * 96)
        print("CROSS-ENCODER RERANKING — depth experiment (experiment only)")
        print("=" * 96)
        print(f"corpus   : {len(messages)} messages / {pipe.db.count_chunks()} chunks")
        print(f"reranker : {cfg.rerank_model}  (max_length={cfg.rerank_max_length})")
        print(f"base     : RRF (K={cfg.rrf_k}, D={cfg.rrf_candidates})")

        hdr = (f"  {'depth':>5s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'R@10':>6s} "
               f"{'MRR':>6s} {'exact':>6s} {'cred':>6s} {'X1':>4s} {'P5':>4s} "
               f"{'E1':>4s} {'E6':>4s} {'ms':>7s} {'sep':>7s}")
        print("\n--- sweep ---")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        payload = []
        for depth in args.depths:
            searcher = RerankedSearch(rrf, ce, cfg, candidates=depth)
            results = evaluate_questions(
                Retriever(pipe.db, searcher, cfg), QUESTIONS, expected, 10)
            s, cats = summarize(results), by_category(results)
            ranks = {r.qid: r.first_relevant_rank for r in results}
            ex, cr = cats.get("exact_term"), cats.get("credential")
            print(f"  {depth:>5d} {s.recall[1]*100:>5.1f}% {s.recall[3]*100:>5.1f}% "
                  f"{s.recall[5]*100:>5.1f}% {s.recall[10]*100:>5.1f}% {s.mrr:>6.3f} "
                  f"{ex.recall[1]*100:>5.1f}% {cr.recall[1]*100:>5.1f}% "
                  f"{_rank(ranks['X1']):>4s} {_rank(ranks['P5']):>4s} "
                  f"{_rank(ranks['E1']):>4s} {_rank(ranks['E6']):>4s} "
                  f"{s.latency_ms_mean:>7.1f} "
                  f"{(s.separation_mean if s.separation_mean is not None else 0):>7.3f}")
            payload.append({
                "depth": depth, "summary": vars(s),
                "per_category": {k: vars(v) for k, v in cats.items()},
                "ranks": {q: ranks.get(q) for q in WATCH},
                "absent_top_mean": s.absent_top_score_mean,
                "absent_top_max": s.absent_top_score_max,
                "correct_mean": s.correct_score_mean,
            })

        print(f"\n  RRF baseline: R@1 {RRF_BASELINE['r1']}%  R@3 {RRF_BASELINE['r3']}%  "
              f"R@5 {RRF_BASELINE['r5']}%  R@10 {RRF_BASELINE['r10']}%  "
              f"MRR {RRF_BASELINE['mrr']}  exact {RRF_BASELINE['ex']}%  "
              f"cred {RRF_BASELINE['cr']}%  latency 21.2ms")

        print("\n--- absent-question behaviour (no threshold is applied) ---")
        print(f"  {'depth':>5s} {'correct mean':>13s} {'absent mean':>12s} "
              f"{'absent max':>11s} {'margin':>8s}")
        for p in payload:
            cm, am, ax = p["correct_mean"], p["absent_top_mean"], p["absent_top_max"]
            print(f"  {p['depth']:>5d} {cm:>13.3f} {am:>12.3f} {ax:>11.3f} "
                  f"{cm - ax:>8.3f}")

        # acceptance test 1: did the reranker even SEE the credential chunk?
        print("\n--- acceptance test: was the credential chunk in the rerank pool? ---")
        for depth in args.depths:
            pool = rrf.search("What is my gmail password?", depth)
            pos = [i for i, h in enumerate(pool, 1) if h.chunk.chunk_id in gold_cred]
            searcher = RerankedSearch(rrf, ce, cfg, candidates=depth)
            out = searcher.search("What is my gmail password?", depth)
            after = [i for i, h in enumerate(out, 1) if h.chunk.chunk_id in gold_cred]
            print(f"  depth {depth:>3d}: in RRF pool at {pos or 'ABSENT'} "
                  f"-> after reranking at {after or 'ABSENT'}")

        pipe.close()
        if args.json:
            args.json.write_text(json.dumps(payload, indent=2, default=str),
                                 encoding="utf-8")
            print(f"\nwrote {args.json}")
        return 0
    finally:
        Config.reload()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
