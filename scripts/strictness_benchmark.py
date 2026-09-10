"""Benchmark Strict vs Balanced vs Permissive.

    python scripts/strictness_benchmark.py                    # retrieval only (free)
    python scripts/strictness_benchmark.py --e2e --only X1 X3 A4 D1
    python scripts/strictness_benchmark.py --json out.json

Retrieval metrics need no LLM and cover all 44 questions. The end-to-end columns
(grounded accuracy, correct abstention, fabrication) need one LLM call per
question per mode, so ``--e2e`` takes a question subset: hosted free tiers cap
daily requests per model.

The corpus is ingested once and the component searchers built once, so the only
thing differing between modes is candidate depth.
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

from src.runtime import init_native_runtimes  # noqa: E402

init_native_runtimes()

from src.config import Config  # noqa: E402
from src.evaluation.corpus import generate_large_corpus  # noqa: E402
from src.evaluation.dataset import (  # noqa: E402
    QUESTIONS, answer_keys_for, resolve_expectations,
)
from src.evaluation.end_to_end import (  # noqa: E402
    ABSENT, WITH_EVIDENCE, WITHOUT_EVIDENCE, observe_answers, report,
)
from src.evaluation.metrics import by_category, summarize  # noqa: E402
from src.evaluation.runner import SAMPLE_DIR, evaluate_questions  # noqa: E402
from src.pipeline.strictness import MODES, all_profiles  # noqa: E402

WATCH = ("X1", "X2", "X3", "P5", "E1", "E6")


def _pct(x):
    return f"{x * 100:5.1f}%" if x is not None else "   -  "


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true", help="also run grounded answering")
    ap.add_argument("--only", nargs="+", default=["X1", "X3", "A4", "D1"],
                    help="questions for the --e2e pass")
    ap.add_argument("--model", default=None, help="override GEMINI_MODEL")
    ap.add_argument("--pace", type=float, default=5.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="convmem-strict-"))
    env = {"DB_PATH": str(tmp / "e.db"), "INDEX_DIR": str(tmp / "i"),
           "UPLOADS_DIR": str(tmp / "u")}
    if not args.e2e:
        env["LLM_PROVIDER"] = "mock"
    if args.model:
        env["GEMINI_MODEL"] = args.model
    os.environ.update(env)
    try:
        cfg = Config.reload()
        from src.pipeline.full_stack import GroundedAnswerer
        from src.pipeline.rag_pipeline import RagPipeline

        pipe = RagPipeline(cfg)
        pipe.ingest(generate_large_corpus(tmp / "c", SAMPLE_DIR), me_names=["Me"])
        messages = [m for cid in pipe.db.conversation_ids()
                    for m in pipe.db.get_conversation_messages(cid)]
        expected = resolve_expectations(QUESTIONS, messages)

        print("=" * 104)
        print("RETRIEVAL STRICTNESS - Strict vs Balanced vs Permissive")
        print("=" * 104)
        print(f"corpus : {len(messages)} messages / {pipe.db.count_chunks()} chunks, "
              f"{len(QUESTIONS)} questions")
        print("note   : strictness moves candidate DEPTH only. No stage's scores are "
              "thresholded, and")
        print("         MIN_RETRIEVAL_SCORE (cosine) never meets an RRF or "
              "cross-encoder value.")

        print("\n--- profiles (chunk counts, not scores) ---")
        print(f"  {'mode':11s} {'rrf pool':>9s} {'reranked':>9s} {'evidence->LLM':>14s}")
        profiles = all_profiles(cfg)
        for m in MODES:
            p = profiles[m]
            print(f"  {m:11s} {p.rrf_candidates:>9d} {p.rerank_candidates:>9d} "
                  f"{p.evidence_chunks:>14d}")

        payload: dict = {"profiles": {m: vars(profiles[m]) for m in MODES},
                         "modes": {}}

        print("\n--- retrieval (all 44 questions, no LLM) ---")
        header = (f"  {'mode':11s} {'R@1':>7s} {'R@3':>7s} {'R@5':>7s} {'R@10':>7s} "
                  f"{'MRR':>7s} {'exact':>7s} {'cred':>7s} "
                  + " ".join(f"{w:>4s}" for w in WATCH) + f" {'ms':>7s}")
        print(header)
        answerers = {}
        for m in MODES:
            a = GroundedAnswerer(pipe, strictness=m)
            answerers[m] = a
            res = evaluate_questions(a.pipeline.retriever, QUESTIONS, expected,
                                     max(10, a.top_n))
            s, cats = summarize(res), by_category(res)
            ranks = {r.qid: r.first_relevant_rank for r in res}
            watch_cells = " ".join(
                f"{(ranks[w] if ranks[w] else '-'):>4}" for w in WATCH)
            print(f"  {m:11s} {s.recall[1] * 100:>6.1f}% {s.recall[3] * 100:>6.1f}% "
                  f"{s.recall[5] * 100:>6.1f}% {s.recall[10] * 100:>6.1f}% "
                  f"{s.mrr:>7.3f} "
                  f"{cats['exact_term'].recall[1] * 100:>6.1f}% "
                  f"{cats['credential'].recall[1] * 100:>6.1f}% "
                  f"{watch_cells} {s.latency_ms_mean:>7.1f}")
            payload["modes"][m] = {
                "retrieval": vars(s),
                "per_category": {k: vars(v) for k, v in cats.items()},
                "ranks": {w: ranks[w] for w in WATCH},
                "evidence_chunks": a.top_n,
            }

        print("\n--- can the credential evidence REACH the grounded answerer? ---")
        print("  (rank within the evidence actually handed to the model)")
        print(f"  {'mode':11s} {'evidence':>9s} "
              + " ".join(f"{q:>13s}" for q in ("X1", "X2", "X3")))
        for m in MODES:
            a = answerers[m]
            cells = []
            for qid in ("X1", "X2", "X3"):
                q = next(x for x in QUESTIONS if x.qid == qid)
                hits = a.pipeline.retriever.retrieve(q.question, k=a.top_n)
                want = expected[qid]
                r = next((i for i, h in enumerate(hits, 1)
                          if want & set(h.chunk.message_ids)), None)
                cells.append(f"rank {r}" if r else "not reached")
            reach = " ".join(f"{c:>13s}" for c in cells)
            payload["modes"][m]["credential_reaches_llm"] = cells
            print(f"  {m:11s} {a.top_n:>9d} {reach}")

        if args.e2e:
            wanted = set(args.only)
            subset = [q for q in QUESTIONS if q.qid in wanted]
            print(f"\n--- end to end ({len(subset)} questions x {len(MODES)} modes, "
                  f"live LLM: {cfg.llm_provider}/{cfg.gemini_model}) ---")
            print(f"  {'mode':11s} {'grounded acc':>13s} {'correct abst':>13s} "
                  f"{'fabrication':>12s} {'unsupported':>12s} {'ms':>8s}")
            for m in MODES:
                a = answerers[m]
                def _note(attempt, exc, _m=m):
                    print(f"      [{_m}] retry {attempt}: {str(exc)[:80]}",
                          flush=True)

                obs = observe_answers(a, subset, expected, answer_keys_for,
                                      pace_s=args.pace, on_retry=_note)
                failed = [o.qid for o in obs if o.failed]
                rep = report(obs)
                we, wo, ab = rep[WITH_EVIDENCE], rep[WITHOUT_EVIDENCE], rep[ABSENT]
                should_refuse = wo.n + ab.n
                refused = wo.abstained + ab.abstained
                fabs = wo.fabrications + ab.fabrications
                unsup = sum(r.unsupported for r in rep.values())
                lat = [o.latency_ms for o in obs]
                abst_cell = f"{refused}/{should_refuse}"
                fab_cell = f"{fabs}/{should_refuse}"
                print(f"  {m:11s} {_pct(we.grounded_accuracy):>13s} "
                      f"{abst_cell:>13s} {fab_cell:>12s} {unsup:>12d} "
                      f"{sum(lat) / len(lat):>8.0f}"
                      + (f"   !! LLM FAILED: {failed}" if failed else ""))
                payload["modes"][m]["e2e"] = {
                    "grounded_accuracy": we.grounded_accuracy,
                    "grounded_success": we.grounded_success,
                    "with_evidence_n": we.n,
                    "correct_abstentions": refused,
                    "should_refuse": should_refuse,
                    "fabrications": fabs, "unsupported": unsup,
                    "leaked_gold": wo.leaked_gold,
                    "llm_failures": failed,
                    "answers": {o.qid: {"abstained": o.abstained,
                                        "grounded": o.grounded_success,
                                        "fabrication": o.fabrication,
                                        "contains_gold": o.contains_gold}
                                for o in obs},
                }
                for o in obs:
                    if o.qid in ("X1", "X2", "X3"):
                        verdict = "ABSTAINED" if o.abstained else "ANSWERED"
                        print(f"      {o.qid}: {verdict}  credential_stated="
                              f"{o.contains_gold}  fabrication={o.fabrication}")

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
