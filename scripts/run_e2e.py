"""End-to-end grounded answering evaluation over the 44-question benchmark.

    python scripts/run_e2e.py                 # Gemini (the real stack)
    python scripts/run_e2e.py --llm mock      # no API calls, plumbing check
    python scripts/run_e2e.py --json e2e.json

Calls the configured LLM once per question. The corpus is synthetic; no real
credential is present or printed.
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
from src.evaluation.dataset import (  # noqa: E402
    CATEGORIES, QUESTIONS, answer_keys_for, resolve_expectations,
)
from src.evaluation.end_to_end import (  # noqa: E402
    ABSENT, GROUPS, WITH_EVIDENCE, WITHOUT_EVIDENCE, observe_answers, report,
)
from src.evaluation.metrics import by_category, summarize  # noqa: E402
from src.evaluation.runner import SAMPLE_DIR, evaluate_questions  # noqa: E402

WATCH = ("X1", "X2", "X3", "P5", "P8", "E1", "E6")
GROUP_LABEL = {
    WITH_EVIDENCE: "answerable, evidence RETRIEVED  (should answer, grounded)",
    WITHOUT_EVIDENCE: "answerable, evidence NOT retrieved (should REFUSE)",
    ABSENT: "no answer exists at all           (should REFUSE)",
}


def _pct(x):
    return f"{x*100:5.1f}%" if x is not None else "   -  "


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None, help="override LLM_PROVIDER")
    ap.add_argument("--rerank-depth", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=None,
                    help="evidence chunks handed to the LLM")
    ap.add_argument("--pace", type=float, default=4.5,
                    help="seconds between LLM calls; hosted free tiers "
                         "rate-limit per minute")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these qids (hosted free tiers cap daily "
                         "requests per model, so the full 44 may not fit)")
    ap.add_argument("--model", default=None, help="override GEMINI_MODEL")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="convmem-e2e-"))
    env = {"DB_PATH": str(tmp / "e.db"), "INDEX_DIR": str(tmp / "i"),
           "UPLOADS_DIR": str(tmp / "u")}
    if args.llm:
        env["LLM_PROVIDER"] = args.llm
    if args.model:
        env["GEMINI_MODEL"] = args.model
    os.environ.update(env)
    try:
        cfg = Config.reload()
        from src.pipeline.full_stack import GroundedAnswerer
        from src.pipeline.rag_pipeline import RagPipeline
        from src.retrieval.retriever import Retriever

        pipe = RagPipeline(cfg)
        pipe.ingest(generate_large_corpus(tmp / "c", SAMPLE_DIR), me_names=["Me"])
        messages = [m for cid in pipe.db.conversation_ids()
                    for m in pipe.db.get_conversation_messages(cid)]
        expected = resolve_expectations(QUESTIONS, messages)
        questions = ([q for q in QUESTIONS if q.qid in set(args.only)]
                     if args.only else list(QUESTIONS))

        answerer = GroundedAnswerer(pipe, top_n=args.top_n,
                                    rerank_depth=args.rerank_depth)

        print("=" * 100)
        print("END-TO-END GROUNDED ANSWERING")
        print("=" * 100)
        print(f"corpus    : {len(messages)} messages / {pipe.db.count_chunks()} chunks")
        print("stack     : vector + BM25 -> RRF -> cross-encoder -> LLM")
        print(f"reranker  : {cfg.rerank_model}, top-"
              f"{args.rerank_depth or cfg.rerank_candidates}")
        print(f"LLM       : {cfg.llm_provider}"
              f"{' / ' + cfg.gemini_model if cfg.llm_provider == 'gemini' else ''}")
        print(f"evidence  : top-{answerer.top_n} chunks to the LLM")
        print("abstention: left to the grounded prompt. No cross-encoder score or "
              "margin threshold is applied (experiments 5-6).")

        # retrieval quality of the very same stack, for context
        rr = evaluate_questions(answerer.pipeline.retriever, questions, expected, 10)
        s, cats = summarize(rr), by_category(rr)
        print(f"\nretrieval : Recall@1 {s.recall[1]*100:.1f}%  "
              f"Recall@3 {s.recall[3]*100:.1f}%  MRR {s.mrr:.3f}")

        def _retry_note(attempt, exc):
            print(f"    retry {attempt}: {str(exc)[:90]}", flush=True)

        obs = observe_answers(answerer, questions, expected, answer_keys_for,
                              on_retry=_retry_note, pace_s=args.pace)
        failed = [o for o in obs if o.failed]
        if failed:
            print(f"\n  !! {len(failed)} question(s) failed after retries: "
                  f"{[o.qid for o in failed]}")
        reports = report(obs)
        by_qid = {o.qid: o for o in obs}

        print("\n" + "=" * 100)
        print("RESULTS BY GROUP  (STRICT — primary)")
        print("=" * 100)
        for g in GROUPS:
            r = reports[g]
            print(f"\n[{g}]  n={r.n}   {GROUP_LABEL[g]}")
            if g == WITH_EVIDENCE:
                print(f"  grounded answer accuracy   {_pct(r.grounded_accuracy)}"
                      f"   ({r.grounded_success}/{r.answered} answered)")
                print(f"  grounded-answer rate       {_pct(r.grounded_answer_rate)}"
                      f"   ({r.answered}/{r.n})")
                print(f"  citation correctness       {_pct(r.citation_correctness)}"
                      f"   ({r.cited_evidence}/{r.answered} cite real evidence)")
                print(f"  stated the gold fact       {_pct(r._rate(r.contains_gold))}")
                print(f"  wrongly abstained          {_pct(r.correct_abstention_rate)}"
                      f"   ({r.abstained}/{r.n})  <- false refusals here")
            else:
                print(f"  correct abstention rate    {_pct(r.correct_abstention_rate)}"
                      f"   ({r.abstained}/{r.n})")
                print(f"  fabrication rate           {_pct(r.fabrication_rate)}"
                      f"   ({r.fabrications}/{r.n})")
                if g == WITHOUT_EVIDENCE:
                    print(f"  leaked the gold value      "
                          f"{_pct(r._rate(r.leaked_gold))}   ({r.leaked_gold}/{r.n})"
                          f"  <- credential fabrication")
            print(f"  unsupported-answer rate    {_pct(r.unsupported_rate)}"
                  f"   ({r.unsupported}/{r.n})")

        answered = sum(r.answered for r in reports.values())
        should_refuse = reports[WITHOUT_EVIDENCE].n + reports[ABSENT].n
        refused_ok = (reports[WITHOUT_EVIDENCE].abstained + reports[ABSENT].abstained)
        fabrications = sum(reports[g].fabrications for g in (WITHOUT_EVIDENCE, ABSENT))
        lat = sorted(o.latency_ms for o in obs)
        print("\n" + "=" * 100)
        print("OVERALL")
        print("=" * 100)
        print(f"  questions                  {len(obs)}")
        print(f"  grounded successes         {reports[WITH_EVIDENCE].grounded_success}"
              f"/{reports[WITH_EVIDENCE].n} of answerable-with-evidence")
        print(f"  correct abstentions        {refused_ok}/{should_refuse} "
              f"of should-refuse")
        print(f"  fabrications               {fabrications}/{should_refuse}")
        print(f"  answers produced           {answered}/{len(obs)}")
        print(f"  latency  mean {sum(lat)/len(lat):7.0f} ms   "
              f"median {lat[len(lat)//2]:7.0f} ms   p95 {lat[int(0.95*(len(lat)-1))]:7.0f} ms")

        print("\n--- watched questions ---")
        for q in [w for w in WATCH if w in by_qid]:
            o = by_qid[q]
            verdict = ("ABSTAINED" if o.abstained else
                       ("GROUNDED" if o.grounded_success else "answered"))
            print(f"\n  {o.qid} [{o.group}] rank={o.evidence_rank or '-'}  {verdict}")
            print(f"     gold_stated={o.contains_gold} cited={len(o.cited_message_ids)} "
                  f"citations_valid={o.citations_valid} fabrication={o.fabrication}")
            print(f"     {o.answer_text[:150].replace(chr(10), ' ')}")

        print("\n--- per question ---")
        print(f"  {'qid':4s} {'group':17s} {'rank':>4s} {'abst':>5s} {'gold':>5s} "
              f"{'cites':>5s} {'valid':>5s} {'OK':>3s} {'ms':>6s}")
        for o in obs:
            print(f"  {o.qid:4s} {o.group:17s} {str(o.evidence_rank or '-'):>4s} "
                  f"{('yes' if o.abstained else 'no'):>5s} "
                  f"{('yes' if o.contains_gold else 'no'):>5s} "
                  f"{len(o.cited_message_ids):>5d} "
                  f"{('yes' if o.citations_valid else 'no'):>5s} "
                  f"{('OK' if o.grounded_success or o.correct_abstention else ''):>3s} "
                  f"{o.latency_ms:6.0f}")

        print("\n--- by question category ---")
        print(f"  {'category':14s} {'n':>3s} {'grounded':>9s} {'abstain':>8s} "
              f"{'fabricate':>10s}")
        for cat in CATEGORIES:
            rows = [o for o in obs if o.category == cat]
            if not rows:
                continue
            print(f"  {cat:14s} {len(rows):>3d} "
                  f"{sum(o.grounded_success for o in rows):>9d} "
                  f"{sum(o.abstained for o in rows):>8d} "
                  f"{sum(o.fabrication for o in rows):>10d}")

        answerer.close()
        if args.json:
            args.json.write_text(json.dumps({
                "reports": {g: vars(reports[g]) for g in GROUPS},
                "observations": [{
                    "qid": o.qid, "category": o.category, "group": o.group,
                    "question": o.question, "evidence_rank": o.evidence_rank,
                    "retrieved_chunk_ids": o.retrieved_chunk_ids,
                    "answer": o.answer_text, "abstained": o.abstained,
                    "cited_message_ids": o.cited_message_ids,
                    "contains_gold": o.contains_gold,
                    "citations_valid": o.citations_valid,
                    "grounded_success": o.grounded_success,
                    "fabrication": o.fabrication, "latency_ms": o.latency_ms,
                } for o in obs],
            }, indent=2, default=str), encoding="utf-8")
            print(f"\nwrote {args.json}")
        return 0
    finally:
        Config.reload()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
