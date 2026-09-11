"""Regressions for the production-hardening increment.

Two classes of bug, both found by auditing rather than by a failing test:

1. FAISS and torch link different OpenMP runtimes; the second to initialise
   aborts the process with exit 3 and no traceback.
2. The Streamlit app fed a 0.0-1.0 *cosine* slider into ``answer(min_score=...)``
   while the retrieval stack now ends in a cross-encoder whose scores are
   unbounded logits, typically negative. Comparing them would have abstained on
   essentially every question.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# A workload that provably aborts without the mitigation: both libraries loaded,
# a batch FAISS search, torch compute, then FAISS again.
_WORKLOAD = """
import numpy as np, faiss, torch
i = faiss.IndexIDMap2(faiss.IndexFlatIP(384))
i.add_with_ids(np.random.rand(2000, 384).astype('float32'), np.arange(2000))
i.search(np.random.rand(32, 384).astype('float32'), 10)
_ = torch.randn(256, 256) @ torch.randn(256, 256)
i.search(np.random.rand(32, 384).astype('float32'), 10)
print('SURVIVED')
"""


def _run(code: str):
    """Run code in a fresh interpreter with the parent environment."""
    import os

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO, capture_output=True, text=True, env={**os.environ}, timeout=600,
    )


# --- the OpenMP mitigation -------------------------------------------

def test_init_native_runtimes_is_idempotent_and_reports_state():
    from src.runtime import init_native_runtimes, is_initialised

    first = init_native_runtimes()
    second = init_native_runtimes()
    assert is_initialised()
    assert first["initialised"] and second["initialised"]
    assert set(first["runtimes"]) == {"faiss", "torch"}


def test_init_sets_the_duplicate_omp_flag_when_unset(monkeypatch):
    from src import runtime

    monkeypatch.delenv(runtime.DUPLICATE_OMP_FLAG, raising=False)
    runtime._reset_for_tests()
    state = runtime.init_native_runtimes()
    assert state["omp_flag"] == "TRUE"
    assert state["flag_set_by_us"] is True
    runtime._reset_for_tests()
    runtime.init_native_runtimes()


def test_init_respects_an_existing_flag_value(monkeypatch):
    from src import runtime

    monkeypatch.setenv(runtime.DUPLICATE_OMP_FLAG, "FALSE")
    runtime._reset_for_tests()
    state = runtime.init_native_runtimes()
    assert state["omp_flag"] == "FALSE", "must not override a deliberate setting"
    assert state["flag_set_by_us"] is False
    runtime._reset_for_tests()
    runtime.init_native_runtimes()


def test_opting_out_of_the_workaround_does_not_set_the_flag(monkeypatch):
    from src import runtime

    monkeypatch.delenv(runtime.DUPLICATE_OMP_FLAG, raising=False)
    runtime._reset_for_tests()
    state = runtime.init_native_runtimes(allow_duplicate_omp=False)
    assert state["flag_set_by_us"] is False
    runtime._reset_for_tests()
    runtime.init_native_runtimes()


@pytest.mark.slow
def test_faiss_and_torch_coexist_after_init_native_runtimes():
    """The regression: this workload aborts the interpreter unmitigated."""
    r = _run("from src.runtime import init_native_runtimes\n"
             "init_native_runtimes()\n" + _WORKLOAD)
    assert "SURVIVED" in r.stdout, f"aborted: rc={r.returncode} err={r.stderr[:300]}"
    assert r.returncode == 0
    assert "OMP: Error" not in r.stderr


@pytest.mark.slow
def test_the_mitigation_is_load_bearing_not_cargo_cult():
    """Without it the same workload really does die, so the guard is doing work.

    If this ever passes unmitigated — a fixed faiss build, a different platform —
    the workaround in src/runtime.py can be removed. Skipped rather than failed
    in that case, because a fixed environment is good news, not a regression.
    """
    import os

    env = {k: v for k, v in os.environ.items() if k != "KMP_DUPLICATE_LIB_OK"}
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(_WORKLOAD)],
                       cwd=REPO, capture_output=True, text=True, env=env,
                       timeout=600)
    if r.returncode == 0 and "SURVIVED" in r.stdout:
        pytest.skip("OpenMP conflict no longer reproduces — mitigation may be "
                    "removable; see src/runtime.py")
    assert "OMP: Error" in r.stderr or r.returncode != 0


# --- no cosine threshold may meet a cross-encoder score ---------------

def test_grounded_answerer_disables_the_cosine_gate_entirely():
    from src.pipeline.full_stack import NO_RETRIEVAL_GATE

    assert NO_RETRIEVAL_GATE == float("-inf")


def test_answerer_never_passes_a_finite_min_score(tmp_path, monkeypatch):
    """The heart of the bug: a cosine floor must never gate a logit."""
    pytest.importorskip("faiss")
    from src.config import Config
    from src.pipeline.full_stack import GroundedAnswerer
    from src.pipeline.rag_pipeline import RagPipeline

    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "i"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cm.db"))
    monkeypatch.setenv("MIN_MESSAGES_PER_CHUNK", "2")
    f = tmp_path / "WhatsApp Chat with Rahul.txt"
    f.write_text("25/07/2026, 20:02 - Rahul: Joining date is 2 September.\n"
                 "25/07/2026, 20:03 - Me: Congrats!\n", encoding="utf-8")
    pipe = RagPipeline(Config.reload())
    pipe.ingest([f], me_names=["Me"])

    seen = {}
    inner = pipe.answer

    def spy(question, filters=None, *, k=None, min_score=None):
        seen["min_score"] = min_score
        return inner(question, filters, k=k, min_score=min_score)

    pipe.answer = spy
    GroundedAnswerer(pipe, rerank=False).answer("when does he join?")
    assert seen["min_score"] == float("-inf")
    pipe.close()


def test_low_scoring_evidence_is_still_answerable(tmp_path, monkeypatch):
    """P5's shape: the correct chunk is retrieved but scores far below every
    other question. With no score gate it must still reach the LLM."""
    pytest.importorskip("faiss")
    from src.config import Config
    from src.pipeline.rag_pipeline import RagPipeline

    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "i"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cm.db"))
    monkeypatch.setenv("MIN_MESSAGES_PER_CHUNK", "2")
    f = tmp_path / "WhatsApp Chat with Aditya.txt"
    f.write_text("01/08/2026, 21:05 - Aditya: trip plan: Goa in the last week of December\n"
                 "01/08/2026, 21:06 - Me: I'm in\n", encoding="utf-8")
    pipe = RagPipeline(Config.reload())
    pipe.ingest([f], me_names=["Me"])

    # a cross-encoder-like score: strongly negative, far below any cosine floor
    ans = pipe.answer("when are we going away on holiday?", min_score=float("-inf"))
    assert ans.abstained is False, "a negative score must not trigger abstention"
    assert ans.retrieved, "evidence must still reach the LLM"
    pipe.close()


# --- the Streamlit app ------------------------------------------------

APP = (REPO / "app.py").read_text(encoding="utf-8")


def test_app_does_not_expose_a_cosine_abstention_slider():
    """Regression: the app used to feed a 0.0-1.0 slider into min_score."""
    assert "min cosine score" not in APP
    assert "min_score=" not in APP, (
        "the app must not pass a retrieval score floor; abstention is the "
        "grounded prompt's decision"
    )


def test_app_initialises_native_runtimes_before_importing_heavy_libraries():
    init_at = APP.index("init_native_runtimes()")
    for later in ("import streamlit", "from src.pipeline", "from src.config"):
        assert APP.index(later) > init_at, (
            f"{later!r} is imported before the OpenMP guard runs"
        )


def test_app_uses_the_full_measured_retrieval_stack():
    assert "GroundedAnswerer" in APP, (
        "the demo must run the stack the experiments measured, not vector-only"
    )


def test_app_reports_missing_api_keys_and_provider_failures():
    assert "GEMINI_API_KEY not set" in APP
    assert "friendly_llm_error" in APP and "LLMError" in APP
    for signal in ("429", "503", "404"):
        assert signal in APP, f"no user-facing handling for HTTP {signal}"


def test_app_does_not_rebuild_the_index_on_every_run():
    """st.cache_resource keeps the pipeline and searchers across reruns; only
    ingestion invalidates them."""
    assert "@st.cache_resource" in APP
    assert "get_answerer.clear()" in APP


def test_app_handles_an_empty_corpus_before_asking():
    assert "count_chunks() == 0" in APP


# --- launching under the wrong interpreter ----------------------------
# Anaconda base has streamlit but not faiss, so `streamlit run app.py` without
# activating convmem starts fine and dies at ingestion. Both the launcher and
# the preflight exist to make that impossible to hit silently.

def test_preflight_detects_a_missing_native_runtime():
    from src.preflight import missing_runtimes

    assert missing_runtimes({"faiss": False, "torch": True}) == ["faiss"]
    assert missing_runtimes({"faiss": True, "torch": True}) == []
    assert missing_runtimes({}) == ["faiss"]


def test_preflight_ignores_runtimes_it_does_not_require():
    """torch missing is survivable (mock embedder); faiss is not."""
    from src.preflight import missing_runtimes

    assert missing_runtimes({"faiss": True, "torch": False}) == []


def test_preflight_message_names_the_interpreter_and_the_fix():
    import sys

    from src.preflight import LAUNCHER, wrong_environment_message

    err, fix = wrong_environment_message(["faiss"])
    assert sys.executable in err, "must say which python is running"
    assert "faiss" in err and "base" in err.lower()
    assert "no faiss wheel" in err, "must rule out 'just pip install it'"
    assert LAUNCHER in fix and "streamlit run app.py" in fix


def test_app_preflights_before_touching_the_pipeline():
    app = (REPO / "app.py").read_text(encoding="utf-8")
    stop_at = app.index("st.stop()")
    assert app.index("missing_runtimes") < stop_at
    assert stop_at < app.index("get_pipeline()"), (
        "the preflight must run before the pipeline is built"
    )


def test_launcher_pins_the_environment_interpreter():
    bat = (REPO / "run_app.bat").read_text(encoding="utf-8")
    assert "convmem" in bat
    assert "-m streamlit run app.py" in bat, "must launch via the pinned python"
    assert "cd /d" in bat, "project and Anaconda can sit on different drives"
    assert "CONVMEM_PYTHON" in bat, "must be overridable on another machine"
    assert "import faiss, torch, streamlit" in bat, "must verify before launching"
