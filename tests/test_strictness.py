"""Retrieval strictness: profiles, wiring, and the guarantees it must not break.

The control adjusts candidate DEPTH. It must never introduce a threshold, never
compare one stage's scores against another's, and never loosen grounding.
"""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from src.config import Config
from src.llm.base import LLMResponse
from src.llm.prompts import NOT_FOUND_TOKEN
from src.pipeline.rag_pipeline import NOT_FOUND_MESSAGE, RagPipeline
from src.pipeline.strictness import (
    BALANCED,
    DEFAULT_MODE,
    MODES,
    PERMISSIVE,
    STRICT,
    StrictnessProfile,
    all_profiles,
    profile_for,
)

# A corpus with the shape that matters: an unlabelled credential-like token and
# an address sitting near talk of an account, and nothing calling either a
# password. All values synthetic.
_EXPORT = """\
21/08/2026, 13:34 - Karan: synthetic sample message
21/08/2026, 13:35 - Me: ok let me set it up
21/08/2026, 13:40 - Me: synthetic sample message
21/08/2026, 14:29 - Me: @sample7handle
21/08/2026, 14:30 - Me: testuser.sample@example.com
21/08/2026, 16:47 - Karan: synthetic sample message
21/08/2026, 17:41 - Me: synthetic sample message
25/07/2026, 20:02 - Rahul: Joining date is 2 September. Bangalore office.
25/07/2026, 20:03 - Me: congrats!
"""

# Enough ordinary traffic that the corpus produces several chunks -- otherwise
# every mode retrieves the single chunk that exists and the depth settings are
# indistinguishable.
_FILLER = chr(10).join(
    f"2{2 + i // 40:d}/08/2026, {8 + (i // 6) % 12:02d}:{i % 60:02d} - "
    f"{'Karan' if i % 2 else 'Me'}: {topic} number {i}"
    for i, topic in enumerate(
        ["mess ka khana", "gym plan", "match score", "lab report", "cab share",
         "chai break", "room key", "assignment", "wifi speed", "dinner plan"] * 6
    )
)


class _StubLLM:
    model = "stub"

    def __init__(self, text):
        self._text = text
        self.calls = []

    def complete(self, system, user, **opts):
        self.calls.append((system, user))
        text = self._text(user) if callable(self._text) else self._text
        return LLMResponse(text=text, model=self.model)


@pytest.fixture()
def pipe(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cm.db"))
    monkeypatch.setenv("MIN_MESSAGES_PER_CHUNK", "2")
    f = tmp_path / "WhatsApp Chat with Karan.txt"
    f.write_text(_EXPORT + _FILLER + chr(10), encoding="utf-8")
    p = RagPipeline(Config.reload())
    p.ingest([f], me_names=["Me"])
    yield p
    p.close()


# --- profiles ---------------------------------------------------------

def test_default_mode_is_balanced():
    assert DEFAULT_MODE == BALANCED


def test_balanced_reproduces_the_configured_production_depths():
    """Balanced must BE the production default, not a copy that can drift."""
    cfg = Config.reload()
    p = profile_for(BALANCED, cfg)
    assert p.rrf_candidates == cfg.rrf_candidates
    assert p.rerank_candidates == cfg.rerank_candidates
    assert p.evidence_chunks == cfg.max_context_chunks


def test_balanced_tracks_config_changes(monkeypatch):
    monkeypatch.setenv("RRF_CANDIDATES", "37")
    monkeypatch.setenv("RERANK_CANDIDATES", "11")
    monkeypatch.setenv("MAX_CONTEXT_CHUNKS", "6")
    cfg = Config.reload()
    p = profile_for(BALANCED, cfg)
    assert (p.rrf_candidates, p.rerank_candidates, p.evidence_chunks) == (37, 11, 6)
    Config.reload()


def test_strict_admits_less_evidence_and_permissive_more():
    cfg = Config.reload()
    s, b, p = (profile_for(m, cfg) for m in (STRICT, BALANCED, PERMISSIVE))
    assert s.evidence_chunks < b.evidence_chunks < p.evidence_chunks


def test_permissive_fuses_a_shallower_pool_on_purpose():
    """Measured, not intuitive: a deeper RRF pool supplies more two-retriever
    chunks, and every one outranks a chunk only one retriever can see — which is
    exactly what the credential chunk is. See docs/BASELINE.md experiment 7."""
    cfg = Config.reload()
    assert profile_for(PERMISSIVE, cfg).rrf_candidates < \
        profile_for(BALANCED, cfg).rrf_candidates


def test_every_mode_resolves_and_is_named():
    profiles = all_profiles()
    assert set(profiles) == set(MODES)
    for mode, p in profiles.items():
        assert p.name == mode and p.description


@pytest.mark.parametrize("mode", ["Strict", "BALANCED", "  permissive  "])
def test_mode_lookup_is_forgiving_about_case_and_spacing(mode):
    assert profile_for(mode).name == mode.strip().lower()


def test_unknown_mode_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown strictness"):
        profile_for("aggressive")


def test_profiles_are_counts_of_chunks_never_scores():
    """A depth is a positive integer. A threshold would be a float."""
    for p in all_profiles().values():
        for value in (p.rrf_candidates, p.rerank_candidates, p.evidence_chunks):
            assert isinstance(value, int) and value >= 1


def test_a_profile_rejects_nonsensical_depths():
    with pytest.raises(ValueError, match="must be >= 1"):
        StrictnessProfile(name="x", rrf_candidates=0, rerank_candidates=5,
                          evidence_chunks=5)
    with pytest.raises(ValueError, match="must be >= 1"):
        StrictnessProfile(name="x", rrf_candidates=5, rerank_candidates=5,
                          evidence_chunks=0)


# --- wiring into the stack -------------------------------------------

def test_default_answerer_uses_balanced_and_preserves_current_behaviour(pipe):
    from src.pipeline.full_stack import GroundedAnswerer

    a = GroundedAnswerer(pipe, rerank=False)
    assert a.profile.name == BALANCED
    assert a.top_n == pipe.config.max_context_chunks
    assert a.pipeline.retriever.vs.candidates == pipe.config.rrf_candidates


def test_modes_actually_change_the_configured_depths(pipe):
    from src.pipeline.full_stack import GroundedAnswerer

    seen = {}
    for mode in MODES:
        a = GroundedAnswerer(pipe, rerank=False, strictness=mode)
        seen[mode] = (a.pipeline.retriever.vs.candidates, a.top_n)
    assert seen[STRICT][1] < seen[BALANCED][1] < seen[PERMISSIVE][1]
    assert seen[PERMISSIVE][0] < seen[BALANCED][0]


def test_permissive_hands_more_evidence_to_the_model(pipe):
    from src.pipeline.full_stack import GroundedAnswerer

    counts = {}
    for mode in (STRICT, BALANCED, PERMISSIVE):
        a = GroundedAnswerer(pipe, rerank=False, strictness=mode)
        stub = _StubLLM(NOT_FOUND_TOKEN)
        a.pipeline._llm = stub
        counts[mode] = len(a.answer("what did we set up?").retrieved)
    assert counts[STRICT] <= counts[BALANCED] <= counts[PERMISSIVE]
    assert counts[PERMISSIVE] > counts[STRICT], "permissive must widen the context"


def test_an_explicit_top_n_still_overrides_the_profile(pipe):
    from src.pipeline.full_stack import GroundedAnswerer

    a = GroundedAnswerer(pipe, rerank=False, strictness=PERMISSIVE, top_n=2)
    assert a.top_n == 2


def test_strictness_reaches_the_rerank_stage_too(pipe):
    """The rerank depth must follow the profile, not stay at the default."""
    from src.pipeline.full_stack import build_searcher

    prof = profile_for(PERMISSIVE, pipe.config)
    searcher = build_searcher(pipe.db, pipe.config, rerank=False, profile=prof)
    assert searcher.candidates == prof.rrf_candidates

    balanced = build_searcher(pipe.db, pipe.config, rerank=False,
                              profile=profile_for(BALANCED, pipe.config))
    assert balanced.candidates == pipe.config.rrf_candidates


# --- no threshold may leak in ----------------------------------------

def test_no_mode_introduces_a_retrieval_score_threshold(pipe):
    """Every mode must still disable the cosine gate entirely."""
    from src.pipeline.full_stack import NO_RETRIEVAL_GATE, GroundedAnswerer

    for mode in MODES:
        a = GroundedAnswerer(pipe, rerank=False, strictness=mode)
        seen = {}
        inner = a.pipeline.answer

        def spy(question, filters=None, *, k=None, min_score=None, _i=inner):
            seen["min_score"] = min_score
            return _i(question, filters, k=k, min_score=min_score)

        a.pipeline.answer = spy
        a.pipeline._llm = _StubLLM(NOT_FOUND_TOKEN)
        a.answer("anything")
        assert seen["min_score"] == NO_RETRIEVAL_GATE == float("-inf"), (
            f"{mode} allowed a finite score floor to reach the pipeline"
        )
        a.pipeline.answer = inner


def test_min_retrieval_score_is_never_compared_to_a_fused_or_reranked_score():
    """The cosine floor stays a cosine floor. If a mode ever passed it into the
    stack, RRF values (~0.03) and cross-encoder logits (mostly negative) would
    fall below 0.25 and abstain on everything."""
    cfg = Config.reload()
    assert cfg.min_retrieval_score == 0.25, "the cosine floor itself is unchanged"
    source = (
        __import__("pathlib").Path("src/pipeline/strictness.py")
        .read_text(encoding="utf-8")
    )
    assert "min_retrieval_score" not in source
    assert "threshold" not in source.split('"""', 2)[2].lower(), (
        "strictness code must not reference thresholds outside its docstring"
    )


# --- grounding is not weakened ---------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_permissive_does_not_bypass_grounding(pipe, mode):
    """More evidence in the context must not turn a refusal into an answer."""
    from src.pipeline.full_stack import GroundedAnswerer

    a = GroundedAnswerer(pipe, rerank=False, strictness=mode)
    a.pipeline._llm = _StubLLM(NOT_FOUND_TOKEN)
    ans = a.answer("What is my gmail password?")
    assert ans.text == NOT_FOUND_MESSAGE
    assert ans.supported is False
    assert ans.citations == []


@pytest.mark.parametrize("mode", MODES)
def test_absent_questions_still_abstain_in_every_mode(pipe, mode):
    from src.pipeline.full_stack import GroundedAnswerer

    a = GroundedAnswerer(pipe, rerank=False, strictness=mode)
    a.pipeline._llm = _StubLLM(NOT_FOUND_TOKEN)
    ans = a.answer("What is my bank account number?")
    assert ans.text == NOT_FOUND_MESSAGE and ans.supported is False


@pytest.mark.parametrize("mode", MODES)
def test_a_fabricated_credential_is_dropped_in_every_mode(pipe, mode):
    """Citation validation must still discard an invented source, so a
    fabricated password cannot be presented as supported."""
    from src.pipeline.full_stack import GroundedAnswerer

    a = GroundedAnswerer(pipe, rerank=False, strictness=mode)
    a.pipeline._llm = _StubLLM("The password is @sample7handle [m:deadbeefdeadbeef].")
    ans = a.answer("What is my gmail password?")
    assert ans.citations == []
    assert ans.supported is False, (
        "an answer whose only citation was invented must not count as supported"
    )


@pytest.mark.parametrize("mode", MODES)
def test_the_credential_rule_is_in_the_prompt_for_every_mode(pipe, mode):
    """Strictness must not swap in a laxer system prompt."""
    from src.llm.prompts import SYSTEM_PROMPT
    from src.pipeline.full_stack import GroundedAnswerer

    a = GroundedAnswerer(pipe, rerank=False, strictness=mode)
    stub = _StubLLM(NOT_FOUND_TOKEN)
    a.pipeline._llm = stub
    a.answer("What is my gmail password?")
    system, _user = stub.calls[0]
    assert system == SYSTEM_PROMPT
    low = system.lower()
    assert "credential" in low and "password" in low
    assert "likely" in low, "hedged credential guesses must stay forbidden"


def test_permissive_context_may_contain_the_token_without_it_being_answerable(pipe):
    """The honest shape of the credential case: permissive can put the token in
    front of the model, and the model must still refuse, because nothing in the
    conversation says it is a password."""
    from src.pipeline.full_stack import GroundedAnswerer

    a = GroundedAnswerer(pipe, rerank=False, strictness=PERMISSIVE)
    stub = _StubLLM(NOT_FOUND_TOKEN)
    a.pipeline._llm = stub
    ans = a.answer("What is my gmail password?")
    _system, user = stub.calls[0]
    assert "@sample7handle" in user, "permissive should surface the nearby token"
    assert ans.supported is False and ans.citations == []


# --- the Streamlit control -------------------------------------------

APP = (__import__("pathlib").Path("app.py")).read_text(encoding="utf-8")


def test_app_exposes_the_strictness_control():
    assert "Retrieval strictness" in APP
    assert "st.radio" in APP


def test_app_control_is_wired_to_the_production_stack():
    assert "get_answerer(pipe, rerank, strictness)" in APP
    assert "strictness=strictness" in APP
    assert "GroundedAnswerer" in APP


def test_app_defaults_to_balanced():
    assert "index=MODES.index(DEFAULT_MODE)" in APP


def test_app_still_exposes_no_score_threshold():
    assert "min cosine score" not in APP
    assert "min_score=" not in APP


def test_app_caches_the_stack_per_mode():
    """A mode change must rebuild the searcher, not silently reuse the old one."""
    assert "def get_answerer(_pipe: RagPipeline, rerank: bool, strictness: str)" in APP
