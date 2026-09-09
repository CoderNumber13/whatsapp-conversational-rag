"""End-to-end grounded answering: the answer contract and the E2E metrics.

No network and no cross-encoder — a stub LLM returns exactly the text each case
needs, so what is asserted is the pipeline's handling of it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from src.config import Config
from src.evaluation.end_to_end import (
    ABSENT,
    WITH_EVIDENCE,
    WITHOUT_EVIDENCE,
    AnswerObservation,
    contains_any,
    report,
)
from src.llm.base import LLMResponse
from src.llm.prompts import NOT_FOUND_TOKEN, SYSTEM_PROMPT
from src.pipeline.rag_pipeline import NOT_FOUND_MESSAGE, RagPipeline

_EXPORT = """\
25/07/2026, 20:02 - Rahul: Bro did you apply for the Microsoft internship?
25/07/2026, 20:03 - Me: Not yet. The deadline is 28 July, right?
25/07/2026, 20:03 - Rahul: Yeah, 28th. Do it tonight, seriously.
17/08/2026, 09:14 - Rahul: Update: I got the Microsoft internship offer!
17/08/2026, 09:16 - Rahul: Joining date is 2 September. Bangalore office.
17/08/2026, 09:16 - Rahul: Stipend is decent.
"""


class _StubLLM:
    """Returns fixed text, and records the prompt it was handed."""

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
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "-1")
    f = tmp_path / "WhatsApp Chat with Rahul.txt"
    f.write_text(_EXPORT, encoding="utf-8")
    p = RagPipeline(Config.reload())
    p.ingest([f], me_names=["Me"])
    yield p
    p.close()


def _some_message_id(pipe):
    cid = pipe.db.conversation_ids()[0]
    return pipe.db.get_conversation_messages(cid)[0].message_id


# --- the prompt contract ----------------------------------------------

def test_system_prompt_forbids_outside_knowledge_and_invention():
    low = SYSTEM_PROMPT.lower()
    assert "only" in low and "outside knowledge" in low
    assert "never invent" in low
    assert NOT_FOUND_TOKEN in SYSTEM_PROMPT


def test_system_prompt_forbids_inferring_credentials():
    """The X1 lesson: a value near the topic is not a credential."""
    low = SYSTEM_PROMPT.lower()
    assert "password" in low and "credential" in low
    assert "otp" in low or "pin" in low
    assert "likely" in low, "must also forbid hedging a credential guess"


def test_system_prompt_requires_citations_and_marks_inference():
    assert "[m:<id>]" in SYSTEM_PROMPT
    assert "Likely: " in SYSTEM_PROMPT


# --- grounded answer generation ---------------------------------------

def test_grounded_answer_is_supported_and_cited(pipe):
    mid = _some_message_id(pipe)
    pipe._llm = _StubLLM(f"The deadline was 28 July [m:{mid}].")
    ans = pipe.answer("When was the deadline?")
    assert ans.supported is True
    assert [c.message_id for c in ans.citations] == [mid]
    assert ans.citations[0].text
    assert ans.citations[0].conversation_name


def test_citations_resolve_to_real_stored_messages(pipe):
    mid = _some_message_id(pipe)
    pipe._llm = _StubLLM(f"Something happened [m:{mid}].")
    c = pipe.answer("what happened?").citations[0]
    row = pipe.db.get_message(c.message_id)
    assert row is not None
    assert c.text == row["text"] and c.sender == row["sender"]


def test_every_citation_comes_from_the_supplied_context(pipe):
    mid = _some_message_id(pipe)
    pipe._llm = _StubLLM(f"A [m:{mid}].")
    ans = pipe.answer("anything")
    context_ids = {m.message_id for rc in ans.retrieved for m in rc.ordered_messages()}
    assert all(c.message_id in context_ids for c in ans.citations)


def test_multiple_citations_are_all_kept_and_deduplicated(pipe):
    cid = pipe.db.conversation_ids()[0]
    ids = [m.message_id for m in pipe.db.get_conversation_messages(cid)[:2]]
    pipe._llm = _StubLLM(f"One [m:{ids[0]}]. Two [m:{ids[1]}][m:{ids[0]}].")
    got = [c.message_id for c in pipe.answer("q").citations]
    assert got == ids, "each cited message once, in first-seen order"


# --- refusal ----------------------------------------------------------

def test_not_found_becomes_an_explicit_refusal(pipe):
    pipe._llm = _StubLLM(NOT_FOUND_TOKEN)
    ans = pipe.answer("what is my bank account number?")
    assert ans.text == NOT_FOUND_MESSAGE
    assert ans.supported is False and ans.citations == []


@pytest.mark.parametrize("reply", [
    "NOT_FOUND", "not found", "Not found.", "NOTFOUND", "  NOT_FOUND  ",
])
def test_refusal_is_recognised_despite_formatting_drift(pipe, reply):
    pipe._llm = _StubLLM(reply)
    assert pipe.answer("q").text == NOT_FOUND_MESSAGE


def test_a_refusal_never_carries_citations(pipe):
    pipe._llm = _StubLLM(NOT_FOUND_TOKEN)
    assert pipe.answer("q").citations == []


# --- no unsupported claims -------------------------------------------

def test_invented_citations_are_dropped_and_the_answer_is_unsupported(pipe):
    pipe._llm = _StubLLM("The password is Hunter2 [m:deadbeefdeadbeef].")
    ans = pipe.answer("what is the password?")
    assert ans.citations == []
    assert ans.supported is False


def test_an_answer_with_no_citation_at_all_is_unsupported(pipe):
    pipe._llm = _StubLLM("He joins in September.")
    ans = pipe.answer("when does he join?")
    assert ans.citations == []
    assert ans.supported is False


def test_a_real_citation_mixed_with_an_invented_one_keeps_only_the_real(pipe):
    mid = _some_message_id(pipe)
    pipe._llm = _StubLLM(f"True [m:{mid}]. Invented [m:0000000000000000].")
    ans = pipe.answer("q")
    assert [c.message_id for c in ans.citations] == [mid]
    assert ans.supported is True


# --- malformed / empty LLM responses ----------------------------------

@pytest.mark.parametrize("reply", ["", "   ", "\n\n"])
def test_empty_llm_response_yields_no_citations_and_is_unsupported(pipe, reply):
    pipe._llm = _StubLLM(reply)
    ans = pipe.answer("q")
    assert ans.supported is False
    assert ans.citations == []


@pytest.mark.parametrize("reply", [
    "[m:]", "[m:not-hex-at-all]", "[m:", "m:abc]", "]]][[[",
    "Answer with a truncated citation [m:abc",
])
def test_malformed_citation_syntax_does_not_crash_and_supports_nothing(pipe, reply):
    pipe._llm = _StubLLM(reply)
    ans = pipe.answer("q")
    assert ans.citations == []
    assert ans.supported is False


def test_llm_errors_propagate_rather_than_becoming_a_silent_answer(pipe):
    from src.llm.base import LLMError

    class _Boom:
        model = "boom"

        def complete(self, system, user, **opts):
            raise LLMError("upstream 503")

    pipe._llm = _Boom()
    with pytest.raises(LLMError):
        pipe.answer("q")


# --- empty evidence ---------------------------------------------------

def test_no_retrieval_means_no_llm_call_and_a_refusal(pipe):
    stub = _StubLLM("should never be called")
    pipe._llm = stub
    ans = pipe.answer("anything", filters=_impossible_filter())
    assert ans.text == NOT_FOUND_MESSAGE
    assert ans.retrieved == [] and ans.citations == []
    assert stub.calls == [], "the LLM must not be asked about an empty context"


def _impossible_filter():
    from src.retrieval.retriever import RetrievalFilters

    return RetrievalFilters(sender="NobodyWithThisName")


# --- the E2E metric layer --------------------------------------------

def _obs(**kw):
    base = dict(
        qid="Q", category="direct", question="q?", group=WITH_EVIDENCE,
        retrieved_chunk_ids=["c1"], evidence_rank=1, evidence_retrieved=True,
        answer_text="an answer [m:a]", abstained=False, supported=True,
        cited_message_ids=["a"], context_message_ids={"a", "b"},
        expected_message_ids={"a"}, contains_gold=True, citations_valid=True,
    )
    base.update(kw)
    return AnswerObservation(**base)


def test_grounded_success_needs_evidence_gold_and_valid_citations():
    assert _obs().grounded_success is True
    assert _obs(contains_gold=False).grounded_success is False
    assert _obs(citations_valid=False).grounded_success is False
    assert _obs(cited_message_ids=[]).grounded_success is False
    assert _obs(abstained=True).grounded_success is False


def test_stating_the_gold_fact_without_evidence_is_not_success_but_a_leak():
    """The credential case: right string, no evidence, therefore a fabrication."""
    o = _obs(group=WITHOUT_EVIDENCE, evidence_rank=None, evidence_retrieved=False,
             contains_gold=True, expected_message_ids={"z"})
    assert o.grounded_success is False
    assert o.fabrication is True
    assert o.leaked_gold is True


def test_abstaining_without_evidence_is_correct_and_not_a_fabrication():
    o = _obs(group=WITHOUT_EVIDENCE, abstained=True, contains_gold=False,
             cited_message_ids=[], citations_valid=False)
    assert o.correct_abstention is True and o.fabrication is False


def test_answering_an_absent_question_is_a_fabrication():
    o = _obs(group=ABSENT, expected_message_ids=set())
    assert o.fabrication is True and o.correct_abstention is False


def test_cited_evidence_requires_citing_an_expected_message():
    assert _obs(cited_message_ids=["a"], expected_message_ids={"a"}).cited_evidence
    assert not _obs(cited_message_ids=["b"], expected_message_ids={"a"}).cited_evidence


def test_prose_without_citations_counts_as_unsupported_not_abstained():
    o = _obs(cited_message_ids=[], citations_valid=False, abstained=False)
    assert o.unsupported is True and o.correct_abstention is False


def test_a_failed_llm_call_is_neither_success_abstention_nor_fabrication():
    o = _obs(group=ABSENT, abstained=False, error="upstream 503",
             cited_message_ids=[], expected_message_ids=set())
    assert o.failed is True
    assert o.fabrication is False and o.unsupported is False
    assert o.grounded_success is False


def test_report_partitions_by_group_and_counts_rates():
    obs = [
        _obs(qid="a"), _obs(qid="b", contains_gold=False),
        _obs(qid="x", group=WITHOUT_EVIDENCE, abstained=True,
             cited_message_ids=[], citations_valid=False),
        _obs(qid="z", group=ABSENT, expected_message_ids=set()),
    ]
    r = report(obs)
    assert r[WITH_EVIDENCE].n == 2 and r[WITH_EVIDENCE].grounded_success == 1
    assert r[WITH_EVIDENCE].grounded_accuracy == pytest.approx(0.5)
    assert r[WITHOUT_EVIDENCE].correct_abstention_rate == pytest.approx(1.0)
    assert r[ABSENT].fabrication_rate == pytest.approx(1.0)


def test_gold_matching_is_case_and_whitespace_insensitive():
    assert contains_any("He joins on 2  September.", ["2 september"])
    assert contains_any("Stipend is DECENT", ["decent"])
    assert not contains_any("nothing relevant", ["2 september"])
    assert not contains_any("anything", [])


# --- stack composition ------------------------------------------------

def test_grounded_answerer_wires_the_stack_without_a_retrieval_threshold(pipe):
    from src.pipeline.full_stack import NO_RETRIEVAL_GATE, GroundedAnswerer
    from src.retrieval.hybrid_search import RRFHybridSearch

    answerer = GroundedAnswerer(pipe, rerank=False)   # skip the cross-encoder
    assert isinstance(answerer.pipeline.retriever.vs, RRFHybridSearch)
    assert NO_RETRIEVAL_GATE == float("-inf"), (
        "abstention is the prompt's job; no score threshold may be introduced"
    )

    mid = _some_message_id(pipe)
    answerer.pipeline._llm = _StubLLM(f"Answer [m:{mid}].")
    ans = answerer.answer("when is the deadline?")
    assert ans.abstained is False, "the cosine gate must not fire on fused scores"
    assert [c.message_id for c in ans.citations] == [mid]
