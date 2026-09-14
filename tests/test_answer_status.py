"""Telling "the model refused" apart from "the model forgot to cite".

Reported against the live system: asking

    "Has Karan Pratap ever said anything in the group?"

showed the refusal UI, so it read as "no data found". Investigation showed the
opposite: retrieval put his messages in all five context chunks and the model
answered correctly -- it simply emitted no ``[m:<id>]`` tag. Zero citations was
being collapsed into "unsupported", and the UI rendered that as a refusal.

``llama3.1:8b`` follows the citation format inconsistently, dropping it on short
summary answers while keeping it on specific factual ones, so this is a routine
outcome locally rather than an edge case.

Nothing here relaxes grounding: an uncited answer is still not ``supported``.
It is merely reported as what it is.
"""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from src.config import Config
from src.llm.base import LLMResponse
from src.llm.prompts import NOT_FOUND_TOKEN
from src.pipeline.rag_pipeline import (
    NOT_FOUND_MESSAGE,
    AnswerStatus,
    RagPipeline,
)
from src.retrieval.retriever import RetrievalFilters

ABHISHEK_QUESTION = "Has Karan Pratap ever said anything in the group?"

# Mirrors the shape of the real group chat: the person speaks several times
# among other participants. All values synthetic.
_EXPORT = """\
14/12/2025, 21:02 - Karan Pratap Singh: kal lab hai kya
14/12/2025, 21:03 - Gautam: haan 9 baje
14/12/2025, 21:05 - Karan Pratap Singh: theek hai aa jaunga
15/12/2025, 08:10 - Vaibhav Jain: mess me khana ready hai
15/12/2025, 08:12 - Karan Pratap Singh: aa raha hu 5 min
16/12/2025, 19:40 - Gautam: match dekhne chalein
16/12/2025, 19:41 - Me: haan chalo
"""


class _StubLLM:
    """Returns exactly the text a case needs, and records the prompt."""

    model = "stub"

    def __init__(self, text):
        self._text = text
        self.calls = []

    def complete(self, system, user, **opts):
        self.calls.append((system, user))
        return LLMResponse(
            text=self._text(user) if callable(self._text) else self._text,
            model=self.model,
        )


@pytest.fixture()
def pipe(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cm.db"))
    monkeypatch.setenv("MIN_MESSAGES_PER_CHUNK", "2")
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "-1")
    f = tmp_path / "WhatsApp Chat with Boys Hostel.txt"
    f.write_text(_EXPORT, encoding="utf-8")
    p = RagPipeline(Config.reload())
    p.ingest([f], me_names=["Me"])
    yield p
    p.close()


def _a_real_message_id(pipe) -> str:
    cid = pipe.db.conversation_ids()[0]
    return pipe.db.get_conversation_messages(cid)[0].message_id


# --- the reported regression -----------------------------------------

def test_uncited_answer_is_not_reported_as_a_refusal(pipe):
    """The exact reported failure: a correct answer with no citation tag."""
    pipe._llm = _StubLLM("Karan Pratap Singh has said something in the group.")
    ans = pipe.answer(ABHISHEK_QUESTION)

    assert ans.status is AnswerStatus.UNCITED
    assert ans.is_refusal is False, "an answered question is not a refusal"
    assert ans.text != NOT_FOUND_MESSAGE, "the model's answer must survive"
    assert "Karan" in ans.text
    assert ans.needs_caution is True


def test_an_uncited_answer_is_still_not_grounded(pipe):
    """Grounding policy is unchanged: no citation, not supported."""
    pipe._llm = _StubLLM("Karan Pratap Singh has said something in the group.")
    ans = pipe.answer(ABHISHEK_QUESTION)
    assert ans.supported is False
    assert ans.citations == []


def test_an_uncited_answer_still_returns_its_evidence_for_inspection(pipe):
    """The user has to verify it themselves, so the chunks must be available."""
    pipe._llm = _StubLLM("Karan Pratap Singh has said something in the group.")
    ans = pipe.answer(ABHISHEK_QUESTION)
    assert ans.retrieved, "retrieved evidence must be returned, not discarded"


def test_no_citation_is_invented_to_fill_the_gap(pipe):
    """The fix must not paper over the omission by attributing the answer to
    whatever happened to be retrieved."""
    pipe._llm = _StubLLM("Karan Pratap Singh has said something in the group.")
    ans = pipe.answer(ABHISHEK_QUESTION)
    assert ans.citations == []
    assert "[m:" not in ans.text


# --- genuine abstention ----------------------------------------------

def test_model_refusal_is_still_an_abstention(pipe):
    pipe._llm = _StubLLM(NOT_FOUND_TOKEN)
    ans = pipe.answer("What is my bank account number?")
    assert ans.status is AnswerStatus.REFUSED
    assert ans.is_refusal is True and ans.needs_caution is False
    assert ans.text == NOT_FOUND_MESSAGE
    assert ans.supported is False and ans.citations == []


def test_retrieval_gate_refusal_is_still_an_abstention(pipe):
    """Nothing retrieved: the LLM is never consulted."""
    stub = _StubLLM("should never be called")
    pipe._llm = stub
    ans = pipe.answer("anything", filters=RetrievalFilters(sender="NobodyAtAll"))
    assert ans.status is AnswerStatus.REFUSED
    assert ans.abstained is True
    assert stub.calls == []


def test_refusal_and_uncited_are_distinguishable(pipe):
    """The whole point: these two must never look alike again."""
    pipe._llm = _StubLLM(NOT_FOUND_TOKEN)
    refused = pipe.answer(ABHISHEK_QUESTION)
    pipe._llm = _StubLLM("Karan Pratap Singh has said something in the group.")
    uncited = pipe.answer(ABHISHEK_QUESTION)

    assert refused.status is not uncited.status
    assert refused.is_refusal and not uncited.is_refusal
    assert refused.supported is uncited.supported is False, (
        "both remain ungrounded; only how they are reported differs"
    )


# --- correctly cited answer ------------------------------------------

def test_cited_answer_is_grounded(pipe):
    mid = _a_real_message_id(pipe)
    pipe._llm = _StubLLM(f"Yes, he has posted in the group [m:{mid}].")
    ans = pipe.answer(ABHISHEK_QUESTION)

    assert ans.status is AnswerStatus.GROUNDED
    assert ans.supported is True
    assert ans.needs_caution is False and ans.is_refusal is False
    assert [c.message_id for c in ans.citations] == [mid]


def test_a_partly_valid_citation_set_still_counts_as_grounded(pipe):
    mid = _a_real_message_id(pipe)
    pipe._llm = _StubLLM(f"He posted [m:{mid}] and also [m:0000000000000000].")
    ans = pipe.answer(ABHISHEK_QUESTION)
    assert ans.status is AnswerStatus.GROUNDED
    assert [c.message_id for c in ans.citations] == [mid], "the fake one is dropped"


# --- invalid / nonexistent citations ---------------------------------

def test_wholly_invented_citations_are_not_treated_as_merely_uncited(pipe):
    """A model that cites sources which do not exist is a different and worse
    failure than one that cites nothing at all."""
    pipe._llm = _StubLLM("He said plenty [m:deadbeefdeadbeef].")
    ans = pipe.answer(ABHISHEK_QUESTION)

    assert ans.status is AnswerStatus.INVALID_CITATIONS
    assert ans.status is not AnswerStatus.UNCITED
    assert ans.supported is False
    assert ans.citations == []
    assert ans.needs_caution is True


def test_a_citation_outside_the_retrieved_context_is_rejected(pipe):
    """A real message id that was not in this answer's context is still not
    evidence for this answer."""
    cid = pipe.db.conversation_ids()[0]
    msgs = pipe.db.get_conversation_messages(cid)
    pipe._llm = _StubLLM(f"Claim [m:{msgs[0].message_id}].")
    ans = pipe.answer(ABHISHEK_QUESTION)
    context_ids = {m.message_id for rc in ans.retrieved for m in rc.ordered_messages()}
    for c in ans.citations:
        assert c.message_id in context_ids


@pytest.mark.parametrize("reply", [
    "He answered [m:]", "He answered [m:zzzz]", "He answered [m:",
    "He answered m:abc]", "He answered ]]][[[",
])
def test_malformed_citation_syntax_is_uncited_not_crashing(pipe, reply):
    pipe._llm = _StubLLM(reply)
    ans = pipe.answer(ABHISHEK_QUESTION)
    assert ans.supported is False
    assert ans.citations == []
    assert ans.status in (AnswerStatus.UNCITED, AnswerStatus.INVALID_CITATIONS)


# --- the status field itself ------------------------------------------

def test_every_answer_carries_exactly_one_status(pipe):
    mid = _a_real_message_id(pipe)
    cases = {
        NOT_FOUND_TOKEN: AnswerStatus.REFUSED,
        "plain answer, no tags": AnswerStatus.UNCITED,
        "bogus [m:ffffffffffffffff]": AnswerStatus.INVALID_CITATIONS,
        f"good [m:{mid}]": AnswerStatus.GROUNDED,
    }
    for reply, expected in cases.items():
        pipe._llm = _StubLLM(reply)
        assert pipe.answer(ABHISHEK_QUESTION).status is expected, reply


def test_supported_still_means_grounded_and_nothing_else(pipe):
    """`supported` semantics are unchanged, so existing callers keep working."""
    mid = _a_real_message_id(pipe)
    for reply in (NOT_FOUND_TOKEN, "no tags here", "bogus [m:aaaaaaaaaaaaaaaa]"):
        pipe._llm = _StubLLM(reply)
        assert pipe.answer(ABHISHEK_QUESTION).supported is False
    pipe._llm = _StubLLM(f"cited [m:{mid}]")
    ans = pipe.answer(ABHISHEK_QUESTION)
    assert ans.supported is (ans.status is AnswerStatus.GROUNDED) is True


# --- the UI renders the distinction -----------------------------------

APP = (__import__("pathlib").Path("app.py")).read_text(encoding="utf-8")


def test_app_branches_on_status_not_on_supported():
    assert "ans.status is AnswerStatus.REFUSED" in APP
    assert "ans.status is AnswerStatus.UNCITED" in APP
    assert "ans.status is AnswerStatus.INVALID_CITATIONS" in APP
    assert "if not ans.supported:" not in APP, (
        "branching on `supported` is what collapsed refusal and uncited"
    )


def test_app_warns_on_an_uncited_answer_without_hiding_it():
    assert "did not provide source citations" in APP
    assert "Treat this answer with caution" in APP


def test_app_flags_invented_citations_more_strongly():
    assert "do not exist in the retrieved" in APP
    assert "st.error(" in APP


def test_app_opens_the_evidence_panel_when_verification_is_needed():
    assert "expanded=ans.needs_caution" in APP


# --- refusals the model dresses up in a sentence -----------------------
# Surfaced while fixing the above: llama3.1:8b answered "The final answer is
# NOT_FOUND." Detection required the bare token, so a genuine refusal was
# classified UNCITED and the raw sentinel was shown to the user as content.

@pytest.mark.parametrize("reply", [
    "NOT_FOUND",
    "not found",
    "Not found.",
    "NOTFOUND",
    "  NOT_FOUND  ",
    "The final answer is NOT_FOUND.",
    "I could not answer: NOT_FOUND",
    "Answer: NOTFOUND",
])
def test_a_refusal_is_detected_however_the_model_phrases_it(pipe, reply):
    from src.llm.prompts import is_not_found

    assert is_not_found(reply) is True
    pipe._llm = _StubLLM(reply)
    ans = pipe.answer(ABHISHEK_QUESTION)
    assert ans.status is AnswerStatus.REFUSED
    assert ans.text == NOT_FOUND_MESSAGE, "the sentinel must never reach the user"


@pytest.mark.parametrize("reply", [
    "They discussed how the missing parcel was not found anywhere near the gate.",
    "Rahul said the file was not found on the server [m:abc123].",
    "Karan Pratap Singh has said something in the group.",
])
def test_prose_containing_not_found_is_not_a_refusal(pipe, reply):
    """'not found' is ordinary English. Only the underscore sentinel counts,
    so a real answer about something not being found is still an answer."""
    from src.llm.prompts import is_not_found

    assert is_not_found(reply) is False


def test_a_cited_answer_is_never_reclassified_as_a_refusal(pipe):
    """Even if the model mentions the sentinel, a citation means it answered."""
    from src.llm.prompts import is_not_found

    mid = _a_real_message_id(pipe)
    reply = f"The answer is NOT_FOUND but he did post here [m:{mid}]."
    assert is_not_found(reply) is False
    pipe._llm = _StubLLM(reply)
    assert pipe.answer(ABHISHEK_QUESTION).status is AnswerStatus.GROUNDED


def test_a_long_answer_mentioning_the_sentinel_is_not_a_refusal(pipe):
    """The length guard stops an essay that happens to discuss NOT_FOUND from
    being swallowed as a refusal."""
    from src.llm.prompts import is_not_found

    long_reply = ("The team debated what NOT_FOUND should mean in the API " * 4)
    assert len(long_reply) > 120
    assert is_not_found(long_reply) is False
