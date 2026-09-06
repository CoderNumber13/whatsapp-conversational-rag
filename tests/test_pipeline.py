"""End-to-end pipeline tests (mock embedder + mock/stub LLM — no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from src.config import Config
from src.llm.base import LLMResponse
from src.pipeline.rag_pipeline import RagPipeline
from src.retrieval.retriever import RetrievalFilters
from scripts.generate_synthetic_chats import write_all


def _build(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cm.db"))
    monkeypatch.setenv("CHUNK_SIZE_MESSAGES", "6")
    monkeypatch.setenv("CHUNK_OVERLAP_MESSAGES", "2")
    monkeypatch.setenv("MIN_MESSAGES_PER_CHUNK", "2")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    cfg = Config.reload()
    files = write_all(tmp_path / "chats")
    pipe = RagPipeline(cfg)
    report = pipe.ingest(files, me_names=["Me"])
    return pipe, cfg, report, files


class _StubLLM:
    model = "stub"

    def __init__(self, text):
        self._text = text

    def complete(self, system, user, **opts):
        return LLMResponse(text=self._text, model=self.model)


def _pipeline_over(tmp_path, monkeypatch, export_text, *, me="You", **env):
    """A pipeline holding one ad-hoc WhatsApp export (mock embedder + LLM)."""
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cm.db"))
    monkeypatch.setenv("MIN_MESSAGES_PER_CHUNK", "2")
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "-1")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    f = tmp_path / "WhatsApp Chat with Karan.txt"
    f.write_text(export_text, encoding="utf-8")
    pipe = RagPipeline(Config.reload())
    pipe.ingest([f], me_names=[me])
    return pipe


# --- regression: "gmail password" query when the data has no password -----
# Original report: asking for the password of a Gmail account returned
# "no record found". Trace showed every stage worked: retrieval surfaced the
# gmail conversation and it was in the LLM prompt, but the chat only contains
# "make a new gmail" + the address — no password exists anywhere in the data.
# The system correctly refused to invent one. This test locks that in.

_GMAIL_EXPORT = """\
21/08/2026, 13:34 - Karan: synthetic sample message
21/08/2026, 13:34 - You: synthetic sample message
21/08/2026, 13:40 - You: synthetic sample message
21/08/2026, 14:29 - You: @sample7handle
21/08/2026, 14:30 - You: testuser.sample@example.com
21/08/2026, 16:47 - Karan: synthetic sample message
21/08/2026, 17:41 - You: synthetic sample message
"""


def test_gmail_password_query_does_not_fabricate_when_absent(tmp_path, monkeypatch):
    pipe = _pipeline_over(tmp_path, monkeypatch, _GMAIL_EXPORT)

    # the corpus genuinely contains no password (root cause of the report)
    assert pipe.db.get_messages(contains="password") == []
    assert pipe.db.get_messages(contains="pwd") == []

    # retrieval is NOT the failing stage: the gmail conversation is retrievable
    hits = pipe.retriever.retrieve(
        "password of the new gmail account I created for Karan", k=5
    )
    assert hits, "retrieval returned nothing at all"
    assert any("gmail" in h.chunk.text.lower() for h in hits), "gmail chunk not retrieved"

    # a grounded LLM sees the gmail messages but no password -> NOT_FOUND.
    pipe._llm = _StubLLM("NOT_FOUND")
    ans = pipe.answer(
        "What is the password of the new Gmail account I created for Karan?"
    )
    assert ans.abstained is False        # threshold was cleared; the LLM was consulted
    assert ans.supported is False        # nothing to support an answer
    assert ans.citations == []           # and nothing fabricated
    assert "password" not in ans.text.lower()


def test_gmail_password_query_drops_hallucinated_password(tmp_path, monkeypatch):
    """If a future model *invents* a password with a made-up citation, the
    citation validator must drop it and the answer must not count as supported."""
    pipe = _pipeline_over(tmp_path, monkeypatch, _GMAIL_EXPORT)
    pipe._llm = _StubLLM("The password is Hunter2! [m:deadbeefdeadbeef].")
    ans = pipe.answer("What is the Gmail password?")
    assert ans.citations == []
    assert ans.supported is False


# --- ingest --------------------------------------------------------

def test_ingest_report(tmp_path, monkeypatch):
    pipe, cfg, report, _ = _build(tmp_path, monkeypatch)
    assert report.conversations == 4
    assert report.messages_added > 40
    assert report.chunks.added > 0
    assert report.index.embedded == pipe.db.count_chunks()
    pipe.close()


def test_context_is_capped_and_budgeted(tmp_path, monkeypatch):
    # tiny ctx budget -> pipeline must trim the chunks it sends to the LLM
    pipe, cfg, _, _ = _build(
        tmp_path, monkeypatch, MIN_RETRIEVAL_SCORE="-1",
        MAX_CONTEXT_CHUNKS="6", OLLAMA_NUM_CTX="700", LLM_MAX_TOKENS="100",
        LLM_PROVIDER="ollama",
    )
    seen = {}

    class _Spy:
        model = "spy"

        def complete(self, system, user, **opts):
            seen["n_excerpts"] = user.count("### Excerpt ")
            from src.llm.base import LLMResponse

            return LLMResponse(text="ok [m:%s]" % pipe.db.all_chunks()[0].message_ids[0], model="spy")

    pipe._llm = _Spy()
    retrieved = pipe.retriever.retrieve("placements internship project", k=8)
    assert len(retrieved) >= 3  # there is more than we can send
    pipe.answer("placements internship project")
    assert 1 <= seen["n_excerpts"] < len(retrieved)  # trimmed
    pipe.close()


def test_reingest_is_idempotent(tmp_path, monkeypatch):
    pipe, cfg, report, files = _build(tmp_path, monkeypatch)
    again = pipe.ingest(files, me_names=["Me"])
    assert again.messages_added == 0
    assert again.chunks.changed_total == 0
    assert again.index.changed_total == 0
    pipe.close()


# --- answer: happy path ----------------------------------------

def test_answer_on_exact_chunk_text_is_grounded(tmp_path, monkeypatch):
    pipe, cfg, _, _ = _build(tmp_path, monkeypatch, MIN_RETRIEVAL_SCORE="-1")
    target = pipe.db.all_chunks()[0]
    ans = pipe.answer(target.text)  # mock embedder -> exact match scores ~1.0
    assert ans.abstained is False
    assert ans.supported is True
    assert ans.citations
    ctx_ids = {m.message_id for rc in ans.retrieved for m in rc.ordered_messages()}
    assert all(c.message_id in ctx_ids for c in ans.citations)
    # citation resolves to a real stored message
    assert all(pipe.db.get_message(c.message_id) is not None for c in ans.citations)
    pipe.close()


# --- answer: abstention (no LLM call) --------------------------

def test_answer_abstains_below_threshold(tmp_path, monkeypatch):
    pipe, cfg, _, _ = _build(tmp_path, monkeypatch, MIN_RETRIEVAL_SCORE="0.99")
    ans = pipe.answer("something totally unrelated to any chat content xyzzy")
    assert ans.abstained is True
    assert ans.supported is False
    assert ans.llm_model == "(none)"
    assert ans.citations == []
    pipe.close()


def test_answer_empty_when_filters_exclude_everything(tmp_path, monkeypatch):
    pipe, cfg, _, _ = _build(tmp_path, monkeypatch, MIN_RETRIEVAL_SCORE="-1")
    ans = pipe.answer("anything", filters=RetrievalFilters(sender="ghost"))
    assert ans.abstained is True and ans.retrieved == []
    pipe.close()


# --- answer: model says NOT_FOUND / hallucinates a citation ----

def test_not_found_response_marks_unsupported(tmp_path, monkeypatch):
    pipe, cfg, _, _ = _build(tmp_path, monkeypatch, MIN_RETRIEVAL_SCORE="-1")
    pipe._llm = _StubLLM("NOT_FOUND")
    ans = pipe.answer("whatever")
    assert ans.supported is False and ans.abstained is False
    assert "couldn't find" in ans.text.lower()
    pipe.close()


def test_hallucinated_citation_is_dropped(tmp_path, monkeypatch):
    pipe, cfg, _, _ = _build(tmp_path, monkeypatch, MIN_RETRIEVAL_SCORE="-1")
    pipe._llm = _StubLLM("The answer is 42 [m:deadbeefdeadbeef].")
    ans = pipe.answer("whatever")
    assert ans.citations == []
    assert ans.supported is False  # no valid citation left
    pipe.close()


# --- persistence: SQLite + index survive a reopen -------------

def test_reopen_pipeline_answers_same(tmp_path, monkeypatch):
    pipe, cfg, _, _ = _build(tmp_path, monkeypatch, MIN_RETRIEVAL_SCORE="-1")
    target = pipe.db.all_chunks()[0].text
    first = pipe.answer(target)
    pipe.close()

    pipe2 = RagPipeline(Config.reload())
    second = pipe2.answer(target)
    assert {c.message_id for c in second.citations} == {c.message_id for c in first.citations}
    pipe2.close()
