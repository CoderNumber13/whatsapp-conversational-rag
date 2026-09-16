"""End-to-end pipeline tests (mock embedder + mock/stub LLM — no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from src.chunking.base import render_embedding_text
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


# --- regression: the "gmail password" not-found report --------------------
# Reported twice. The first investigation searched the corpus for the keywords
# password/pwd/login/otp, found none, and concluded no password existed — so it
# recorded the refusal as correct behaviour. That conclusion was WRONG: the
# credential was there all along as a bare, unlabelled token sent one minute
# before the address. A keyword scan cannot see it, because nothing in the chat
# says the word "password".
#
# The real defect is that an unlabelled high-entropy token carries almost no
# semantic signal, so cosine similarity against the word "password" cannot reach
# it. Vector-only retrieval structurally cannot answer this; it needs the exact/
# keyword half of hybrid retrieval (Phase 2). See the xfail below.
#
# All values here are SYNTHETIC. Never put a real credential in a fixture — the
# earlier version of this file committed one to git history.

_NO_CREDENTIAL_EXPORT = """\
21/08/2026, 13:34 - Karan: can you set up a new gmail for the shared plan
21/08/2026, 13:34 - You: sure, keeping the spend under ten dollars
21/08/2026, 13:40 - You: let me check the options
21/08/2026, 14:30 - You: testuser.sample@example.com
21/08/2026, 16:47 - Karan: go ahead and buy it, I will send the money
21/08/2026, 17:41 - You: done, please check and let me know
"""

# Same conversation, but with an unlabelled credential token at 14:29 — the
# shape of the real export. "@sample7handle" stands in for the real value.
_UNLABELLED_CREDENTIAL_EXPORT = """\
21/08/2026, 13:34 - Karan: can you set up a new gmail for the shared plan
21/08/2026, 13:34 - You: sure, keeping the spend under ten dollars
21/08/2026, 13:40 - You: let me check the options
21/08/2026, 14:29 - You: @sample7handle
21/08/2026, 14:30 - You: testuser.sample@example.com
21/08/2026, 16:47 - Karan: go ahead and buy it, I will send the money
21/08/2026, 17:41 - You: done, please check and let me know
"""


def test_gmail_password_query_does_not_fabricate_when_absent(tmp_path, monkeypatch):
    """With no credential in the corpus at all, the system must refuse rather
    than promote a nearby token into an answer."""
    pipe = _pipeline_over(tmp_path, monkeypatch, _NO_CREDENTIAL_EXPORT)

    hits = pipe.retriever.retrieve(
        "password of the new gmail account I created for Karan", k=5
    )
    assert hits, "retrieval returned nothing at all"
    assert any("gmail" in h.chunk.text.lower() for h in hits), "gmail chunk not retrieved"

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
    pipe = _pipeline_over(tmp_path, monkeypatch, _NO_CREDENTIAL_EXPORT)
    pipe._llm = _StubLLM("The password is Hunter2! [m:deadbeefdeadbeef].")
    ans = pipe.answer("What is the Gmail password?")
    assert ans.citations == []
    assert ans.supported is False


def test_unlabelled_credential_is_stored_and_reachable_by_exact_text(
    tmp_path, monkeypatch
):
    """The credential IS in the corpus — the first investigation's premise that
    it was absent is false. Structured lookup finds it; only the semantic query
    cannot. This is what makes the xfail below a retrieval gap, not missing data.
    """
    pipe = _pipeline_over(tmp_path, monkeypatch, _UNLABELLED_CREDENTIAL_EXPORT)

    # present in the DB, and adjacent to the gmail discussion
    assert pipe.db.get_messages(contains="@sample7handle"), "credential not stored"
    # but invisible to every word a user would search for
    for word in ("password", "pwd", "login", "credential"):
        assert pipe.db.get_messages(contains=word) == [], (
            f"unexpected {word!r} match — fixture no longer models the real export"
        )


def test_unlabelled_credential_has_no_lexical_overlap_with_the_query(
    tmp_path, monkeypatch
):
    """Why vector-only retrieval cannot close this: the stored credential shares
    no word with any phrasing a user would search by.

    Deliberately *not* an xfail on ``retrieve()``. The suite runs on MockEmbedder
    over tiny fixtures, where top-k returns everything and the query would
    'succeed' for reasons unrelated to semantics — a green test that proves
    nothing. The real measurement lives in docs/PLAN.md's Phase 2 criteria.
    """
    pipe = _pipeline_over(tmp_path, monkeypatch, _UNLABELLED_CREDENTIAL_EXPORT)
    msgs = pipe.db.get_messages(contains="@sample7handle")
    assert msgs, "credential not stored"

    stored = {w.strip("@.,:").lower() for w in msgs[0]["text"].split()}
    query_words = {"what", "is", "my", "gmail", "password"}
    assert not (stored & query_words), (
        "fixture no longer models the failure: the credential must share no "
        "term with the query, which is exactly what defeats cosine similarity"
    )


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
    # chunks are embedded from their embedding view, so that is what an exact
    # match must be queried with (mock embedder -> score ~1.0)
    ans = pipe.answer(render_embedding_text(target.text))
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
