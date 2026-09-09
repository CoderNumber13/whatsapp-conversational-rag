"""Cross-encoder reranking stage.

The contract is tested against a stub reranker with hand-chosen scores, so the
expected ordering is unambiguous. One test exercises the real cross-encoder and
skips if the model is not available locally.
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.retrieval.base import ChunkSearcher, ScoredChunk
from src.retrieval.reranker import (
    CrossEncoderReranker,
    IdentityReranker,
    Reranker,
    RerankedSearch,
)
from src.storage.models import Chunk


def _chunk(cid: str, text: str = "body") -> Chunk:
    return Chunk(
        chunk_id=cid, conversation_id="c1", seq_start=0, seq_end=1,
        ts_start="2026-01-01T00:00:00", ts_end="2026-01-01T00:01:00",
        participants=["A", "Me"], message_ids=[f"{cid}-m1", f"{cid}-m2"],
        text=text, token_estimate=10,
    )


def _cand(cid, score=0.0, prov=None, text="body"):
    return ScoredChunk(chunk=_chunk(cid, text), score=score, provenance=prov)


class StubSearcher:
    def __init__(self, ids):
        self.ids = list(ids)
        self.calls = []

    def search(self, query, k, candidate_chunk_ids=None):
        self.calls.append((query, k, candidate_chunk_ids))
        ids = [i for i in self.ids
               if candidate_chunk_ids is None or i in candidate_chunk_ids]
        return [_cand(i, 1.0 / n, {"vector": n}) for n, i in enumerate(ids, 1)][:k]


class StubReranker:
    """Scores from a table; unknown chunks score 0. Deterministic by design."""

    name = "stub"

    def __init__(self, table):
        self.table = dict(table)
        self.seen = []

    def rerank(self, query, candidates):
        self.seen.append((query, [c.chunk.chunk_id for c in candidates]))
        out = [ScoredChunk(chunk=c.chunk, score=self.table.get(c.chunk.chunk_id, 0.0),
                           provenance=c.provenance)
               for c in candidates]
        out.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
        return out


@pytest.fixture()
def cfg():
    return Config.reload()


# --- interface --------------------------------------------------------

def test_stub_and_real_rerankers_satisfy_the_protocol(cfg):
    assert isinstance(IdentityReranker(), Reranker)
    assert isinstance(StubReranker({}), Reranker)
    assert isinstance(CrossEncoderReranker(config=cfg), Reranker)


def test_reranked_search_is_itself_a_chunk_searcher(cfg):
    rs = RerankedSearch(StubSearcher(["a"]), IdentityReranker(), cfg)
    assert isinstance(rs, ChunkSearcher)


def test_identity_reranker_leaves_order_untouched(cfg):
    cands = [_cand("a", 3.0), _cand("b", 2.0), _cand("c", 1.0)]
    assert [c.chunk.chunk_id for c in IdentityReranker().rerank("q", cands)] == \
        ["a", "b", "c"]


# --- score ordering ---------------------------------------------------

def test_results_are_ordered_by_reranker_score_not_input_order(cfg):
    """The whole point: the reranker must be able to invert the input order."""
    base = StubSearcher(["a", "b", "c"])          # base likes a > b > c
    rr = StubReranker({"a": -5.0, "b": 1.0, "c": 9.0})
    hits = RerankedSearch(base, rr, cfg, candidates=3).search("q", k=3)
    assert [h.chunk.chunk_id for h in hits] == ["c", "b", "a"]
    assert [h.score for h in hits] == [9.0, 1.0, -5.0]


def test_scores_are_the_rerankers_not_the_base_searchers(cfg):
    base = StubSearcher(["a", "b"])
    hits = RerankedSearch(base, StubReranker({"a": 7.5, "b": -2.5}), cfg).search("q", k=2)
    assert [h.score for h in hits] == [7.5, -2.5]


def test_negative_scores_are_preserved_not_clamped(cfg):
    """Cross-encoder logits are routinely negative; nothing may floor them."""
    hits = RerankedSearch(StubSearcher(["a"]), StubReranker({"a": -11.3}), cfg).search("q", k=1)
    assert hits[0].score == pytest.approx(-11.3)


# --- candidate preservation ------------------------------------------

def test_reranking_preserves_the_candidate_set_exactly(cfg):
    cands = [_cand(c) for c in "abcde"]
    out = StubReranker({"c": 5.0, "a": 1.0}).rerank("q", cands)
    assert len(out) == len(cands)
    assert {c.chunk.chunk_id for c in out} == {c.chunk.chunk_id for c in cands}


def test_reranking_never_duplicates_a_candidate(cfg):
    cands = [_cand(c) for c in "abcd"]
    ids = [c.chunk.chunk_id for c in StubReranker({}).rerank("q", cands)]
    assert len(ids) == len(set(ids))


def test_chunk_metadata_survives_reranking(cfg):
    original = _chunk("a", "the original body")
    out = StubReranker({"a": 1.0}).rerank("q", [ScoredChunk(chunk=original, score=0.1)])
    got = out[0].chunk
    assert got.chunk_id == original.chunk_id
    assert got.conversation_id == original.conversation_id
    assert got.message_ids == original.message_ids
    assert got.participants == original.participants
    assert got.text == original.text
    assert (got.ts_start, got.ts_end) == (original.ts_start, original.ts_end)


def test_base_provenance_is_carried_through(cfg):
    base = StubSearcher(["a", "b"])
    hits = RerankedSearch(base, StubReranker({"b": 1.0}), cfg).search("q", k=2)
    assert all(h.provenance for h in hits)
    assert all("vector" in h.provenance for h in hits)


# --- empty / degenerate ----------------------------------------------

def test_empty_candidate_list_reranks_to_empty(cfg):
    assert StubReranker({}).rerank("q", []) == []
    assert CrossEncoderReranker(config=cfg).rerank("q", []) == []


def test_empty_base_result_short_circuits_the_reranker(cfg):
    rr = StubReranker({})
    assert RerankedSearch(StubSearcher([]), rr, cfg).search("q", k=5) == []
    assert rr.seen == [], "reranker must not be invoked with nothing to rank"


def test_blank_query_returns_candidates_unreordered(cfg):
    cands = [_cand("a"), _cand("b")]
    assert CrossEncoderReranker(config=cfg).rerank("   ", cands) == cands


def test_empty_candidate_id_filter_returns_nothing(cfg):
    rs = RerankedSearch(StubSearcher(["a", "b"]), StubReranker({}), cfg)
    assert rs.search("q", k=5, candidate_chunk_ids=set()) == []


# --- determinism ------------------------------------------------------

def test_repeated_reranking_is_identical(cfg):
    rs = RerankedSearch(StubSearcher(list("abcdef")),
                        StubReranker({"c": 2.0, "e": 2.0, "a": 1.0}), cfg)
    runs = [[(h.chunk.chunk_id, h.score) for h in rs.search("q", k=6)] for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_tied_scores_break_on_chunk_id(cfg):
    """Every candidate scoring the same must still order reproducibly."""
    out = StubReranker({}).rerank("q", [_cand("c"), _cand("a"), _cand("b")])
    assert [x.chunk.chunk_id for x in out] == ["a", "b", "c"]


def test_ordering_does_not_depend_on_the_order_candidates_arrive(cfg):
    table = {"a": 3.0, "b": 3.0, "c": 1.0}
    fwd = StubReranker(table).rerank("q", [_cand(c) for c in "abc"])
    rev = StubReranker(table).rerank("q", [_cand(c) for c in "cba"])
    assert [x.chunk.chunk_id for x in fwd] == [x.chunk.chunk_id for x in rev]


# --- configurable depth ----------------------------------------------

def test_candidate_depth_controls_how_many_are_reranked(cfg):
    base = StubSearcher(list("abcdefghij"))
    rr = StubReranker({})
    RerankedSearch(base, rr, cfg, candidates=4).search("q", k=2)
    assert base.calls[0][1] == 4, "base must be asked for the rerank depth"
    assert len(rr.seen[0][1]) == 4, "only the shortlist may reach the reranker"


def test_depth_never_falls_below_k(cfg):
    base = StubSearcher(list("abcdefgh"))
    rr = StubReranker({})
    RerankedSearch(base, rr, cfg, candidates=2).search("q", k=6)
    assert base.calls[0][1] == 6
    assert len(rr.seen[0][1]) == 6


def test_only_k_results_are_returned(cfg):
    rs = RerankedSearch(StubSearcher(list("abcdefghij")), StubReranker({}), cfg,
                        candidates=10)
    assert len(rs.search("q", k=3)) == 3


def test_depth_defaults_to_config_and_rejects_nonsense(cfg):
    assert RerankedSearch(StubSearcher([]), IdentityReranker(), cfg).candidates == \
        cfg.rerank_candidates
    with pytest.raises(ValueError, match="depth must be"):
        RerankedSearch(StubSearcher([]), IdentityReranker(), cfg, candidates=0)


def test_rerank_settings_come_from_env(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "some/other-model")
    monkeypatch.setenv("RERANK_CANDIDATES", "7")
    monkeypatch.setenv("RERANK_MAX_LENGTH", "128")
    c = Config.reload()
    assert (c.rerank_model, c.rerank_candidates, c.rerank_max_length) == \
        ("some/other-model", 7, 128)
    ce = CrossEncoderReranker(config=c)
    assert ce.model_id == "some/other-model" and ce.max_length == 128
    Config.reload()


# --- the real model ---------------------------------------------------

@pytest.fixture(scope="module")
def real_ce():
    cfg = Config.reload()
    ce = CrossEncoderReranker(config=cfg)
    try:
        ce.warmup()
    except Exception as exc:                       # no model cached / offline
        pytest.skip(f"cross-encoder unavailable: {exc}")
    return ce


def test_cross_encoder_ranks_a_relevant_chunk_above_an_irrelevant_one(real_ce):
    cands = [
        _cand("irrelevant", text="[2026-01-01 10:00] A: mess ka khana theek tha"),
        _cand("relevant", text="[2026-01-01 10:00] A: Joining date is 2 September, "
                               "Bangalore office"),
    ]
    out = real_ce.rerank("When does Rahul join and where?", cands)
    assert out[0].chunk.chunk_id == "relevant"
    assert out[0].score > out[1].score


def test_cross_encoder_preserves_and_does_not_duplicate_candidates(real_ce):
    cands = [_cand(c, text=f"[2026-01-01 10:00] A: message {c}") for c in "abcdef"]
    out = real_ce.rerank("anything at all", cands)
    ids = [c.chunk.chunk_id for c in out]
    assert sorted(ids) == sorted(c.chunk.chunk_id for c in cands)
    assert len(ids) == len(set(ids))


def test_cross_encoder_is_deterministic(real_ce):
    cands = [_cand(c, text=f"[2026-01-01 10:00] A: message about {c}") for c in "abcd"]
    a = [(h.chunk.chunk_id, round(h.score, 5)) for h in real_ce.rerank("q about b", cands)]
    b = [(h.chunk.chunk_id, round(h.score, 5)) for h in real_ce.rerank("q about b", cands)]
    assert a == b


def test_cross_encoder_records_the_pre_rerank_position(real_ce):
    cands = [_cand(c, text=f"[2026-01-01 10:00] A: text {c}") for c in "abc"]
    out = real_ce.rerank("something", cands)
    assert all("pre_rerank" in h.provenance for h in out)
    assert sorted(h.provenance["pre_rerank"] for h in out) == [1, 2, 3]
