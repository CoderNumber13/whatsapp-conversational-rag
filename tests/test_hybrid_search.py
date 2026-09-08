"""Reciprocal Rank Fusion.

Fusion maths is tested against stub searchers with known rankings, so the
expected RRF scores can be written out by hand. Integration against the real
retrievers is limited to the exact-term regression at the bottom.
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.retrieval.base import ChunkSearcher, ScoredChunk
from src.retrieval.hybrid_search import RRFHybridSearch
from src.storage.models import Chunk


def _chunk(cid: str, text: str = "body") -> Chunk:
    return Chunk(
        chunk_id=cid, conversation_id="c1", seq_start=0, seq_end=1,
        ts_start="2026-01-01T00:00:00", ts_end="2026-01-01T00:01:00",
        participants=["A", "Me"], message_ids=[f"{cid}-m1", f"{cid}-m2"],
        text=text, token_estimate=10,
    )


class StubSearcher:
    """Returns a fixed ranking. Scores are deliberately absurd, to prove fusion
    never looks at them."""

    def __init__(self, ids, scores=None):
        self.ids = list(ids)
        self.scores = scores or [1000.0 - i for i in range(len(self.ids))]
        self.calls = []

    def search(self, query, k, candidate_chunk_ids=None):
        self.calls.append((query, k, candidate_chunk_ids))
        out = []
        for cid, sc in zip(self.ids, self.scores):
            if candidate_chunk_ids is not None and cid not in candidate_chunk_ids:
                continue
            out.append(ScoredChunk(chunk=_chunk(cid), score=sc))
        return out[:k]


def _rrf(k_rrf, *ranks):
    return sum(1.0 / (k_rrf + r) for r in ranks)


@pytest.fixture()
def cfg():
    return Config.reload()


# --- basic fusion -----------------------------------------------------

def test_fuses_ranks_and_ignores_raw_scores(cfg):
    """B is 2nd in both lists; A is 1st in one and absent from the other.
    Agreement wins, which is the entire point of RRF."""
    a = StubSearcher(["A", "B", "C"], scores=[999.0, 1.0, 0.5])
    b = StubSearcher(["D", "B", "C"], scores=[0.9, 0.8, 0.7])
    rrf = RRFHybridSearch([("v", a), ("k", b)], cfg, k_rrf=60)

    hits = rrf.search("q", k=4)
    ids = [h.chunk.chunk_id for h in hits]
    assert ids[0] == "B", f"expected the doubly-ranked chunk first, got {ids}"
    assert hits[0].score == pytest.approx(_rrf(60, 2, 2))


def test_scores_match_the_rrf_formula_exactly(cfg):
    a = StubSearcher(["A", "B"])
    b = StubSearcher(["B", "A"])
    rrf = RRFHybridSearch([("v", a), ("k", b)], cfg, k_rrf=10)
    by_id = {h.chunk.chunk_id: h.score for h in rrf.search("q", k=5)}
    assert by_id["A"] == pytest.approx(_rrf(10, 1, 2))
    assert by_id["B"] == pytest.approx(_rrf(10, 2, 1))


def test_a_chunk_ranked_by_one_retriever_still_appears(cfg):
    """Fusion must not silently drop single-retriever results — only rank them
    lower. This is the credential case's only route into the output."""
    a = StubSearcher(["A", "SOLO"])
    b = StubSearcher(["A", "B"])
    hits = RRFHybridSearch([("v", a), ("k", b)], cfg, k_rrf=60).search("q", k=10)
    ids = [h.chunk.chunk_id for h in hits]
    assert "SOLO" in ids
    assert ids.index("A") < ids.index("SOLO")


# --- missing candidates from one retriever ----------------------------

def test_one_retriever_returning_nothing_degrades_to_the_other(cfg):
    a = StubSearcher(["A", "B", "C"])
    b = StubSearcher([])
    hits = RRFHybridSearch([("v", a), ("k", b)], cfg, k_rrf=60).search("q", k=3)
    assert [h.chunk.chunk_id for h in hits] == ["A", "B", "C"]
    assert all(h.provenance == {"v": i} for i, h in enumerate(hits, 1))


def test_both_retrievers_empty_yields_nothing(cfg):
    rrf = RRFHybridSearch([("v", StubSearcher([])), ("k", StubSearcher([]))], cfg)
    assert rrf.search("q", k=5) == []


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_blank_query_short_circuits_without_calling_searchers(cfg, query):
    a, b = StubSearcher(["A"]), StubSearcher(["B"])
    assert RRFHybridSearch([("v", a), ("k", b)], cfg).search(query, k=5) == []
    assert a.calls == [] and b.calls == []


def test_empty_candidate_set_returns_nothing(cfg):
    a, b = StubSearcher(["A"]), StubSearcher(["B"])
    rrf = RRFHybridSearch([("v", a), ("k", b)], cfg)
    assert rrf.search("q", k=5, candidate_chunk_ids=set()) == []
    assert a.calls == [] and b.calls == []


def test_candidate_filter_is_passed_through_to_every_searcher(cfg):
    a, b = StubSearcher(["A", "B"]), StubSearcher(["B", "A"])
    rrf = RRFHybridSearch([("v", a), ("k", b)], cfg)
    hits = rrf.search("q", k=5, candidate_chunk_ids={"B"})
    assert [h.chunk.chunk_id for h in hits] == ["B"]
    assert a.calls[0][2] == {"B"} and b.calls[0][2] == {"B"}


# --- duplicates -------------------------------------------------------

def test_a_chunk_repeated_by_one_searcher_is_counted_once_at_its_best_rank(cfg):
    """A buggy or overlapping searcher must not be able to inflate a chunk by
    returning it twice."""
    dupe = StubSearcher(["A", "B", "A"])
    other = StubSearcher(["B"])
    rrf = RRFHybridSearch([("v", dupe), ("k", other)], cfg, k_rrf=60)
    hits = {h.chunk.chunk_id: h for h in rrf.search("q", k=5)}

    assert len([h for h in rrf.search("q", k=5) if h.chunk.chunk_id == "A"]) == 1
    assert hits["A"].score == pytest.approx(_rrf(60, 1))       # best rank only
    assert hits["A"].provenance == {"v": 1}
    assert hits["B"].score == pytest.approx(_rrf(60, 2, 1))


# --- determinism ------------------------------------------------------

def test_ties_break_on_chunk_id_not_insertion_order(cfg):
    """Symmetric rankings make every score identical; ordering must still be
    stable and independent of the order searchers were consulted."""
    # mirrored rankings: each chunk is 1st in one list and 2nd in the other, so
    # every RRF score is identical and only the tie-break can decide order
    fwd = RRFHybridSearch(
        [("v", StubSearcher(["b", "a"])), ("k", StubSearcher(["a", "b"]))],
        cfg, k_rrf=60)
    rev = RRFHybridSearch(
        [("v", StubSearcher(["a", "b"])), ("k", StubSearcher(["b", "a"]))],
        cfg, k_rrf=60)

    hits = fwd.search("q", k=2)
    assert hits[0].score == pytest.approx(hits[1].score), "expected a genuine tie"
    ids_fwd = [h.chunk.chunk_id for h in hits]
    assert ids_fwd == ["a", "b"], "tied results must be ordered by chunk_id"
    assert ids_fwd == [h.chunk.chunk_id for h in rev.search("q", k=2)]


def test_repeated_searches_are_identical(cfg):
    rrf = RRFHybridSearch(
        [("v", StubSearcher(list("abcdef"))), ("k", StubSearcher(list("fedcba")))],
        cfg, k_rrf=60)
    runs = [[(h.chunk.chunk_id, h.score) for h in rrf.search("q", k=6)] for _ in range(5)]
    assert all(r == runs[0] for r in runs)


# --- configuration ----------------------------------------------------

def test_rrf_k_is_configurable_and_changes_the_ranking(cfg):
    """Small K makes a strong single-retriever hit competitive; large K makes
    agreement dominate. This is the parameter's whole behaviour."""
    # SOLO is 1st for one retriever only; X is 5th for both. Which wins is
    # decided entirely by K.
    # filler is distinct per searcher, so X is the ONLY shared chunk
    solo_first = StubSearcher(["SOLO", "v1", "v2", "v3", "X"])
    pair = StubSearcher(["Y", "k1", "k2", "k3", "X"])

    small = RRFHybridSearch([("v", solo_first), ("k", pair)], cfg, k_rrf=1)
    large = RRFHybridSearch([("v", solo_first), ("k", pair)], cfg, k_rrf=1000)
    # K=1:   SOLO 1/2 = 0.500  vs  X 2/6 = 0.333  -> rank wins
    assert small.search("q", k=5)[0].chunk.chunk_id == "SOLO"
    # K=1000: SOLO 1/1001 = 0.000999 vs X 2/1005 = 0.00199 -> agreement wins
    assert large.search("q", k=5)[0].chunk.chunk_id != "SOLO"
    assert large.search("q", k=5)[0].provenance.keys() == {"v", "k"}


def test_rrf_k_defaults_to_the_documented_value(cfg):
    assert cfg.rrf_k == 60, "documented default from Cormack et al. (2009)"
    assert RRFHybridSearch([("v", StubSearcher([]))], cfg).k_rrf == 60


def test_rrf_k_and_candidates_come_from_env(monkeypatch):
    monkeypatch.setenv("RRF_K", "7")
    monkeypatch.setenv("RRF_CANDIDATES", "13")
    c = Config.reload()
    rrf = RRFHybridSearch([("v", StubSearcher([]))], c)
    assert (rrf.k_rrf, rrf.candidates) == (7, 13)
    Config.reload()


def test_candidate_depth_is_what_each_searcher_is_asked_for(cfg):
    a = StubSearcher(list("abcdefghij"))
    RRFHybridSearch([("v", a)], cfg, candidates=25).search("q", k=3)
    assert a.calls[0][1] == 25, "must read deeper than k before fusing"


def test_depth_never_falls_below_k(cfg):
    a = StubSearcher(list("abc"))
    RRFHybridSearch([("v", a)], cfg, candidates=2).search("q", k=9)
    assert a.calls[0][1] == 9


def test_rejects_invalid_configuration(cfg):
    with pytest.raises(ValueError, match="at least one searcher"):
        RRFHybridSearch([], cfg)
    with pytest.raises(ValueError, match="unique for provenance"):
        RRFHybridSearch([("v", StubSearcher([])), ("v", StubSearcher([]))], cfg)
    with pytest.raises(ValueError, match="k_rrf must be"):
        RRFHybridSearch([("v", StubSearcher([]))], cfg, k_rrf=-1)


# --- metadata / provenance --------------------------------------------

def test_chunk_metadata_survives_fusion_intact(cfg):
    a = StubSearcher(["A"])
    b = StubSearcher(["A"])
    hit = RRFHybridSearch([("v", a), ("k", b)], cfg).search("q", k=1)[0]
    original = _chunk("A")
    assert hit.chunk.chunk_id == original.chunk_id
    assert hit.chunk.conversation_id == original.conversation_id
    assert hit.chunk.message_ids == original.message_ids
    assert hit.chunk.participants == original.participants
    assert hit.chunk.text == original.text
    assert (hit.chunk.ts_start, hit.chunk.ts_end) == (original.ts_start, original.ts_end)


def test_provenance_records_every_contributing_searcher_and_rank(cfg):
    a = StubSearcher(["X", "A"])
    b = StubSearcher(["A", "Y"])
    by_id = {h.chunk.chunk_id: h.provenance for h in
             RRFHybridSearch([("v", a), ("k", b)], cfg).search("q", k=5)}
    assert by_id["A"] == {"v": 2, "k": 1}
    assert by_id["X"] == {"v": 1}
    assert by_id["Y"] == {"k": 2}


def test_rrf_satisfies_the_shared_searcher_protocol(cfg):
    assert isinstance(RRFHybridSearch([("v", StubSearcher([]))], cfg), ChunkSearcher)


# --- integration: the exact-term cases --------------------------------

@pytest.fixture(scope="module")
def real_rrf(tmp_path_factory):
    pytest.importorskip("faiss")
    from src.chunking.message_chunker import FixedCountChunker
    from src.chunking.service import ChunkingService
    from src.evaluation.dataset import build_corpus
    from src.evaluation.runner import SAMPLE_DIR
    from src.ingestion.normalizer import WhatsAppExportSource
    from src.retrieval.indexer import EmbeddingIndexer
    from src.retrieval.keyword_search import BM25Search
    from src.retrieval.vector_search import VectorSearch
    from src.embeddings.mock import MockEmbedder
    from src.storage.database import Database

    tmp = tmp_path_factory.mktemp("rrf-real")
    import os
    os.environ["INDEX_DIR"] = str(tmp / "idx")
    cfg = Config.reload()

    files = build_corpus(tmp / "corpus", SAMPLE_DIR)
    src = WhatsAppExportSource(files, me_names=["Me"])
    src.ingest()
    db = Database(tmp / "cm.db")
    for c in src.get_conversations():
        db.upsert_conversation(c, me_names=["Me"])
    db.insert_messages(src.get_messages())
    ChunkingService(db, FixedCountChunker(8, 2, 3)).sync_all()
    indexer = EmbeddingIndexer(db, embedder=MockEmbedder(64), config=cfg)
    indexer.sync()

    yield RRFHybridSearch(
        [("vector", VectorSearch(db, indexer.embedder, indexer.store)),
         ("bm25", BM25Search.open(db, cfg))], cfg)
    db.close()
    Config.reload()


@pytest.mark.parametrize("query,must_contain", [
    ("TCS", "TCS"),                              # E6
    ("CGPA", "CGPA"),                            # E7
    ("gaming night", "gaming night"),            # E5
    ("What is Sneha building?", "Sneha"),        # E1
])
def test_exact_term_cases_survive_fusion(real_rrf, query, must_contain):
    """BM25 ranks these first on its own; fusion with a non-semantic mock
    embedder must not bury them."""
    hits = real_rrf.search(query, k=5)
    assert hits, f"{query!r} retrieved nothing"
    assert any(must_contain.lower() in h.chunk.text.lower() for h in hits[:3]), (
        f"{query!r} lost its exact-term match in the top 3 after fusion"
    )


def test_fusion_output_carries_provenance_from_real_searchers(real_rrf):
    hits = real_rrf.search("Microsoft internship", k=5)
    assert hits
    assert all(h.provenance for h in hits)
    assert all(set(h.provenance) <= {"vector", "bm25"} for h in hits)


# --- the property that explains the credential regression --------------

def test_agreement_strictly_dominates_rank_at_the_default_settings(cfg):
    """At RRF_K=60 with a 50-deep candidate list, ANY chunk both retrievers
    return outranks ANY chunk only one returns — whatever their positions:

        best single-retriever score = 1/(60+1)  = 0.016393
        worst two-retriever score   = 2/(60+50) = 0.018182

    This is not a bug, it is what "reciprocal rank fusion" means; but it is
    exactly why fusion demoted the credential chunk from vector rank 6 to 28.
    Only one retriever can see that chunk, so it can never outrank the crowd of
    chunks both retrievers agree on.

    Pinned so that a future change to RRF_K or RRF_CANDIDATES surfaces this
    trade-off deliberately rather than silently.
    """
    best_single = 1.0 / (cfg.rrf_k + 1)
    worst_double = 2.0 / (cfg.rrf_k + cfg.rrf_candidates)
    assert worst_double > best_single

    # demonstrated end to end: SOLO is rank 1 for one retriever, LAST is at the
    # very bottom of both, and LAST still wins
    depth = cfg.rrf_candidates
    a = StubSearcher(["SOLO"] + [f"v{i}" for i in range(depth - 2)] + ["LAST"])
    b = StubSearcher([f"k{i}" for i in range(depth - 1)] + ["LAST"])
    hits = RRFHybridSearch([("v", a), ("k", b)], cfg).search("q", k=2)
    assert hits[0].chunk.chunk_id == "LAST"
    assert hits[0].provenance.keys() == {"v", "k"}
