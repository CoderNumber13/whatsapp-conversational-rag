"""BM25 lexical search: tokenization, indexing, ranking, and the exact-term
cases dense retrieval could not reach.

Ranks are asserted, never raw BM25 scores — scores are corpus-relative and shift
whenever the corpus does, so pinning them would make every corpus edit a test
failure.
"""

from __future__ import annotations

import pytest

from src.chunking.message_chunker import FixedCountChunker
from src.chunking.service import ChunkingService
from src.config import Config
from src.evaluation.dataset import build_corpus
from src.evaluation.runner import SAMPLE_DIR
from src.ingestion.normalizer import WhatsAppExportSource
from src.retrieval.base import ChunkSearcher, ScoredChunk
from src.retrieval.keyword_search import BM25Search, tokenize
from src.storage.database import Database


# --- tokenization -----------------------------------------------------

def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Hello, WORLD!") == ["hello", "world"]


def test_tokenize_drops_only_the_small_stopword_set():
    # "password" and "gmail" must survive — they are the query's whole content
    assert tokenize("What is my gmail password?") == ["what", "my", "gmail", "password"]


def test_tokenize_keeps_acronyms_and_alphanumerics_intact():
    assert "tcs" in tokenize("TCS and Infosys")
    assert "cgpa" in tokenize("Fill: name, college, CGPA")
    assert "0.71" in tokenize("baseline accuracy 0.71")


def test_tokenize_strips_leading_at_so_handles_match_either_way():
    """Handles get written both with and without the @."""
    assert tokenize("@sample7handle") == tokenize("sample7handle") == ["sample7handle"]


def test_tokenize_preserves_dotted_and_hyphenated_terms():
    # "on" is a stopword and is dropped; the compound terms survive whole
    assert tokenize("test.user got gpt-4 on 2026-08-21") == [
        "test.user", "got", "gpt-4", "2026-08-21",
    ]


def test_tokenize_of_empty_or_symbol_only_text_is_empty():
    assert tokenize("") == []
    assert tokenize("??? !!! ---") == []


# --- fixture ----------------------------------------------------------

@pytest.fixture()
def searcher(tmp_path, monkeypatch):
    """A BM25 index over the benchmark corpus. No embedder is involved: BM25
    reads the chunks table directly."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cm.db"))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    cfg = Config.reload()

    files = build_corpus(tmp_path / "corpus", SAMPLE_DIR)
    src = WhatsAppExportSource(files, me_names=["Me"])
    src.ingest()
    db = Database(tmp_path / "cm.db")
    for c in src.get_conversations():
        db.upsert_conversation(c, me_names=["Me"])
    db.insert_messages(src.get_messages())
    ChunkingService(db, FixedCountChunker(8, 2, 3)).sync_all()

    yield BM25Search.open(db, cfg), db
    db.close()


def _texts(hits):
    return " ".join(h.chunk.text for h in hits)


# --- indexing ---------------------------------------------------------

def test_index_covers_every_chunk_exactly_once(searcher):
    bm25, db = searcher
    assert len(bm25) == db.count_chunks() > 0
    assert len(set(bm25.chunk_ids)) == len(bm25.chunk_ids)
    assert set(bm25.chunk_ids) == {c.chunk_id for c in db.all_chunks()}


def test_index_preserves_chunk_ids_for_comparison_with_vector_results(searcher):
    """Chunk ids are the join key between the two retrievers' results."""
    bm25, db = searcher
    hits = bm25.search("Microsoft internship", k=3)
    assert hits
    known = {c.chunk_id for c in db.all_chunks()}
    for h in hits:
        assert h.chunk.chunk_id in known


def test_citation_tags_are_not_indexed(searcher):
    """[m:<id>] tags would otherwise add ~12 junk terms per chunk and skew
    BM25's length normalisation."""
    bm25, _ = searcher
    hex_terms = [t for toks in bm25._tokens for t in toks if t.startswith("m:")]
    assert hex_terms == []
    assert bm25.search("m", k=5) == [] or all(
        "[m:" not in h.chunk.chunk_id for h in bm25.search("m", k=5)
    )


def test_empty_corpus_searches_cleanly(tmp_path):
    db = Database(tmp_path / "empty.db")
    bm25 = BM25Search(db, [], [])
    assert len(bm25) == 0
    assert bm25.search("anything", k=5) == []
    db.close()


# --- interface --------------------------------------------------------

def test_bm25_satisfies_the_shared_searcher_protocol(searcher):
    bm25, _ = searcher
    assert isinstance(bm25, ChunkSearcher)
    hits = bm25.search("Goa trip", k=2)
    assert all(isinstance(h, ScoredChunk) for h in hits)


def test_vector_search_satisfies_the_same_protocol():
    """The Protocol must fit the existing retriever without changing it."""
    pytest.importorskip("faiss")
    from src.retrieval.vector_search import VectorSearch

    assert issubclass(VectorSearch, ChunkSearcher)


# --- retrieval behaviour ----------------------------------------------

def test_results_are_ordered_by_descending_score_and_capped_at_k(searcher):
    bm25, _ = searcher
    hits = bm25.search("data science project backend frontend", k=3)
    assert 0 < len(hits) <= 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_zero_scoring_chunks_are_never_returned(searcher):
    """A chunk sharing no term with the query is not a result at rank 10 — it is
    not a result at all."""
    bm25, _ = searcher
    hits = bm25.search("Microsoft", k=100)
    assert all(h.score > 0 for h in hits)
    assert len(hits) < len(bm25), "every chunk matched; query was not selective"


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "??? !!!", "the of and"])
def test_empty_or_contentless_queries_return_nothing(searcher, query):
    bm25, _ = searcher
    assert bm25.search(query, k=5) == []


def test_ranking_is_deterministic_across_repeated_searches(searcher):
    bm25, _ = searcher
    runs = [[(h.chunk.chunk_id, h.score) for h in bm25.search("placements", k=5)]
            for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_ranking_is_deterministic_across_rebuilds_with_shuffled_corpus_order(searcher):
    """Ties must not resolve by insertion order, or ranking drifts whenever
    chunks are re-inserted."""
    bm25, db = searcher
    first = [h.chunk.chunk_id for h in bm25.search("deadline", k=5)]

    pairs = list(zip(bm25.chunk_ids, bm25._tokens))
    reordered = BM25Search(db, [c for c, _ in reversed(pairs)],
                           [t for _, t in reversed(pairs)])
    assert [h.chunk.chunk_id for h in reordered.search("deadline", k=5)] == first


# --- candidate filtering (metadata filters must behave identically) ----

def test_candidate_filter_restricts_results(searcher):
    bm25, db = searcher
    allowed = {db.all_chunks()[0].chunk_id}
    hits = bm25.search("the project deadline", k=10, candidate_chunk_ids=allowed)
    assert {h.chunk.chunk_id for h in hits} <= allowed


def test_empty_candidate_set_returns_nothing_rather_than_falling_back(searcher):
    bm25, _ = searcher
    assert bm25.search("Microsoft internship", k=5, candidate_chunk_ids=set()) == []


# --- regression: the exact-term cases dense retrieval could not reach ---
# At large scale the vector baseline scored 0.0% Recall@1 on exact_term, and
# never retrieved "TCS" or "What is Sneha building?" within the top 10 of 144
# chunks. These are the queries BM25 exists to answer.

@pytest.mark.parametrize("query,must_contain", [
    ("TCS", "TCS"),                       # E6 - vector: never retrieved
    ("CGPA", "CGPA"),                     # E7
    ("gaming night", "gaming night"),     # E5
    ("What is Sneha building?", "Sneha"),  # E1 - vector: never retrieved
    ("movie recommender", "movie recommender"),  # E4
])
def test_exact_term_queries_rank_the_right_chunk_first(searcher, query, must_contain):
    bm25, _ = searcher
    hits = bm25.search(query, k=5)
    assert hits, f"{query!r} retrieved nothing"
    assert must_contain.lower() in hits[0].chunk.text.lower(), (
        f"{query!r} did not rank the chunk containing {must_contain!r} first"
    )


def test_rare_acronym_is_highly_selective(searcher):
    """The mechanism behind the exact-term win: a token appearing in one chunk
    picks out that chunk and essentially nothing else."""
    bm25, _ = searcher
    assert len(bm25.search("CGPA", k=100)) <= 2


def test_a_term_in_half_the_corpus_scores_zero_and_returns_nothing(searcher):
    """Documented BM25Okapi behaviour, not a bug: its IDF is
    log(N-n+0.5) - log(n+0.5), which is exactly 0 when a term appears in half
    the documents and negative beyond that. Such a term carries no discriminating
    evidence, so returning nothing beats returning the whole corpus unordered.

    Pinned because it explains part of BM25's weakness on common-word queries.
    """
    bm25, _ = searcher
    n = len(bm25)
    common = [t for t in {tok for toks in bm25._tokens for tok in toks}
              if sum(t in toks for toks in bm25._tokens) >= n / 2]
    assert common, "corpus has no term frequent enough to exercise this"
    assert bm25.search(common[0], k=10) == []


# --- documented limitation --------------------------------------------

def test_unlabelled_credential_is_reachable_only_by_its_own_token(searcher):
    """BM25 finds the credential only when the query contains the token itself —
    which a user asking "what is my password" cannot do."""
    bm25, _ = searcher
    exact = bm25.search("sample7handle", k=3)
    assert exact and "@sample7handle" in exact[0].chunk.text


@pytest.fixture(scope="module")
def large_searcher(tmp_path_factory):
    """BM25 over the production-scale corpus. Needed for the credential case:
    at sample scale the credential shares its chunk with "synthetic sample message", so
    the query matches on "gmail" and the failure is invisible."""
    from src.evaluation.corpus import generate_large_corpus

    tmp = tmp_path_factory.mktemp("bm25-large")
    files = generate_large_corpus(tmp / "corpus", SAMPLE_DIR)
    src = WhatsAppExportSource(files, me_names=["Me"])
    src.ingest()
    db = Database(tmp / "cm.db")
    for c in src.get_conversations():
        db.upsert_conversation(c, me_names=["Me"])
    db.insert_messages(src.get_messages())
    ChunkingService(db, FixedCountChunker(12, 3, 3)).sync_all()
    yield BM25Search.open(db)
    db.close()


def test_bm25_does_not_fix_the_credential_failure_at_scale(large_searcher):
    """BM25 does NOT fix the reported failure — it makes it worse. The benchmark
    agrees: X1 goes from rank 6 under vector search to no result at all.

    The credential is a bare token, so "What is my gmail password?" shares no
    term with the chunk holding it, while the password DISTRACTORS share both
    "gmail" and "password" and rank confidently. Lexical retrieval needs a term
    in common and there is none.

    Pinned so the limitation is not quietly assumed away when hybrid lands: a
    hybrid of two retrievers that both miss this will still miss it.
    """
    hits = large_searcher.search("What is my gmail password?", k=10)
    assert hits, "expected the distractors to be retrieved"
    assert not any("@sample7handle" in h.chunk.text for h in hits), (
        "credential unexpectedly retrieved — re-check the documented limitation"
    )
