"""Reciprocal Rank Fusion of independent searchers.

Phase 2, step 2. The dense and lexical retrievers are complementary — at large
scale BM25 takes ``exact_term`` Recall@1 from 0.0% to 85.7% while losing 37
points on ``paraphrase`` — so the question is whether combining them keeps both
strengths.

Why rank fusion and not score fusion
------------------------------------
The two score scales are not commensurable. Cosine similarity is bounded in
[-1, 1] and clusters around 0.2-0.5 here; BM25 is an unbounded corpus-relative
sum reaching 20.8 on the same corpus. Adding them lets BM25 decide every
ranking; rescaling them (min-max, z-score) invents a comparison that the numbers
do not support and shifts with corpus composition. RRF discards magnitudes
entirely and uses only position:

    score(d) = Σ_r  1 / (K + rank_r(d))

summed over the retrievers that returned ``d``, with ranks 1-based. A document
two retrievers both rank modestly well beats one that a single retriever loves,
which is the whole point. K (``RRF_K``, default 60) damps the head of the curve.

What fusion cannot do
---------------------
RRF only reorders the union of its inputs. A chunk that neither retriever
returns within ``RRF_CANDIDATES`` cannot appear in the output at any K. Fusion
improves ranking, never reach.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from src.config import CONFIG, Config
from src.retrieval.base import ChunkSearcher, ScoredChunk


class RRFHybridSearch:
    """Fuse any number of ``ChunkSearcher``s by Reciprocal Rank Fusion.

    Satisfies ``ChunkSearcher`` itself, so it can stand anywhere a single
    searcher can — including inside another fusion. The component searchers are
    untouched and remain independently usable.
    """

    def __init__(
        self,
        searchers: Sequence[tuple[str, ChunkSearcher]],
        config: Config = CONFIG,
        *,
        k_rrf: Optional[int] = None,
        candidates: Optional[int] = None,
    ) -> None:
        if not searchers:
            raise ValueError("RRFHybridSearch needs at least one searcher")
        names = [n for n, _ in searchers]
        if len(set(names)) != len(names):
            raise ValueError(f"searcher names must be unique for provenance: {names}")
        self.searchers = list(searchers)
        self.config = config
        self.k_rrf = config.rrf_k if k_rrf is None else k_rrf
        self.candidates = config.rrf_candidates if candidates is None else candidates
        if self.k_rrf < 0:
            raise ValueError("k_rrf must be >= 0")

    @classmethod
    def open(cls, db, config: Config = CONFIG, **kwargs) -> "RRFHybridSearch":
        """The standard pairing: dense + lexical over the same chunks."""
        from src.retrieval.keyword_search import BM25Search
        from src.retrieval.vector_search import VectorSearch

        return cls(
            [("vector", VectorSearch.open(db, config)),
             ("bm25", BM25Search.open(db, config))],
            config, **kwargs,
        )

    def search(
        self,
        query: str,
        k: int,
        candidate_chunk_ids: Optional[set[str]] = None,
    ) -> list[ScoredChunk]:
        if not query.strip():
            return []
        if candidate_chunk_ids is not None and not candidate_chunk_ids:
            return []

        depth = max(self.candidates, k)
        fused: dict[str, float] = {}
        provenance: dict[str, dict[str, int]] = {}
        chunks: dict[str, object] = {}

        for name, searcher in self.searchers:
            hits = searcher.search(query, depth, candidate_chunk_ids)
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk.chunk_id
                # A searcher returning the same chunk twice must not be counted
                # twice; keep its best (lowest) rank.
                if name in provenance.get(cid, {}):
                    continue
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.k_rrf + rank)
                provenance.setdefault(cid, {})[name] = rank
                chunks.setdefault(cid, hit.chunk)

        # Ties are common in RRF — two chunks found at the same rank by the same
        # number of retrievers score identically — so break on chunk_id to keep
        # ranking reproducible rather than dependent on dict insertion order.
        order = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))

        return [
            ScoredChunk(chunk=chunks[cid], score=score, provenance=dict(provenance[cid]))
            for cid, score in order[:k]
        ]

    def candidate_ids(
        self, query: str, candidate_chunk_ids: Optional[set[str]] = None
    ) -> dict[str, list[str]]:
        """Each component's candidate list, for diagnosing what fusion had to
        work with. Used to answer "could RRF have retrieved this at all?"."""
        return {
            name: [h.chunk.chunk_id
                   for h in s.search(query, self.candidates, candidate_chunk_ids)]
            for name, s in self.searchers
        }
