"""The shared retrieval interface.

Both the dense (FAISS) and lexical (BM25) searchers return the same type and
take the same arguments, so the retriever can hold either one — and, later, a
fusion of both — without knowing which it has.

``ChunkSearcher`` is a Protocol, i.e. structural: ``VectorSearch`` satisfies it
without inheriting from anything and without a single behavioural change. The
only thing genuinely shared is ``ScoredChunk``, which lives here so neither
implementation has to import the other.

Scores are NOT comparable across implementations: cosine similarity is bounded
in [-1, 1] while BM25 is an unbounded corpus-relative sum. Compare ranks between
searchers, never raw scores. Fusing them therefore needs rank-based combination
(RRF) or per-searcher normalisation — deliberately not done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from src.storage.models import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@runtime_checkable
class ChunkSearcher(Protocol):
    """Rank chunks for a query, best first.

    Implementations must:
      * return at most ``k`` results, ordered by descending score;
      * return ``[]`` for a blank query or an empty index;
      * honour ``candidate_chunk_ids`` when given — an empty set means "nothing
        is eligible" and must yield ``[]``, never a fallback to the full corpus;
      * be deterministic: the same query against the same corpus ranks the same.
    """

    def search(
        self,
        query: str,
        k: int,
        candidate_chunk_ids: Optional[set[str]] = None,
    ) -> list[ScoredChunk]:
        ...
