"""Cross-encoder reranking.

Phase 2, step 3. The last stage of the retrieval pipeline:

    vector + BM25 -> RRF -> candidate pool -> RERANKER -> final ranking

Why a cross-encoder can do what the earlier stages cannot: retrieval scores the
query and a chunk *independently* (one embedding each, or IDF-weighted term
overlap) and compares the results. A cross-encoder reads the query and the chunk
**together** in one forward pass, so it can weigh how the two relate rather than
how similar they look. That is the only mechanism in the pipeline capable of
judging a chunk whose relevance is not visible from either side alone.

The trade is cost: no index, one model pass per candidate, so it can only be
applied to a shortlist. Depth is ``RERANK_CANDIDATES``.

Scores are cross-encoder logits: unbounded, roughly [-11, +11] for this model,
and not comparable with cosine similarity, BM25 sums or RRF values.
``MIN_RETRIEVAL_SCORE`` is a cosine threshold and MUST NOT be applied to them;
no threshold is introduced here.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence, runtime_checkable

from src.chunking.base import render_embedding_text
from src.config import CONFIG, Config
from src.retrieval.base import ChunkSearcher, ScoredChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Reorder candidates by relevance to the query.

    Implementations must return exactly the candidates they were given — same
    chunks, no additions, no drops, no duplicates — reordered by descending
    relevance, and must be deterministic for a fixed query and candidate set.
    """

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        ...


class IdentityReranker:
    """Passes candidates through untouched. The control condition: makes the
    reranking stage measurable against itself."""

    name = "identity"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return list(candidates)


class CrossEncoderReranker:
    """Local cross-encoder. Default is the MS MARCO MiniLM model: 6 layers,
    ~80MB, CPU-friendly, and trained for exactly this query/passage relevance
    task."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        config: Config = CONFIG,
        *,
        max_length: Optional[int] = None,
        batch_size: int = 32,
    ) -> None:
        self.config = config
        self.model_id = model_id or config.rerank_model
        self.max_length = max_length or config.rerank_max_length
        self.batch_size = batch_size
        self._model = None

    @property
    def name(self) -> str:
        return self.model_id

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, max_length=self.max_length)
        return self._model

    def warmup(self) -> None:
        _ = self.model

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        if not query.strip():
            return list(candidates)

        # Same view the retrievers index: [m:<id>] tags and per-line timestamps
        # are scaffolding, and here they also consume the 512-token budget the
        # cross-encoder has to read the chunk within.
        pairs = [(query, render_embedding_text(c.chunk.text)) for c in candidates]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        rescored = [
            ScoredChunk(
                chunk=c.chunk,
                score=float(s),
                # keep where the candidate came from, and where it stood before
                provenance={**(c.provenance or {}), "pre_rerank": i},
            )
            for i, (c, s) in enumerate(zip(candidates, scores), start=1)
        ]
        # Ties break on chunk_id so ordering never depends on candidate order.
        rescored.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
        return rescored


class RerankedSearch:
    """A ``ChunkSearcher`` that reranks another searcher's shortlist.

    Composes rather than modifies: the wrapped searcher (RRF, or either
    retriever alone) is used exactly as-is and remains independently usable.
    """

    def __init__(
        self,
        base: ChunkSearcher,
        reranker: Reranker,
        config: Config = CONFIG,
        *,
        candidates: Optional[int] = None,
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.config = config
        self.candidates = (
            config.rerank_candidates if candidates is None else candidates
        )
        if self.candidates < 1:
            raise ValueError("rerank candidate depth must be >= 1")

    def search(
        self,
        query: str,
        k: int,
        candidate_chunk_ids: Optional[set[str]] = None,
    ) -> list[ScoredChunk]:
        # Read at least k, but rerank no more than the configured depth: the
        # reranker is the expensive stage and its cost is linear in candidates.
        pool = self.base.search(query, max(self.candidates, k), candidate_chunk_ids)
        if not pool:
            return []
        return self.reranker.rerank(query, pool[: max(self.candidates, k)])[:k]
