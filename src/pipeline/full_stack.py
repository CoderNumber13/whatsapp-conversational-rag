"""The full retrieval stack, composed for end-to-end answering.

    query -> vector search
          -> BM25
          -> RRF
          -> cross-encoder reranking
          -> top-N evidence chunks
          -> LLM -> grounded answer + citations

Pure composition. Every stage is used exactly as built and measured in the
retrieval experiments; nothing here modifies vector search, BM25, RRF or the
cross-encoder, and ``RagPipeline`` is untouched — the assembled searcher is
injected through the ``Retriever`` seam that already exists.

On the abstention gate
----------------------
``RagPipeline.answer`` abstains before calling the LLM when the top retrieval
score falls below ``MIN_RETRIEVAL_SCORE``. That constant is a **cosine**
threshold calibrated for the dense retriever. Once the cross-encoder is in the
stack the top score is an unbounded logit, typically negative, so comparing it
against 0.25 would abstain on essentially everything — including questions the
system answers correctly.

Experiments 5 and 6 established that no absolute-score or margin threshold on
cross-encoder output is well enough calibrated to gate on, so this layer does
not introduce one. Instead the retrieval gate is disabled (``min_score=-inf``)
and abstention is left entirely to the grounded prompt, which refuses by
emitting NOT_FOUND. That is a deliberate design choice, not an oversight:
the decision moves from an uncalibrated number to the model that can actually
read the evidence.
"""

from __future__ import annotations

import math
from typing import Optional

from src.config import CONFIG, Config
from src.pipeline.rag_pipeline import Answer, RagPipeline
from src.retrieval.base import ChunkSearcher
from src.retrieval.retriever import RetrievalFilters, Retriever

# The retrieval gate is bypassed rather than retuned; see the module docstring.
NO_RETRIEVAL_GATE = -math.inf


def build_searcher(db, config: Config = CONFIG, *, rerank: bool = True,
                   rerank_depth: Optional[int] = None) -> ChunkSearcher:
    """vector + BM25 -> RRF -> (optional) cross-encoder."""
    from src.retrieval.hybrid_search import RRFHybridSearch

    searcher: ChunkSearcher = RRFHybridSearch.open(db, config)
    if rerank:
        from src.retrieval.reranker import CrossEncoderReranker, RerankedSearch

        reranker = CrossEncoderReranker(config=config)
        reranker.warmup()
        searcher = RerankedSearch(searcher, reranker, config,
                                  candidates=rerank_depth)
    return searcher


class GroundedAnswerer:
    """End-to-end answering over the full stack.

    Thin wrapper over ``RagPipeline``: it owns the stack assembly and the
    decision to leave abstention to the prompt, and delegates retrieval,
    prompting, citation validation and answer construction to the pipeline
    unchanged.
    """

    def __init__(
        self,
        pipeline: RagPipeline,
        *,
        top_n: Optional[int] = None,
        rerank: bool = True,
        rerank_depth: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = pipeline.config
        self.top_n = top_n or self.config.max_context_chunks
        self.pipeline._retriever = Retriever(
            pipeline.db,
            build_searcher(pipeline.db, self.config, rerank=rerank,
                           rerank_depth=rerank_depth),
            self.config,
        )

    @property
    def db(self):
        return self.pipeline.db

    def answer(
        self,
        question: str,
        filters: Optional[RetrievalFilters] = None,
        *,
        k: Optional[int] = None,
    ) -> Answer:
        return self.pipeline.answer(
            question, filters, k=k or self.top_n, min_score=NO_RETRIEVAL_GATE,
        )

    def close(self) -> None:
        self.pipeline.close()
