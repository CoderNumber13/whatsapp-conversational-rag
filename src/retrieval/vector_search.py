"""Query-time semantic search over the FAISS index.

Returns chunks with cosine scores. Optional ``candidate_chunk_ids`` restricts
results to a metadata-filtered set (the retriever computes that set in SQL);
when filtering we over-fetch from FAISS and trim afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import CONFIG, Config
from src.embeddings.base import EmbeddingService
from src.embeddings.factory import get_embedder
from src.storage.database import Database
from src.storage.models import Chunk
from src.retrieval.vector_store import FaissVectorStore


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorSearch:
    def __init__(
        self, db: Database, embedder: EmbeddingService, store: FaissVectorStore
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.store = store

    @classmethod
    def open(cls, db: Database, config: Config = CONFIG) -> "VectorSearch":
        embedder = get_embedder(config)
        if getattr(embedder, "dim", 0) <= 0:
            embedder.warmup()
        store = FaissVectorStore(
            dim=embedder.dim, path=config.index_dir, model_id=embedder.model_id
        ).load_or_create()
        return cls(db, embedder, store)

    def search(
        self,
        query: str,
        k: int,
        candidate_chunk_ids: Optional[set[str]] = None,
    ) -> list[ScoredChunk]:
        if not query.strip() or len(self.store) == 0:
            return []
        qv = self.embedder.embed_query(query)
        fetch = k if candidate_chunk_ids is None else max(k * 5, 50)
        raw = self.store.search(qv, fetch)
        id_map = self.db.chunk_ids_for_faiss([fid for fid, _ in raw])

        out: list[ScoredChunk] = []
        for fid, score in raw:
            cid = id_map.get(fid)
            if cid is None:
                continue  # index/DB drift — skip, rebuild will fix
            if candidate_chunk_ids is not None and cid not in candidate_chunk_ids:
                continue
            chunk = self.db.get_chunk(cid)
            if chunk is not None:
                out.append(ScoredChunk(chunk=chunk, score=score))
            if len(out) >= k:
                break
        return out
