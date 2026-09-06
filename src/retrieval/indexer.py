"""Keep the FAISS index in step with the ``chunks`` table.

Staleness is decided per chunk from ``embedding_meta``: a chunk is (re)embedded
when it is new, when its ``content_hash`` changed, or when the embedding model
changed. Chunks that vanished have their vector and metadata removed. A chunk
keeps the same ``faiss_id`` for life, so re-embedding is remove-then-add on one
id.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import CONFIG, Config
from src.embeddings.base import EmbeddingService
from src.embeddings.factory import get_embedder
from src.storage.database import Database
from src.retrieval.vector_store import FaissVectorStore


@dataclass
class IndexSyncResult:
    embedded: int = 0
    reembedded: int = 0
    removed: int = 0
    unchanged: int = 0

    @property
    def changed_total(self) -> int:
        return self.embedded + self.reembedded + self.removed


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class EmbeddingIndexer:
    def __init__(
        self,
        db: Database,
        embedder: EmbeddingService | None = None,
        store: FaissVectorStore | None = None,
        config: Config = CONFIG,
    ) -> None:
        self.db = db
        self.config = config
        self.embedder = embedder or get_embedder(config)
        if getattr(self.embedder, "dim", 0) <= 0 and hasattr(self.embedder, "warmup"):
            self.embedder.warmup()
        self.store = store or FaissVectorStore(
            dim=self.embedder.dim, path=config.index_dir, model_id=self.embedder.model_id
        ).load_or_create()

    def sync(self) -> IndexSyncResult:
        chunks = self.db.all_chunks()
        desired = {c.chunk_id: c for c in chunks}
        meta = self.db.embedding_meta_map()  # chunk_id -> (hash, model, faiss_id)

        removed = [cid for cid in meta if cid not in desired]
        self.store.remove([meta[cid][2] for cid in removed])
        for cid in removed:
            self.db.delete_embedding_meta(cid, commit=False)

        stale = [
            c
            for cid, c in desired.items()
            if cid not in meta
            or meta[cid][0] != c.content_hash
            or meta[cid][1] != self.embedder.model_id
        ]
        res = IndexSyncResult(
            removed=len(removed), unchanged=len(desired) - len(stale)
        )

        next_fid = self.db.max_faiss_id() + 1
        for batch in _batched(stale, self.config.embedding_batch_size):
            vectors = self.embedder.embed_texts([c.text for c in batch])
            ids: list[int] = []
            reembed_ids: list[int] = []
            for c in batch:
                if c.chunk_id in meta:
                    fid = meta[c.chunk_id][2]
                    reembed_ids.append(fid)
                    res.reembedded += 1
                else:
                    fid = next_fid
                    next_fid += 1
                    res.embedded += 1
                ids.append(fid)
            self.store.remove(reembed_ids)
            self.store.add(ids, vectors)
            for c, fid in zip(batch, ids):
                self.db.upsert_embedding_meta(
                    c.chunk_id, c.content_hash, self.embedder.model_id,
                    self.embedder.dim, fid, commit=False,
                )

        self.db.commit()
        self.store.save()
        return res

    def rebuild(self) -> IndexSyncResult:
        self.store.reset()
        self.db.clear_embedding_meta()
        return self.sync()
