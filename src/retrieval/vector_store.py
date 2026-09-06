"""FAISS wrapper — the only place FAISS is touched.

Inner-product index over L2-normalized vectors == cosine similarity. Vectors
carry our own int64 ids (from ``embedding_meta.faiss_id``); mapping an id back
to a chunk is the DB's job. The index is derived data: it can always be
rebuilt from ``chunks`` + the embedder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissVectorStore:
    def __init__(self, dim: int, path: str | Path, model_id: str) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be known before creating the store")
        self.dim = dim
        self.path = Path(path)
        self.model_id = model_id
        self.index = None  # set by load_or_create()

    # --- lifecycle -------------------------------------------------
    def _new_index(self):
        import faiss

        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    @property
    def _index_file(self) -> Path:
        return self.path / "vectors.faiss"

    @property
    def _meta_file(self) -> Path:
        return self.path / "meta.json"

    def load_or_create(self) -> "FaissVectorStore":
        import faiss

        if self._index_file.exists() and self._meta_file.exists():
            meta = json.loads(self._meta_file.read_text(encoding="utf-8"))
            if meta.get("model_id") == self.model_id and meta.get("dim") == self.dim:
                self.index = faiss.read_index(str(self._index_file))
                return self
        self.index = self._new_index()
        return self

    def reset(self) -> None:
        self.index = self._new_index()

    def save(self) -> None:
        import faiss

        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_file))
        self._meta_file.write_text(
            json.dumps({"model_id": self.model_id, "dim": self.dim}), encoding="utf-8"
        )

    # --- writes -------------------------------------------------
    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids:
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))

    def remove(self, ids: list[int]) -> None:
        if ids:
            self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    # --- reads --------------------------------------------------
    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index is None or self.index.ntotal == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def __len__(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)
