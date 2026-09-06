"""Deterministic hash-based embedder — no model download, used in tests and as
a fallback provider. Similar strings do NOT get similar vectors; it only
guarantees determinism and correct shape/norm.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.embeddings.base import EmbeddingService, l2_normalize


class MockEmbedder(EmbeddingService):
    def __init__(self, dim: int = 64) -> None:
        self.model_id = f"mock-{dim}"
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.vstack([self._one(t) for t in texts]))
