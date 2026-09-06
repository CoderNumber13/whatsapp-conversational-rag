"""Embedding interface.

Swappable behind these two calls so the model can change (or be mocked in
tests) without touching chunking, retrieval or the pipeline. Implementations
must return L2-normalized float32 vectors so a FAISS inner-product index gives
cosine similarity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingService(ABC):
    model_id: str
    dim: int

    def warmup(self) -> None:
        """Force any lazy model load so ``dim`` is known. No-op by default."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalized."""

    def embed_query(self, text: str) -> np.ndarray:
        """(dim,) float32, L2-normalized."""
        return self.embed_texts([text])[0]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
