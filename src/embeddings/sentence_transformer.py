"""SentenceTransformers embedder (Phase 1 default: all-MiniLM-L6-v2).

The model is loaded lazily on first use so importing this module (and the app)
stays cheap and does not require torch to be importable until embeddings are
actually needed.
"""

from __future__ import annotations

import numpy as np

from src.embeddings.base import EmbeddingService, l2_normalize

_DIM_HINT = {"sentence-transformers/all-MiniLM-L6-v2": 384}


class SentenceTransformerEmbedder(EmbeddingService):
    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self.dim = _DIM_HINT.get(model_id, 0)

    def warmup(self) -> None:
        _ = self.model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 384), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves for consistency
            show_progress_bar=False,
        )
        return l2_normalize(vecs)
