"""SentenceTransformers embedder (Phase 1 default: all-MiniLM-L6-v2).

The model is loaded lazily on first use so importing this module (and the app)
stays cheap and does not require torch to be importable until embeddings are
actually needed.
"""

from __future__ import annotations

import logging

import numpy as np

from src.embeddings.base import EmbeddingService, l2_normalize

logger = logging.getLogger(__name__)

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

    @property
    def max_input_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text))

    def _warn_if_truncated(self, texts: list[str]) -> None:
        """The model silently drops everything past ``max_seq_length``; say so.

        A truncated chunk loses the tail of the conversation from its vector, so
        it can become unretrievable by the very words it contains.
        """
        limit = self.max_input_tokens
        over = [n for n in (self.count_tokens(t) for t in texts) if n > limit]
        if over:
            logger.warning(
                "%d/%d texts exceed %s's %d-token input limit (largest %d) and "
                "were truncated before embedding; content past the limit is not "
                "searchable. Reduce CHUNK_SIZE_MESSAGES.",
                len(over), len(texts), self.model_id, limit, max(over),
            )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 384), dtype=np.float32)
        self._warn_if_truncated(texts)
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves for consistency
            show_progress_bar=False,
        )
        return l2_normalize(vecs)
