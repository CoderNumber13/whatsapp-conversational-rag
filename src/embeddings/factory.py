from __future__ import annotations

from src.config import CONFIG, Config
from src.embeddings.base import EmbeddingService


def get_embedder(config: Config = CONFIG) -> EmbeddingService:
    """`EMBEDDING_MODEL=mock` (optionally `mock-<dim>`) skips the model download."""
    model = config.embedding_model
    if model == "mock" or model.startswith("mock-"):
        from src.embeddings.mock import MockEmbedder

        dim = int(model.split("-", 1)[1]) if "-" in model else 64
        return MockEmbedder(dim=dim)

    from src.embeddings.sentence_transformer import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        model_id=model,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
    )
