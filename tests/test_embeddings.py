"""Embedding service tests (mock embedder — no model download)."""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.base import l2_normalize
from src.embeddings.factory import get_embedder
from src.embeddings.mock import MockEmbedder
from src.config import Config


def test_mock_embedder_shape_dtype_and_norm():
    emb = MockEmbedder(dim=32)
    v = emb.embed_texts(["hello", "world", "again"])
    assert v.shape == (3, 32)
    assert v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)


def test_mock_embedder_is_deterministic():
    a = MockEmbedder(16).embed_texts(["x", "y"])
    b = MockEmbedder(16).embed_texts(["x", "y"])
    np.testing.assert_array_equal(a, b)


def test_embed_query_returns_1d():
    q = MockEmbedder(16).embed_query("hi")
    assert q.shape == (16,)


def test_empty_input():
    assert MockEmbedder(16).embed_texts([]).shape == (0, 16)


def test_l2_normalize_handles_zero_vector():
    out = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "model,expected_dim", [("mock", 64), ("mock-128", 128)]
)
def test_factory_selects_mock(monkeypatch, model, expected_dim):
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    emb = get_embedder(Config.reload())
    assert isinstance(emb, MockEmbedder)
    assert emb.dim == expected_dim
