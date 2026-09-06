"""Retriever: metadata filtering + context expansion (mock embedder)."""

from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("faiss")

from src.chunking.message_chunker import FixedCountChunker
from src.chunking.service import ChunkingService
from src.config import Config
from src.embeddings.mock import MockEmbedder
from src.ingestion.normalizer import WhatsAppExportSource
from src.retrieval.indexer import EmbeddingIndexer
from src.retrieval.retriever import RetrievalFilters, Retriever
from src.retrieval.vector_search import VectorSearch
from src.storage.database import Database
from scripts.generate_synthetic_chats import write_all


@pytest.fixture()
def retriever(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("CONTEXT_WINDOW_MESSAGES", "2")
    cfg = Config.reload()

    files = write_all(tmp_path / "chats")
    src = WhatsAppExportSource(files, me_names=["Me"])
    src.ingest()
    db = Database(tmp_path / "cm.db")
    for c in src.get_conversations():
        db.upsert_conversation(c, me_names=["Me"])
    db.insert_messages(src.get_messages())
    ChunkingService(db, FixedCountChunker(6, 2, 2)).sync_all()
    ix = EmbeddingIndexer(db, embedder=MockEmbedder(64), config=cfg)
    ix.sync()
    r = Retriever(db, VectorSearch(db, ix.embedder, ix.store), cfg)
    yield db, cfg, r
    db.close()


def _conv_id(db, name):
    return next(c["conversation_id"] for c in db.list_conversations() if c["name"] == name)


def test_no_filter_searches_everything(retriever):
    db, cfg, r = retriever
    hits = r.retrieve("placements and internships", k=5)
    assert hits
    assert len({h.chunk.conversation_id for h in hits}) >= 1


def test_conversation_filter_restricts_results(retriever):
    db, cfg, r = retriever
    cid = _conv_id(db, "DS Project Team")
    hits = r.retrieve("backend and frontend", k=5, filters=RetrievalFilters(conversation_id=cid))
    assert hits
    assert all(h.chunk.conversation_id == cid for h in hits)


def test_sender_filter(retriever):
    db, cfg, r = retriever
    hits = r.retrieve("anything", k=8, filters=RetrievalFilters(sender="sneha"))
    assert hits
    for h in hits:
        rows = db.get_messages_by_ids(h.chunk.message_ids)
        assert any(row["sender"].lower() == "sneha" for row in rows)


def test_date_range_filter(retriever):
    db, cfg, r = retriever
    hits = r.retrieve(
        "internship",
        k=8,
        filters=RetrievalFilters(
            date_from=datetime(2026, 7, 1), date_to=datetime(2026, 7, 31, 23, 59)
        ),
    )
    for h in hits:
        assert h.chunk.ts_start <= datetime(2026, 7, 31, 23, 59)
        assert h.chunk.ts_end >= datetime(2026, 7, 1)


def test_is_group_filter(retriever):
    db, cfg, r = retriever
    hits = r.retrieve("hi", k=10, filters=RetrievalFilters(is_group=False))
    assert hits
    assert all(h.conversation["is_group"] is False for h in hits)


def test_impossible_filter_returns_empty(retriever):
    db, cfg, r = retriever
    hits = r.retrieve("x", k=5, filters=RetrievalFilters(sender="nobody-here"))
    assert hits == []


def test_context_expansion_adds_surrounding_messages(retriever):
    db, cfg, r = retriever
    cid = _conv_id(db, "Rahul")
    hits = r.retrieve("microsoft internship offer", k=3, filters=RetrievalFilters(conversation_id=cid))
    assert hits
    h = hits[0]
    member_seqs = {m.seq for m in h.messages}
    ctx = h.context_before + h.context_after
    assert all(m.seq not in member_seqs for m in ctx)
    ordered = h.ordered_messages()
    assert [m.seq for m in ordered] == sorted(m.seq for m in ordered)
    assert len({m.message_id for m in ordered}) == len(ordered)  # de-duped
