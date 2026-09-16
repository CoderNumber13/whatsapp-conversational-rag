"""FAISS store + EmbeddingIndexer + VectorSearch (mock embedder)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")

from src.chunking.base import EMBED_TEXT_VERSION, render_embedding_text
from src.chunking.message_chunker import FixedCountChunker
from src.chunking.service import ChunkingService
from src.config import Config
from src.embeddings.mock import MockEmbedder
from src.ingestion.normalizer import WhatsAppExportSource
from src.retrieval.indexer import EmbeddingIndexer
from src.retrieval.vector_search import VectorSearch
from src.retrieval.vector_store import FaissVectorStore
from src.storage.database import Database
from scripts.generate_synthetic_chats import write_all


# --- store ------------------------------------------------------------

def test_store_add_search_remove(tmp_path):
    store = FaissVectorStore(dim=4, path=tmp_path, model_id="mock-4").load_or_create()
    vecs = np.eye(4, dtype=np.float32)
    store.add([10, 11, 12, 13], vecs)
    hits = store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=2)
    assert hits[0][0] == 10 and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    store.remove([10])
    assert 10 not in [i for i, _ in store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=4)]
    assert len(store) == 3


def test_store_save_load_roundtrip(tmp_path):
    s1 = FaissVectorStore(4, tmp_path, "mock-4").load_or_create()
    s1.add([1, 2], np.eye(4, dtype=np.float32)[:2])
    s1.save()
    s2 = FaissVectorStore(4, tmp_path, "mock-4").load_or_create()
    assert len(s2) == 2


def test_store_model_mismatch_starts_fresh(tmp_path):
    s1 = FaissVectorStore(4, tmp_path, "mock-4").load_or_create()
    s1.add([1], np.eye(4, dtype=np.float32)[:1])
    s1.save()
    s2 = FaissVectorStore(4, tmp_path, "other-model").load_or_create()
    assert len(s2) == 0


# --- indexer + search ------------------------------------------------

@pytest.fixture()
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    cfg = Config.reload()

    files = write_all(tmp_path / "chats")
    src = WhatsAppExportSource(files, me_names=["Me"])
    src.ingest()
    db = Database(tmp_path / "cm.db")
    for c in src.get_conversations():
        db.upsert_conversation(c, me_names=["Me"])
    db.insert_messages(src.get_messages())
    ChunkingService(db, FixedCountChunker(8, 2, 3)).sync_all()

    indexer = EmbeddingIndexer(db, embedder=MockEmbedder(64), config=cfg)
    yield db, cfg, indexer
    db.close()


def test_first_sync_embeds_all_chunks(wired):
    db, cfg, indexer = wired
    res = indexer.sync()
    n = db.count_chunks()
    assert res.embedded == n and res.reembedded == 0 and res.removed == 0
    assert len(indexer.store) == n
    assert len(db.embedding_meta_map()) == n


def test_second_sync_is_noop(wired):
    db, cfg, indexer = wired
    indexer.sync()
    res = indexer.sync()
    assert res.changed_total == 0
    assert res.unchanged == db.count_chunks()


def test_edited_chunk_is_reembedded_with_stable_faiss_id(wired):
    db, cfg, indexer = wired
    indexer.sync()
    meta_before = db.embedding_meta_map()
    cid = max(db.conversation_ids(), key=lambda c: len(db.get_conversation_messages(c)))
    victim = db.get_conversation_messages(cid)[9]
    db.conn.execute(
        "UPDATE messages SET text = text || ' (edited)' WHERE message_id = ?", (victim.message_id,)
    )
    db.commit()
    ChunkingService(db, FixedCountChunker(8, 2, 3)).sync_conversation(cid)

    res = indexer.sync()
    assert res.reembedded >= 1 and res.embedded == 0 and res.removed == 0
    assert len(indexer.store) == db.count_chunks()
    # ids of surviving chunks are unchanged
    meta_after = db.embedding_meta_map()
    for k in meta_before.keys() & meta_after.keys():
        assert meta_before[k][2] == meta_after[k][2]


def test_rebuild_reproduces_index(wired):
    db, cfg, indexer = wired
    indexer.sync()
    n = len(indexer.store)
    res = indexer.rebuild()
    assert res.embedded == db.count_chunks()
    assert len(indexer.store) == n


def test_vector_search_finds_relevant_chunk(wired):
    db, cfg, indexer = wired
    indexer.sync()
    vs = VectorSearch(db, indexer.embedder, indexer.store)
    # MockEmbedder isn't semantic, so query with text taken from a real chunk.
    # Chunks are indexed from their embedding view (no [m:] tags / line
    # timestamps), so that is the view an exact self-match must be built from.
    target = db.all_chunks()[0]
    hits = vs.search(render_embedding_text(target.text), k=3)
    assert hits and hits[0].chunk.chunk_id == target.chunk_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_vector_search_candidate_filter(wired):
    db, cfg, indexer = wired
    indexer.sync()
    vs = VectorSearch(db, indexer.embedder, indexer.store)
    allowed = {db.all_chunks()[1].chunk_id}
    hits = vs.search(db.all_chunks()[0].text, k=5, candidate_chunk_ids=allowed)
    assert {h.chunk.chunk_id for h in hits} <= allowed


# --- regression: what actually gets embedded --------------------------
# The "gmail password" investigation found chunks were embedded from their
# DISPLAY text: every line carried an ISO timestamp and an [m:<id>] citation
# tag. Those were ~60% of the tokens and pushed 13 of 15 real chunks past
# all-MiniLM-L6-v2's 256-token input limit, where the tail is silently dropped
# — one gmail chunk lost the word "gmail" itself and so could not be found by
# the word it contained. Only the embedding input changed; chunk.text still
# goes to the prompt verbatim so citations are unaffected.

def test_embedding_view_strips_tags_and_timestamps_but_keeps_content():
    raw = (
        "Conversation: chat (direct) | Participants: A, You\n"
        "Dates: 2026-08-21 13:34 - 2026-08-21 14:30\n"
        "[2026-08-21 13:34] A: can you set up a new gmail  [m:48a93cb88e43aaaa]\n"
        "[2026-08-21 14:30] You: mail bhej diya  [m:e8586b63d1b6bbbb]"
    )
    view = render_embedding_text(raw)

    assert "[m:" not in view, "citation tags must not be embedded"
    assert "2026-08-21 13:34]" not in view, "per-line timestamps must not be embedded"
    # content and speakers survive; the chunk's own date header survives
    assert "can you set up a new gmail" in view
    assert "mail bhej diya" in view
    assert "A:" in view and "You:" in view
    assert "Dates: 2026-08-21 13:34 - 2026-08-21 14:30" in view
    assert len(view) < len(raw)


def test_embedding_view_is_substantially_shorter_than_display_text(wired):
    """The dilution the fix removes: display text is mostly scaffolding."""
    db, _cfg, _indexer = wired
    for chunk in db.all_chunks():
        view = render_embedding_text(chunk.text)
        assert len(view) < len(chunk.text) * 0.8, (
            f"chunk {chunk.chunk_id} barely shrank — tags/timestamps not stripped"
        )


def test_indexer_embeds_the_view_not_the_display_text(wired):
    """The fix at the point it matters: no [m:] tag ever reaches the embedder."""
    db, cfg, indexer = wired
    seen: list[str] = []
    inner = indexer.embedder.embed_texts

    def spy(texts):
        seen.extend(texts)
        return inner(texts)

    indexer.embedder.embed_texts = spy
    indexer.rebuild()

    assert seen, "nothing was embedded"
    assert not any("[m:" in t for t in seen), "citation tags leaked into the embedder"
    assert any("Dates:" in t for t in seen), "chunk header should still be embedded"


def test_embedding_recipe_version_forces_reembed(wired):
    """Changing the embedding input must invalidate stored vectors, or upgrades
    silently keep vectors built from the old recipe."""
    db, cfg, indexer = wired
    indexer.sync()
    assert indexer.sync().changed_total == 0, "second sync should be a no-op"

    # simulate the recipe changing under an existing index
    stale = {cid: (h, "old-model#et1", fid)
             for cid, (h, _m, fid) in db.embedding_meta_map().items()}
    for cid, (h, m, fid) in stale.items():
        db.upsert_embedding_meta(cid, h, m, indexer.embedder.dim, fid, commit=False)
    db.commit()

    assert indexer.sync().reembedded == db.count_chunks()


def test_embedding_key_includes_recipe_version(wired):
    db, cfg, indexer = wired
    assert indexer.embedding_key.startswith(indexer.embedder.model_id)
    assert f"#et{EMBED_TEXT_VERSION}" in indexer.embedding_key
