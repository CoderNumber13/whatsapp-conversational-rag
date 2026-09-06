"""Tests for conversation-aware chunking and chunk/DB sync."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.chunking.base import CHUNKER_VERSION, render_chunk_text
from src.chunking.message_chunker import FixedCountChunker
from src.chunking.service import ChunkingService
from src.chunking.time_chunker import TimeWindowChunker
from src.ingestion.normalizer import WhatsAppExportSource
from src.storage.database import Database
from src.storage.models import Message

BASE = datetime(2026, 8, 1, 9, 0)


def _messages(n: int, conv="c", start_seq=0) -> list[Message]:
    out = []
    for i in range(n):
        out.append(
            Message(
                conversation_id=conv,
                conversation_name="Rahul",
                timestamp=BASE + timedelta(minutes=i),
                sender="Me" if i % 2 else "Rahul",
                text=f"message {i}",
                seq=start_seq + i,
            )
        )
    return out


# --- span maths -------------------------------------------------------

@pytest.mark.parametrize(
    "n,size,overlap,expected",
    [
        (12, 12, 3, [(0, 12)]),
        (21, 12, 3, [(0, 12), (9, 21)]),
        (24, 12, 3, [(0, 12), (9, 21), (18, 24)]),
        (10, 12, 3, [(0, 10)]),
        (3, 12, 3, [(0, 3)]),
    ],
)
def test_spans(n, size, overlap, expected):
    assert FixedCountChunker(size, overlap, min_messages=4)._spans(n) == expected


def test_undersized_tail_is_folded_into_predecessor():
    # size 10, step 8: [0,10),[8,18),[16,19) -> last has 3 (< min 4) -> folded
    spans = FixedCountChunker(10, 2, min_messages=4)._spans(19)
    assert spans == [(0, 10), (8, 19)]


def test_chunk_preserves_order_and_overlap():
    chunks = FixedCountChunker(12, 3, 4).chunk("c", "Rahul", False, _messages(21))
    assert len(chunks) == 2
    a, b = chunks
    assert a.message_ids == [m.message_id for m in _messages(21)[0:12]]
    assert b.message_ids == [m.message_id for m in _messages(21)[9:21]]
    assert a.message_ids[9:12] == b.message_ids[0:3]  # 3-message overlap


def test_chunk_id_and_hash_are_deterministic():
    c1 = FixedCountChunker(12, 3, 4).chunk("c", "Rahul", False, _messages(21))
    c2 = FixedCountChunker(12, 3, 4).chunk("c", "Rahul", False, _messages(21))
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
    assert [c.content_hash for c in c1] == [c.content_hash for c in c2]


def test_chunk_id_is_independent_of_absolute_seq():
    a = FixedCountChunker(12, 3, 4).chunk("c", "Rahul", False, _messages(12, start_seq=0))
    b = FixedCountChunker(12, 3, 4).chunk("c", "Rahul", False, _messages(12, start_seq=1000))
    assert a[0].chunk_id == b[0].chunk_id  # same messages, shifted seq -> same id
    assert a[0].seq_start == 0 and b[0].seq_start == 1000


def test_rendered_text_carries_every_message_id():
    (chunk,) = FixedCountChunker(12, 3, 4).chunk("c", "Rahul", False, _messages(8))
    for m in _messages(8):
        assert f"[m:{m.message_id}]" in chunk.text
    assert chunk.text.startswith("Conversation: Rahul (direct)")


def test_media_only_window_is_skipped():
    msgs = _messages(6)
    for m in msgs:
        m.text = ""
        m.media_type = "image"
    assert FixedCountChunker(6, 0, 1).chunk("c", "Rahul", False, msgs) == []


def test_time_window_chunker_splits_on_gap():
    msgs = _messages(4)
    msgs[2].timestamp = msgs[1].timestamp + timedelta(hours=2)  # big gap before #2
    msgs[3].timestamp = msgs[2].timestamp + timedelta(minutes=1)
    chunks = TimeWindowChunker(window_minutes=45).chunk("c", "Rahul", False, msgs)
    assert [len(c.message_ids) for c in chunks] == [2, 2]


# --- sync against the DB -------------------------------------------

@pytest.fixture()
def db_with_chats(tmp_path):
    from scripts.generate_synthetic_chats import write_all

    files = write_all(tmp_path / "chats")
    src = WhatsAppExportSource(files, me_names=["Me"])
    src.ingest()
    db = Database(tmp_path / "cm.db")
    for conv in src.get_conversations():
        db.upsert_conversation(conv, me_names=["Me"])
    db.insert_messages(src.get_messages())
    yield db
    db.close()


def test_sync_is_idempotent(db_with_chats):
    svc = ChunkingService(db_with_chats, FixedCountChunker(8, 2, 3))
    first = svc.sync_all()
    assert first.added > 0 and first.updated == 0 and first.deleted == 0
    second = svc.sync_all()
    assert second.added == 0 and second.updated == 0 and second.deleted == 0
    assert second.unchanged == first.added


def test_editing_one_message_only_rehashes_overlapping_chunks(db_with_chats):
    svc = ChunkingService(db_with_chats, FixedCountChunker(8, 2, 3))
    svc.sync_all()
    before = db_with_chats.count_chunks()
    cid = max(
        db_with_chats.conversation_ids(),
        key=lambda c: len(db_with_chats.get_conversation_messages(c)),
    )
    victim = db_with_chats.get_conversation_messages(cid)[10]
    db_with_chats.conn.execute(
        "UPDATE messages SET text = text || ' (edited)' WHERE message_id = ?",
        (victim.message_id,),
    )
    db_with_chats.commit()
    res = svc.sync_conversation(cid)
    assert res.added == 0 and res.deleted == 0
    assert 1 <= res.updated <= 2  # message 10 lands in at most two overlapping windows
    assert db_with_chats.count_chunks() == before


def test_every_chunk_message_id_resolves(db_with_chats):
    svc = ChunkingService(db_with_chats, FixedCountChunker(8, 2, 3))
    svc.sync_all()
    for chunk in db_with_chats.all_chunks():
        rows = db_with_chats.get_messages_by_ids(chunk.message_ids)
        assert len(rows) == len(chunk.message_ids)
        assert chunk.participants == sorted({r["sender"] for r in rows if r["sender"]})
