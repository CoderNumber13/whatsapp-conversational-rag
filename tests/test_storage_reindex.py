"""D0 invariants: id stability, seq recomputation and idempotency under re-import."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.ingestion.normalizer import normalize_export
from src.ingestion.whatsapp_parser import parse_export
from src.storage.database import Database
from src.storage.models import Chunk

FIRST_EXPORT = """\
10/08/2026, 09:00 - Rahul: message four
10/08/2026, 09:01 - Me: message five
10/08/2026, 09:02 - Rahul: message six
"""

# same chat re-exported later, now including three older messages at the top
SECOND_EXPORT = """\
09/08/2026, 21:00 - Rahul: message one
09/08/2026, 21:01 - Me: message two
09/08/2026, 21:02 - Rahul: message three
10/08/2026, 09:00 - Rahul: message four
10/08/2026, 09:01 - Me: message five
10/08/2026, 09:02 - Rahul: message six
"""


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    yield d
    d.close()


def _ingest(db, text):
    conv, msgs = normalize_export(parse_export(text), conversation_name="Rahul", me_names=["Me"])
    db.upsert_conversation(conv, me_names=["Me"])
    return db.insert_messages(msgs), conv


def test_reimport_with_prepended_history_keeps_seq_dense_and_ordered(db):
    _ingest(db, FIRST_EXPORT)
    added, conv = _ingest(db, SECOND_EXPORT)
    assert added == 3  # only the 3 new older messages

    msgs = db.get_conversation_messages(conv.conversation_id)
    assert [m.text for m in msgs] == [
        "message one", "message two", "message three",
        "message four", "message five", "message six",
    ]
    assert [m.seq for m in msgs] == [0, 1, 2, 3, 4, 5]  # dense, no collisions


def test_identical_reexport_is_idempotent(db):
    _ingest(db, SECOND_EXPORT)
    cid = db.conversation_ids()[0]
    before = [(m.message_id, m.seq) for m in db.get_conversation_messages(cid)]
    added, _ = _ingest(db, SECOND_EXPORT)
    assert added == 0
    after = [(m.message_id, m.seq) for m in db.get_conversation_messages(cid)]
    assert before == after


def test_genuine_repeats_survive_db_insert(db):
    text = "10/08/2026, 09:00 - Rahul: ok\n10/08/2026, 09:00 - Rahul: ok\n"
    added, conv = _ingest(db, text)
    assert added == 2
    assert len(db.get_conversation_messages(conv.conversation_id)) == 2


def test_conversation_stats_recomputed_after_merge(db):
    _ingest(db, FIRST_EXPORT)
    _, conv = _ingest(db, SECOND_EXPORT)
    row = db.get_conversation(conv.conversation_id)
    assert row["message_count"] == 6
    assert row["first_ts"].startswith("2026-08-09")
    assert row["last_ts"].startswith("2026-08-10")


def test_get_conversation_messages_is_typed_and_excludes_system_by_default(db):
    _ingest(
        db,
        "10/08/2026, 09:00 - Messages and calls are end-to-end encrypted. Tap to learn more.\n"
        "10/08/2026, 09:01 - Rahul: hi\n",
    )
    cid = db.conversation_ids()[0]
    msgs = db.get_conversation_messages(cid)
    assert [m.text for m in msgs] == ["hi"]
    assert isinstance(msgs[0].timestamp, datetime)
    assert db.get_conversation_messages(cid, include_system=True)[0].is_system is True


def test_context_window_before_after(db):
    _ingest(db, SECOND_EXPORT)
    cid = db.conversation_ids()[0]
    win = db.context_window(cid, 3, 3, before=2, after=1)
    assert [m.seq for m in win] == [1, 2, 3, 4]


def test_insert_chunk_populates_participants_and_delete_cleans_up(db):
    _ingest(db, SECOND_EXPORT)
    cid = db.conversation_ids()[0]
    msgs = db.get_conversation_messages(cid)
    chunk = Chunk(
        chunk_id="c1",
        conversation_id=cid,
        seq_start=msgs[0].seq,
        seq_end=msgs[2].seq,
        ts_start=msgs[0].timestamp,
        ts_end=msgs[2].timestamp,
        participants=["Me", "Rahul"],
        message_ids=[m.message_id for m in msgs[:3]],
        text="…",
    )
    db.insert_chunk(chunk)
    parts = db.conn.execute(
        "SELECT display_name FROM chunk_participants WHERE chunk_id='c1' ORDER BY display_name"
    ).fetchall()
    assert [p["display_name"] for p in parts] == ["Me", "Rahul"]

    db.delete_chunk("c1")
    for table in ("chunks", "chunk_messages", "chunk_participants"):
        assert db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE chunk_id='c1'"
        ).fetchone()[0] == 0
