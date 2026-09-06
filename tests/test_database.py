"""Tests for the SQLite store."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.ingestion.normalizer import normalize_export
from src.ingestion.whatsapp_parser import parse_export
from src.storage.database import Database

SAMPLE = """\
25/07/2026, 20:02 - Rahul: earlier
25/07/2026, 20:03 - Me: hi
26/07/2026, 09:00 - Rahul: about the microsoft internship
27/07/2026, 10:00 - Me: system test
"""


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture()
def loaded(db):
    conv, msgs = normalize_export(parse_export(SAMPLE), conversation_name="Rahul", me_names=["Me"])
    db.upsert_conversation(conv, me_names=["Me"])
    added = db.insert_messages(msgs)
    return db, conv, msgs, added


def test_insert_counts_and_dedup(loaded):
    db, conv, msgs, added = loaded
    assert added == len(msgs)
    # re-inserting the same messages adds nothing
    assert db.insert_messages(msgs) == 0


def test_stats(loaded):
    db, *_ = loaded
    s = db.stats()
    assert s["messages"] == 4
    assert s["conversations"] == 1
    assert "Rahul" in s["participants"] and "Me" in s["participants"]
    assert s["date_range"][0].startswith("2026-07-25")


def test_get_messages_by_sender(loaded):
    db, *_ = loaded
    rows = db.get_messages(sender="rahul")
    assert {r["text"] for r in rows} == {"earlier", "about the microsoft internship"}


def test_get_messages_by_date_range(loaded):
    db, *_ = loaded
    rows = db.get_messages(start=datetime(2026, 7, 26), end=datetime(2026, 7, 26, 23, 59))
    assert [r["text"] for r in rows] == ["about the microsoft internship"]


def test_get_messages_contains(loaded):
    db, *_ = loaded
    rows = db.get_messages(contains="microsoft")
    assert len(rows) == 1


def test_get_messages_by_ids_preserves_order(loaded):
    db, conv, msgs, _ = loaded
    ids = [msgs[2].message_id, msgs[0].message_id]
    rows = db.get_messages_by_ids(ids)
    assert [r["message_id"] for r in rows] == ids


def test_conversation_window(loaded):
    db, conv, msgs, _ = loaded
    rows = db.conversation_window(conv.conversation_id, 1, 2)
    assert [r["seq"] for r in rows] == [1, 2]


def test_reopen_database_keeps_data(loaded, tmp_path):
    db, *_ = loaded
    db.close()
    again = Database(tmp_path / "test.db")
    assert again.stats()["messages"] == 4
    again.close()
