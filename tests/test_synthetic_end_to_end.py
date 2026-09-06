"""Increment-1 smoke test: parse every synthetic export, normalize, load into
SQLite, and assert the high-level facts the demo will rely on."""

from __future__ import annotations

from src.ingestion.normalizer import WhatsAppExportSource
from src.storage.database import Database
from scripts.generate_synthetic_chats import write_all

ME = ["Me"]


def _load(tmp_path):
    chat_dir = tmp_path / "chats"
    files = write_all(chat_dir)
    src = WhatsAppExportSource(files, me_names=ME)
    src.ingest()
    db = Database(tmp_path / "cm.db")
    for conv in src.get_conversations():
        db.upsert_conversation(conv, me_names=ME)
    db.insert_messages(src.get_messages())
    return db, src


def test_all_synthetic_files_ingest(tmp_path):
    db, src = _load(tmp_path)
    stats = db.stats()
    assert stats["conversations"] == 4
    assert stats["messages"] > 40
    lo, hi = stats["date_range"]
    assert lo.startswith("2026-07") and hi.startswith("2026-08")
    db.close()


def test_group_vs_direct_classification(tmp_path):
    db, src = _load(tmp_path)
    by_name = {c.name: c for c in src.get_conversations()}
    assert by_name["Rahul"].is_group is False
    assert by_name["Amma"].is_group is False
    assert by_name["College Friends"].is_group is True
    assert by_name["DS Project Team"].is_group is True
    db.close()


def test_known_facts_are_queryable(tmp_path):
    db, src = _load(tmp_path)
    # the Microsoft internship offer is retrievable by keyword
    hits = db.get_messages(contains="Microsoft internship offer")
    assert any("2 September" in h["text"] for h in db.get_messages(sender="Rahul"))
    assert len(hits) == 1
    # a promise the user made
    mine = db.get_messages(sender="Me", contains="promise")
    assert len(mine) >= 2
    db.close()


def test_media_and_system_lines_are_flagged(tmp_path):
    db, src = _load(tmp_path)
    msgs = src.get_messages()
    assert any(m.media_type == "media" for m in msgs)      # Android <Media omitted>
    assert any(m.media_type == "image" for m in msgs)      # iOS "image omitted"
    assert any(m.is_system for m in msgs)                  # e2e / group-create
    assert any(m.deleted for m in msgs)                    # "This message was deleted"
    db.close()


def test_ios_narrow_nbsp_timestamps_parsed(tmp_path):
    db, src = _load(tmp_path)
    cf = next(c for c in src.get_conversations() if c.name == "College Friends")
    assert cf.first_ts is not None and cf.first_ts.year == 2026
    db.close()
