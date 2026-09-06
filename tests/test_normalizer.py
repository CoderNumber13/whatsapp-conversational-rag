"""Tests for normalization: identity, ids, ordering, grouping, dedup."""

from __future__ import annotations

from src.ingestion.normalizer import (
    WhatsAppExportSource,
    derive_conversation_id,
    normalize_export,
)
from src.ingestion.whatsapp_parser import parse_export

ONE_TO_ONE = """\
25/07/2026, 20:03 - Me: hi
25/07/2026, 20:02 - Rahul: earlier message
25/07/2026, 20:04 - Rahul: ok
"""

GROUP = """\
01/08/2026, 21:00 - Priya created group "College Friends"
01/08/2026, 21:01 - Priya: hey
01/08/2026, 21:02 - Rahul Kumar: yo
01/08/2026, 21:03 - Me: hello
"""


def _norm(text, name="Rahul", me=("Me",)):
    return normalize_export(parse_export(text), conversation_name=name, me_names=me)


def test_conversation_id_is_deterministic_and_name_insensitive():
    a = derive_conversation_id("whatsapp", "College Friends")
    b = derive_conversation_id("whatsapp", "  college   friends ")
    assert a == b


def test_messages_sorted_by_timestamp_and_seq_assigned():
    _, msgs = _norm(ONE_TO_ONE)
    assert [m.text for m in msgs] == ["earlier message", "hi", "ok"]
    assert [m.seq for m in msgs] == [0, 1, 2]


def test_is_from_me_case_insensitive():
    _, msgs = _norm("25/07/2026, 20:03 - me: hi\n25/07/2026, 20:04 - Rahul: yo", me=("Me",))
    assert msgs[0].is_from_me is True
    assert msgs[1].is_from_me is False


def test_one_to_one_not_group():
    conv, _ = _norm(ONE_TO_ONE)
    assert conv.is_group is False
    assert conv.participants == ["Me", "Rahul"]


def test_group_detected_by_participant_count():
    conv, msgs = normalize_export(
        parse_export(GROUP), conversation_name="College Friends", me_names=["Me"]
    )
    assert conv.is_group is True
    assert set(conv.participants) == {"Me", "Priya", "Rahul Kumar"}
    assert conv.message_count == len(msgs)


def test_exact_duplicate_messages_are_deduped():
    conv, msgs = _norm(ONE_TO_ONE + "25/07/2026, 20:02 - Rahul: earlier message\n")
    # the repeated (ts, sender, text) collapses to one row
    assert sum(m.text == "earlier message" for m in msgs) == 1


def test_message_id_stable_across_runs():
    _, a = _norm(ONE_TO_ONE)
    _, b = _norm(ONE_TO_ONE)
    assert [m.message_id for m in a] == [m.message_id for m in b]


def test_provenance_fields_populated():
    _, msgs = _norm(ONE_TO_ONE)
    assert all(m.src_line_start > 0 for m in msgs)


def test_source_ingests_synthetic_files(tmp_path):
    f = tmp_path / "WhatsApp Chat with Rahul.txt"
    f.write_text(ONE_TO_ONE, encoding="utf-8")
    src = WhatsAppExportSource([f], me_names=["Me"])
    src.ingest()
    assert len(src.get_messages()) == 3
    (conv,) = src.get_conversations()
    assert conv.name == "Rahul"
    assert conv.conversation_id == derive_conversation_id("whatsapp", "Rahul")
