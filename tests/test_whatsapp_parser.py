"""Unit tests for the WhatsApp export parser."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.ingestion.whatsapp_parser import (
    conversation_name_from_filename,
    parse_export,
)

NBSP_NARROW = " "
LTR = "‎"


def only_messages(text, **kw):
    return parse_export(text, **kw).messages


# --- basic shapes ------------------------------------------------------

def test_android_single_line():
    (m,) = only_messages("25/08/26, 21:14 - Rahul: Bro did you apply?")
    assert m.sender == "Rahul"
    assert m.text == "Bro did you apply?"
    assert m.is_system is False
    assert m.dt == datetime(2026, 8, 25, 21, 14)


def test_ios_bracket_single_line():
    (m,) = only_messages("[25/08/2026, 21:14:07] Rahul: Bro did you apply?")
    assert m.sender == "Rahul"
    assert m.dt == datetime(2026, 8, 25, 21, 14, 7)


def test_ios_12h_with_narrow_nbsp():
    line = f"[8/1/26, 9:14:07{NBSP_NARROW}PM] Priya: hey"
    (m,) = only_messages(line, dayfirst=False)
    assert m.sender == "Priya"
    assert m.dt == datetime(2026, 8, 1, 21, 14, 7)


def test_am_pm_midnight_and_noon():
    a = only_messages("1/1/2026, 12:30 AM - X: a")[0]
    b = only_messages("1/1/2026, 12:30 PM - X: b")[0]
    assert a.dt.hour == 0 and b.dt.hour == 12


def test_two_and_four_digit_years():
    a = only_messages("25/08/26, 08:00 - X: a")[0]
    b = only_messages("25/08/2026, 08:00 - X: b")[0]
    assert a.dt.year == b.dt.year == 2026


# --- multiline / tricky bodies -------------------------------------

def test_multiline_message_merges_continuations():
    text = (
        "28/08/2026, 12:01 - Me: new plan:\n"
        "1. finish backend\n"
        "2. build UI\n"
        "28/08/2026, 12:02 - Rahul: ok"
    )
    msgs = only_messages(text)
    assert len(msgs) == 2
    assert msgs[0].text == "new plan:\n1. finish backend\n2. build UI"
    assert msgs[0].line_start == 1 and msgs[0].line_end == 3
    assert msgs[1].text == "ok"


def test_body_with_commas_and_colons():
    (m,) = only_messages("25/08/26, 21:14 - Rahul Kumar: meet at 5:30, bring the laptop: charged")
    assert m.sender == "Rahul Kumar"
    assert m.text == "meet at 5:30, bring the laptop: charged"


def test_sender_with_spaces():
    (m,) = only_messages("25/08/26, 21:14 - Aditya R Sharma: yo")
    assert m.sender == "Aditya R Sharma"


def test_emoji_and_unicode_preserved():
    (m,) = only_messages("25/08/26, 21:14 - X: nailed it 🎉💀 — ok?")
    assert "🎉💀" in m.text and "—" in m.text


def test_ltr_marks_stripped():
    (m,) = only_messages(f"25/08/26, 21:14 - X: {LTR}hello{LTR}")
    assert m.text == "hello"


# --- system / notification lines ---------------------------------

@pytest.mark.parametrize(
    "body",
    [
        "Messages and calls are end-to-end encrypted. Tap to learn more.",
        'Priya created group "College Friends"',
        "Priya added you",
        "Sneha added Rahul Kumar",
        "Aditya changed the subject to “Trip”",
        "Missed voice call",
        "You're now an admin",
    ],
)
def test_system_lines_have_no_sender(body):
    (m,) = only_messages(f"25/08/26, 21:14 - {body}")
    assert m.is_system is True
    assert m.sender is None


def test_system_line_with_name_prefix_is_still_system():
    (m,) = only_messages('25/08/26, 21:14 - Rahul pinned a message: "read this"')
    assert m.is_system is True


def test_deleted_message_is_content_not_system():
    (m,) = only_messages("25/08/26, 21:14 - Aditya: This message was deleted")
    assert m.is_system is False
    assert m.deleted is True
    assert m.sender == "Aditya"


# --- media placeholders ----------------------------------------------

@pytest.mark.parametrize(
    "body,expected",
    [
        ("<Media omitted>", "media"),
        (f"{LTR}image omitted", "image"),
        (f"{LTR}video omitted", "video"),
        (f"{LTR}audio omitted", "audio"),
        (f"{LTR}sticker omitted", "sticker"),
        (f"{LTR}Contact card omitted", "contact"),
        ("<attached: 00000042-PHOTO-2026-08-17.jpg>", "image"),
        ("IMG-20260817-WA0001.jpg (file attached)", "image"),
        ("PTT-20260817-WA0002.opus (file attached)", "audio"),
        ("STK-20260817-WA0003.webp (file attached)", "sticker"),
        ("contact.vcf (file attached)", "contact"),
        ("location: https://maps.google.com/?q=19.07,72.87", "location"),
    ],
)
def test_media_detection(body, expected):
    (m,) = only_messages(f"25/08/26, 21:14 - Rahul: {body}")
    assert m.media_type == expected
    assert m.text == ""
    assert m.is_system is False


def test_edited_suffix_flagged_and_stripped():
    (m,) = only_messages(
        f"25/08/26, 21:14 - Rahul: actually 6pm works better {LTR}<This message was edited>"
    )
    assert m.edited is True
    assert m.text == "actually 6pm works better"


# --- date-order inference -----------------------------------------

def test_dayfirst_inferred_from_day_gt_12():
    exp = parse_export("25/07/2026, 20:02 - X: a\n03/07/2026, 20:02 - X: b")
    assert exp.dayfirst is True
    assert exp.messages[0].dt == datetime(2026, 7, 25, 20, 2)


def test_monthfirst_inferred_from_second_field_gt_12():
    exp = parse_export("8/20/26, 6:00 PM - X: a\n8/3/26, 6:00 PM - X: b")
    assert exp.dayfirst is False
    assert exp.messages[0].dt == datetime(2026, 8, 20, 18, 0)


def test_ambiguous_dates_flagged_but_still_parse():
    exp = parse_export("01/02/2026, 10:00 - X: a")
    assert exp.date_ambiguous is True
    assert exp.messages[0].dt == datetime(2026, 2, 1, 10, 0)  # dayfirst default


def test_explicit_dayfirst_override_wins():
    exp = parse_export("01/02/2026, 10:00 - X: a", dayfirst=False)
    assert exp.messages[0].dt == datetime(2026, 1, 2, 10, 0)


# --- platform + filename helpers ---------------------------------

def test_platform_detection():
    assert parse_export("[1/2/26, 3:04:05 PM] X: a").platform == "ios"
    assert parse_export("1/2/26, 15:04 - X: a").platform == "android"


def test_stray_leading_line_ignored():
    msgs = only_messages("some export header junk\n25/08/26, 21:14 - X: real")
    assert len(msgs) == 1 and msgs[0].text == "real"


@pytest.mark.parametrize(
    "fname,expected",
    [
        ("WhatsApp Chat with Rahul.txt", "Rahul"),
        ("WhatsApp Chat with College Friends.txt", "College Friends"),
        ("_chat.txt", "_chat"),
    ],
)
def test_conversation_name_from_filename(fname, expected):
    assert conversation_name_from_filename(fname) == expected
