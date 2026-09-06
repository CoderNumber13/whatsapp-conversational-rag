"""Robust parser for WhatsApp exported ``.txt`` chats.

Handles the two dominant export shapes and their many small variations:

    iOS      [08/25/2026, 9:14:07 PM] Rahul: Bro did you apply?
    Android  25/08/26, 21:14 - Rahul: Bro did you apply?

...plus multiline messages, system/notification lines, media placeholders,
edited-message markers, 2- vs 4-digit years, 12h/24h clocks, the narrow
no-break space (U+202F) iOS puts before AM/PM, and LTR/RTL marks (U+200E/F).

The parser is format-only: it produces :class:`ParsedMessage` objects.
Turning those into the normalized :class:`~src.storage.models.Message` schema
(conversation id, "is this me", ordering, dedup) is the normalizer's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Invisible directionality marks WhatsApp sprinkles into exports.
_BIDI = dict.fromkeys(map(ord, "\u200E\u200F\u202A\u202B\u202C"), None)
# Horizontal whitespace only (space, tab, NBSP U+00A0, narrow NBSP U+202F) - never a newline.
_SP = r"[^\S\n]"

# --- header regexes --------------------------------------------------------
# Groups: d1, d2, year, hour, minute, second?, ampm?, rest
_DATE = r"(\d{1,2})/(\d{1,2})/(\d{2,4})"
_TIME = rf"(\d{{1,2}}):(\d{{2}})(?::(\d{{2}}))?(?:{_SP}*([APap][Mm]))?"

_IOS_RE = re.compile(rf"^\[{_DATE},{_SP}{_TIME}\]{_SP}?(.*)$", re.DOTALL)
_ANDROID_RE = re.compile(rf"^{_DATE},{_SP}{_TIME}{_SP}-{_SP}(.*)$", re.DOTALL)

# "Sender: body" split - sender part kept short and colon-free.
_SENDER_RE = re.compile(rf"^([^:\n]{{1,80}}):{_SP}(.*)$", re.DOTALL)

# System lines that arrive *with* a "Name: " prefix ("Rahul pinned a message: ...").
_SYSTEM_VERB_RE = re.compile(
    r"\b(pinned a message|changed the subject|changed this group|changed the group|"
    r"changed their|added|removed|left|created|turned on disappearing|"
    r"turned off disappearing)\b",
    re.IGNORECASE,
)

_SYSTEM_TEXT_RE = re.compile(
    r"("
    r"messages and calls are end-to-end encrypted|"
    r"your security code with|security code changed|"
    r"created (this )?group|created group|"
    r"added you|joined using (this group's|your) invite link|"
    r"changed the subject|changed this group's icon|changed the group description|"
    r"changed the group name|changed the group's settings|"
    r"changed their phone number|changed to \+?\d|"
    r"you're now an admin|you were added|"
    r"turned (on|off) disappearing messages|"
    r"missed (a )?(voice|video) call|"
    r"blocked this contact|you unblocked|"
    r"this business (uses|works with)"
    r")",
    re.IGNORECASE,
)

_DELETED = {
    "this message was deleted",
    "this message was deleted.",
    "you deleted this message",
    "you deleted this message.",
}

_EDITED_SUFFIX_RE = re.compile(rf"{_SP}*<This message was edited>{_SP}*$", re.IGNORECASE)

# --- media placeholders --------------------------------------------------
_IOS_OMITTED_RE = re.compile(
    r"^(image|video|audio|sticker|gif|document|contact card)(?: file)? omitted$",
    re.IGNORECASE,
)
_IOS_ATTACHED_RE = re.compile(r"^<attached:\s*(.+?)>$", re.IGNORECASE)
_ANDROID_ATTACHED_RE = re.compile(r"^(?:(.+?)\s+)?\(file attached\)$", re.IGNORECASE)
_LOCATION_RE = re.compile(
    r"^location:\s*https?://|maps\.google\.com/|maps\.app\.goo\.gl/", re.IGNORECASE
)

_EXT_MEDIA = {
    "jpg": "image", "jpeg": "image", "png": "image", "webp": "sticker", "gif": "gif",
    "mp4": "video", "3gp": "video", "mov": "video",
    "opus": "audio", "m4a": "audio", "mp3": "audio", "aac": "audio", "ogg": "audio",
    "pdf": "document", "doc": "document", "docx": "document", "txt": "document",
    "xlsx": "document", "ppt": "document", "pptx": "document", "zip": "document",
    "vcf": "contact",
}
_PREFIX_MEDIA = {
    "IMG": "image", "VID": "video", "AUD": "audio", "PTT": "audio",
    "STK": "sticker", "GIF": "gif", "DOC": "document",
}


@dataclass
class ParsedMessage:
    dt: datetime
    sender: Optional[str]  # None => system / notification line
    text: str
    is_system: bool = False
    media_type: Optional[str] = None
    edited: bool = False
    deleted: bool = False
    line_start: int = 0
    line_end: int = 0


@dataclass
class ParsedExport:
    messages: list[ParsedMessage] = field(default_factory=list)
    platform: str = "unknown"  # ios | android | unknown
    dayfirst: bool = True
    date_ambiguous: bool = False
    name_hint: Optional[str] = None


# --- helpers -------------------------------------------------------------

_NAME_FROM_FILE_RE = re.compile(r"(?:whatsapp chat (?:with|-) )(.+?)(?:\.txt)?$", re.IGNORECASE)


def conversation_name_from_filename(path: str | Path) -> Optional[str]:
    stem = Path(path).name
    m = _NAME_FROM_FILE_RE.search(stem)
    if m:
        return m.group(1).strip()
    return Path(path).stem or None


def _clean(s: str) -> str:
    return s.translate(_BIDI)


def _make_dt(d1: int, d2: int, year: int, hh: int, mm: int, ss: int, dayfirst: bool) -> datetime:
    if year < 100:
        year += 2000
    day, month = (d1, d2) if dayfirst else (d2, d1)
    try:
        return datetime(year, month, day, hh, mm, ss)
    except ValueError:
        # Swapped order is the only thing that can save this line.
        day, month = month, day
        return datetime(year, month, day, hh, mm, ss)


def _apply_ampm(hh: int, ampm: Optional[str]) -> int:
    if not ampm:
        return hh
    ampm = ampm.lower()
    if ampm == "am":
        return 0 if hh == 12 else hh
    return hh if hh == 12 else hh + 12


def _match_header(line: str) -> Optional[tuple[str, tuple]]:
    m = _IOS_RE.match(line)
    if m:
        return "ios", m.groups()
    m = _ANDROID_RE.match(line)
    if m:
        return "android", m.groups()
    return None


def _classify_media(body: str) -> Optional[str]:
    b = body.strip()
    if not b:
        return None
    if b.lower() in {"<media omitted>", "null"}:
        return "media"
    m = _IOS_OMITTED_RE.match(b)
    if m:
        kind = m.group(1).lower()
        return {"contact card": "contact"}.get(kind, kind)
    m = _IOS_ATTACHED_RE.match(b) or _ANDROID_ATTACHED_RE.match(b)
    if m:
        fname = (m.group(1) or "").strip()
        if not fname:
            return "media"
        prefix = fname.split("-", 1)[0].upper()
        if prefix in _PREFIX_MEDIA:
            return _PREFIX_MEDIA[prefix]
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        return _EXT_MEDIA.get(ext, "media")
    if _LOCATION_RE.search(b):
        return "location"
    return None


def _finalize(raw_body: str) -> tuple[str, Optional[str], bool, bool]:
    """Return (text, media_type, edited, deleted) for a message body."""
    body = _clean(raw_body).strip()

    edited = bool(_EDITED_SUFFIX_RE.search(body))
    if edited:
        body = _EDITED_SUFFIX_RE.sub("", body).rstrip()

    if body.lower() in _DELETED:
        return body, None, edited, True

    media = _classify_media(body)
    if media:
        return "", media, edited, False

    return body, None, edited, False


# --- main entry points -------------------------------------------------

def parse_export(
    text: str,
    *,
    dayfirst: Optional[bool] = None,
    name_hint: Optional[str] = None,
) -> ParsedExport:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Pass 1: collect raw (header, body, span) records, merging continuations.
    raw: list[dict] = []
    platform_votes = {"ios": 0, "android": 0}
    for i, line in enumerate(lines):
        hdr = _match_header(line)
        if hdr is None:
            if raw:  # continuation of the previous message
                raw[-1]["body"] += "\n" + line
                raw[-1]["end"] = i
            # else: stray leading line -> ignored
            continue
        plat, g = hdr
        platform_votes[plat] += 1
        d1, d2, year, hh, mm, ss, ampm, rest = g
        raw.append(
            {
                "nums": (int(d1), int(d2), int(year), int(hh), int(mm), int(ss or 0), ampm),
                "body": rest,
                "start": i,
                "end": i,
            }
        )

    platform = (
        max(platform_votes, key=platform_votes.get) if any(platform_votes.values()) else "unknown"
    )

    # Infer day/month order from the whole file if not told.
    ambiguous = False
    if dayfirst is None:
        seen_d1_gt12 = any(r["nums"][0] > 12 for r in raw)
        seen_d2_gt12 = any(r["nums"][1] > 12 for r in raw)
        if seen_d1_gt12 and not seen_d2_gt12:
            dayfirst = True
        elif seen_d2_gt12 and not seen_d1_gt12:
            dayfirst = False
        else:
            dayfirst = True  # rest-of-world default
            ambiguous = not (seen_d1_gt12 or seen_d2_gt12) and bool(raw)

    # Pass 2: build ParsedMessage objects.
    messages: list[ParsedMessage] = []
    for r in raw:
        d1, d2, year, hh, mm, ss, ampm = r["nums"]
        hh = _apply_ampm(hh, ampm)
        try:
            dt = _make_dt(d1, d2, year, hh, mm, ss, dayfirst)
        except ValueError:
            continue  # unrecoverable timestamp; drop the line

        body = r["body"]
        sender: Optional[str] = None
        is_system = False

        sm = _SENDER_RE.match(body)
        if sm and not _SYSTEM_VERB_RE.search(sm.group(1)):
            sender = _clean(sm.group(1)).strip()
            text_body, media, edited, deleted = _finalize(sm.group(2))
        else:
            # No "Name: " prefix (or it was a system verb phrase) -> notification.
            is_system = True
            cleaned = _clean(body).strip()
            text_body, media, edited, deleted = cleaned, None, False, False
            bare_media = _classify_media(cleaned)
            if bare_media:  # rare export bug: media placeholder with no sender
                is_system = False
                text_body, media = "", bare_media

        # A "deleted message" notice is content from a real sender, not a system line.
        if deleted and sender is not None:
            is_system = False

        messages.append(
            ParsedMessage(
                dt=dt,
                sender=sender,
                text=text_body,
                is_system=is_system,
                media_type=media,
                edited=edited,
                deleted=deleted,
                line_start=r["start"] + 1,
                line_end=r["end"] + 1,
            )
        )

    return ParsedExport(
        messages=messages,
        platform=platform,
        dayfirst=bool(dayfirst),
        date_ambiguous=ambiguous,
        name_hint=name_hint,
    )


def parse_file(path: str | Path, *, dayfirst: Optional[bool] = None) -> ParsedExport:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return parse_export(text, dayfirst=dayfirst, name_hint=conversation_name_from_filename(p))
