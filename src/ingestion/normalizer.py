"""Turn a format-specific :class:`ParsedExport` into the normalized schema."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from src.ingestion.base import ChatSource
from src.ingestion.whatsapp_parser import (
    ParsedExport,
    conversation_name_from_filename,
    parse_file,
)
from src.storage.models import (
    SOURCE_WHATSAPP,
    Conversation,
    Message,
    stable_id,
)

_GROUP_CREATE_HINTS = ("created group", "created this group", "added you", "you were added")


def _norm_name(name: str) -> str:
    return " ".join(name.split()).strip()


def derive_conversation_id(source: str, name: str) -> str:
    return stable_id(source, _norm_name(name).lower())


def normalize_export(
    parsed: ParsedExport,
    *,
    conversation_name: str,
    me_names: Iterable[str],
    conversation_id: Optional[str] = None,
    source: str = SOURCE_WHATSAPP,
    src_file: str = "",
) -> tuple[Conversation, list[Message]]:
    name = _norm_name(conversation_name) or "Unknown"
    conv_id = conversation_id or derive_conversation_id(source, name)
    me_lower = {m.strip().lower() for m in me_names if m.strip()}

    # Stable chronological order (parser order breaks ties).
    ordered = sorted(enumerate(parsed.messages), key=lambda t: (t[1].dt, t[0]))

    messages: list[Message] = []
    seen_ids: set[str] = set()
    senders: list[str] = []
    group_hint = False

    for _, pm in ordered:
        sender_raw = pm.sender or ""
        sender = _norm_name(sender_raw)
        is_from_me = sender.lower() in me_lower if sender else False

        if pm.is_system and any(h in pm.text.lower() for h in _GROUP_CREATE_HINTS):
            group_hint = True

        msg = Message(
            conversation_id=conv_id,
            conversation_name=name,
            timestamp=pm.dt,
            sender=sender,
            text=pm.text,
            source=source,
            sender_raw=sender_raw,
            is_from_me=is_from_me,
            is_system=pm.is_system,
            media_type=pm.media_type,
            edited=pm.edited,
            deleted=pm.deleted,
            src_file=src_file,
            src_line_start=pm.line_start,
            src_line_end=pm.line_end,
        )
        if msg.message_id in seen_ids:  # exact duplicate (e.g. re-exported file)
            continue
        seen_ids.add(msg.message_id)
        messages.append(msg)
        if sender and not pm.is_system:
            senders.append(sender)

    for i, msg in enumerate(messages):
        msg.seq = i

    distinct_senders = sorted(set(senders))
    is_group = len(distinct_senders) > 2 or group_hint

    conv = Conversation(
        conversation_id=conv_id,
        name=name,
        source=source,
        is_group=is_group,
        participants=distinct_senders,
        first_ts=messages[0].timestamp if messages else None,
        last_ts=messages[-1].timestamp if messages else None,
        message_count=len(messages),
    )
    return conv, messages


class WhatsAppExportSource(ChatSource):
    """Ingest one or more WhatsApp ``.txt`` export files."""

    source_name = SOURCE_WHATSAPP

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        *,
        me_names: Iterable[str] = ("Me",),
        dayfirst: Optional[bool] = None,
    ) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        self.me_names = list(me_names)
        self.dayfirst = dayfirst
        self._messages: list[Message] = []
        self._conversations: list[Conversation] = []

    def ingest(self) -> None:
        self._messages = []
        self._conversations = []
        for path in self.paths:
            parsed = parse_file(path, dayfirst=self.dayfirst)
            name = parsed.name_hint or conversation_name_from_filename(path) or path.stem
            conv, msgs = normalize_export(
                parsed,
                conversation_name=name,
                me_names=self.me_names,
                source=self.source_name,
                src_file=path.name,
            )
            self._conversations.append(conv)
            self._messages.extend(msgs)

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def get_conversations(self) -> list[Conversation]:
        return list(self._conversations)
