"""Normalized, source-independent data model.

Nothing downstream (chunking, retrieval, graph, LLM) should depend on the
original WhatsApp text format — it depends only on these objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

SOURCE_WHATSAPP = "whatsapp"

# Coarse media categories used across every source.
MEDIA_TYPES = {
    "image",
    "video",
    "audio",  # includes voice notes
    "document",
    "sticker",
    "gif",
    "contact",
    "location",
    "media",  # type present but unknown (e.g. Android "<Media omitted>")
}


def stable_id(*parts: object, length: int = 16) -> str:
    """Deterministic short id from its parts (used for message/conversation ids)."""
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


@dataclass
class Message:
    """A single normalized message."""

    conversation_id: str
    conversation_name: str
    timestamp: datetime  # naive, local wall-clock (WhatsApp exports carry no tz)
    sender: str  # normalized display name ("" for system messages)
    text: str
    source: str = SOURCE_WHATSAPP

    sender_raw: str = ""  # sender exactly as it appeared in the export
    is_from_me: bool = False
    is_system: bool = False
    media_type: Optional[str] = None
    edited: bool = False
    deleted: bool = False

    # Provenance
    src_file: str = ""
    src_line_start: int = 0
    src_line_end: int = 0

    message_id: str = ""
    seq: int = -1  # order within the conversation, filled by the normalizer

    def __post_init__(self) -> None:
        if not self.sender_raw:
            self.sender_raw = self.sender
        if self.media_type is not None and self.media_type not in MEDIA_TYPES:
            raise ValueError(f"unknown media_type: {self.media_type!r}")
        if not self.message_id:
            self.message_id = stable_id(
                self.conversation_id,
                self.timestamp.isoformat(),
                self.sender_raw,
                self.text,
            )

    @property
    def is_media(self) -> bool:
        return self.media_type is not None

    def display_line(self) -> str:
        """`[HH:MM] Sender: text` — the form used inside chunk text."""
        who = self.sender or ("system" if self.is_system else "unknown")
        body = self.text if self.text else (f"[{self.media_type}]" if self.media_type else "")
        return f"[{self.timestamp:%H:%M}] {who}: {body}".rstrip()


@dataclass
class Conversation:
    conversation_id: str
    name: str
    source: str = SOURCE_WHATSAPP
    is_group: bool = False
    participants: list[str] = field(default_factory=list)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    message_count: int = 0


@dataclass
class Chunk:
    """A contiguous, conversation-bounded span of messages (increment 2 populates text)."""

    chunk_id: str
    conversation_id: str
    seq_start: int
    seq_end: int
    ts_start: datetime
    ts_end: datetime
    participants: list[str]
    message_ids: list[str]
    text: str
    content_hash: str = ""
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
