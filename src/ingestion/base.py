"""Ingestion source interface.

WhatsApp is only the first source. Anything that can yield normalized
:class:`~src.storage.models.Message` objects (Discord, Telegram, a JSON dump)
implements :class:`ChatSource`; the rest of the pipeline never sees the raw
format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.storage.models import Conversation, Message


class ChatSource(ABC):
    source_name: str = "base"

    @abstractmethod
    def ingest(self) -> None:
        """Read + parse + normalize the underlying data into memory.

        Idempotent: calling it again re-reads the source.
        """

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """All normalized messages, ordered by (conversation, seq)."""

    @abstractmethod
    def get_conversations(self) -> list[Conversation]:
        """Conversation summaries derived from the ingested messages."""
