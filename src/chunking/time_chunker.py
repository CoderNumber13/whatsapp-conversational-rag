"""Time-window chunker: split a conversation into sessions separated by a gap
of at least ``window_minutes``, then cap over-long sessions at ``max_messages``.

Not the Phase 1 default; provided so retrieval strategies can be compared
(spec §8). Selected via CHUNK_STRATEGY=time_window.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Sequence

from src.chunking.base import Chunker, build_chunk
from src.chunking.message_chunker import _has_real_text
from src.storage.models import Chunk, Message


class TimeWindowChunker(Chunker):
    name = "time_window"

    def __init__(self, window_minutes: int = 45, max_messages: int = 30) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.max_messages = max_messages

    def _sessions(self, messages: Sequence[Message]) -> list[list[Message]]:
        sessions: list[list[Message]] = []
        cur: list[Message] = []
        for m in messages:
            if cur and (m.timestamp - cur[-1].timestamp > self.window or len(cur) >= self.max_messages):
                sessions.append(cur)
                cur = []
            cur.append(m)
        if cur:
            sessions.append(cur)
        return sessions

    def chunk(
        self,
        conversation_id: str,
        conversation_name: str,
        is_group: bool,
        messages: Sequence[Message],
    ) -> list[Chunk]:
        out: list[Chunk] = []
        for session in self._sessions(list(messages)):
            if not _has_real_text(session):
                continue
            out.append(
                build_chunk(
                    conversation_id,
                    conversation_name,
                    is_group,
                    session,
                    self.name,
                    int(self.window.total_seconds() // 60),
                )
            )
        return out
