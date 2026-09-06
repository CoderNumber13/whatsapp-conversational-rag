"""Fixed message-count sliding-window chunker (the Phase 1 default).

Conversation-bounded. Windows of ``size`` messages step forward by
``size - overlap``. A trailing window shorter than ``min_messages`` is folded
into its predecessor so we never emit a 1- or 2-message tail chunk.
"""

from __future__ import annotations

from typing import Sequence

from src.chunking.base import Chunker, build_chunk
from src.storage.models import Chunk, Message


def _has_real_text(messages: Sequence[Message]) -> bool:
    return any(m.text and not m.deleted for m in messages)


class FixedCountChunker(Chunker):
    name = "fixed_count"

    def __init__(self, size: int = 12, overlap: int = 3, min_messages: int = 4) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        if not 0 <= overlap < size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap
        self.min_messages = min_messages
        self.step = size - overlap

    def _spans(self, n: int) -> list[tuple[int, int]]:
        if n == 0:
            return []
        spans: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + self.size, n)
            spans.append((start, end))
            if end == n:
                break
            start += self.step
        # fold an undersized tail into its predecessor
        if len(spans) >= 2 and spans[-1][1] - spans[-1][0] < self.min_messages:
            prev_start, _ = spans[-2]
            spans[-2] = (prev_start, spans[-1][1])
            spans.pop()
        return spans

    def chunk(
        self,
        conversation_id: str,
        conversation_name: str,
        is_group: bool,
        messages: Sequence[Message],
    ) -> list[Chunk]:
        ms = list(messages)
        out: list[Chunk] = []
        for start, end in self._spans(len(ms)):
            window = ms[start:end]
            if not _has_real_text(window):
                continue  # media-only / all-deleted span carries no retrievable signal
            out.append(
                build_chunk(
                    conversation_id,
                    conversation_name,
                    is_group,
                    window,
                    self.name,
                    self.size,
                    self.overlap,
                )
            )
        return out
