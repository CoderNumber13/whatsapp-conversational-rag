"""Retrieval orchestration: metadata filter -> semantic search -> context.

Phase 1 is vector-only. BM25 / hybrid / reranking are Phase 2 and slot in
between the candidate filter and the returned list without changing this API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.config import CONFIG, Config
from src.storage.database import Database, row_to_message
from src.storage.models import Chunk, Message
from src.retrieval.vector_search import ScoredChunk, VectorSearch


@dataclass
class RetrievalFilters:
    conversation_id: Optional[str] = None
    sender: Optional[str] = None
    participant: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    source: Optional[str] = None
    is_group: Optional[bool] = None

    def as_kwargs(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    messages: list[Message]
    conversation: dict
    context_before: list[Message] = field(default_factory=list)
    context_after: list[Message] = field(default_factory=list)

    def ordered_messages(self) -> list[Message]:
        """Chunk messages plus any expanded context, in conversation order, de-duped."""
        seen: set[str] = set()
        out: list[Message] = []
        for m in [*self.context_before, *self.messages, *self.context_after]:
            if m.message_id not in seen:
                seen.add(m.message_id)
                out.append(m)
        return sorted(out, key=lambda m: m.seq)


class Retriever:
    def __init__(self, db: Database, vector_search: VectorSearch, config: Config = CONFIG) -> None:
        self.db = db
        self.vs = vector_search
        self.config = config

    @classmethod
    def open(cls, db: Database, config: Config = CONFIG) -> "Retriever":
        return cls(db, VectorSearch.open(db, config), config)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None,
        *,
        expand: bool = True,
    ) -> list[RetrievedChunk]:
        k = k or self.config.retrieval_top_k
        candidates = (
            self.db.filter_chunk_ids(**filters.as_kwargs()) if filters else None
        )
        if candidates is not None and not candidates:
            return []  # filters excluded everything — do not fall back to unfiltered

        scored = self.vs.search(query, k=k, candidate_chunk_ids=candidates)
        return [self._hydrate(sc, expand) for sc in scored]

    def _hydrate(self, sc: ScoredChunk, expand: bool) -> RetrievedChunk:
        chunk = sc.chunk
        conv = self.db.get_conversation(chunk.conversation_id)
        typed = [row_to_message(r) for r in self.db.get_messages_by_ids(chunk.message_ids)]

        before: list[Message] = []
        after: list[Message] = []
        if expand and self.config.context_window_messages > 0:
            w = self.config.context_window_messages
            span = self.db.context_window(
                chunk.conversation_id, chunk.seq_start, chunk.seq_end, before=w, after=w
            )
            member_ids = set(chunk.message_ids)
            for m in span:
                if m.message_id in member_ids:
                    continue
                (before if m.seq < chunk.seq_start else after).append(m)

        return RetrievedChunk(
            chunk=chunk,
            score=sc.score,
            messages=typed,
            conversation={
                "conversation_id": chunk.conversation_id,
                "name": conv["name"] if conv else "",
                "is_group": bool(conv["is_group"]) if conv else False,
                "source": conv["source"] if conv else "",
            },
            context_before=before,
            context_after=after,
        )
