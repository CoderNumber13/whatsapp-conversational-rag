"""Chunk the whole corpus and keep the ``chunks`` tables in sync.

Diffing is by ``chunk_id`` + ``content_hash``: unchanged chunks are left alone
(so their embeddings survive), changed/added chunks are rewritten, and chunks
that no longer exist are deleted (their embedding rows cascade).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chunking.base import Chunker
from src.chunking.message_chunker import FixedCountChunker
from src.chunking.time_chunker import TimeWindowChunker
from src.config import CONFIG, Config
from src.storage.database import Database


def get_chunker(config: Config = CONFIG) -> Chunker:
    if config.chunk_strategy == "time_window":
        return TimeWindowChunker(window_minutes=config.chunk_time_window_minutes)
    if config.chunk_strategy == "fixed_count":
        return FixedCountChunker(
            size=config.chunk_size_messages,
            overlap=config.chunk_overlap_messages,
            min_messages=config.min_messages_per_chunk,
        )
    raise ValueError(f"unknown CHUNK_STRATEGY: {config.chunk_strategy!r}")


@dataclass
class ChunkSyncResult:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0

    @property
    def changed_total(self) -> int:
        return self.added + self.updated + self.deleted

    def __add__(self, other: "ChunkSyncResult") -> "ChunkSyncResult":
        return ChunkSyncResult(
            self.added + other.added,
            self.updated + other.updated,
            self.deleted + other.deleted,
            self.unchanged + other.unchanged,
        )


class ChunkingService:
    def __init__(self, db: Database, chunker: Chunker | None = None, config: Config = CONFIG) -> None:
        self.db = db
        self.chunker = chunker or get_chunker(config)

    def sync_conversation(self, conversation_id: str) -> ChunkSyncResult:
        conv = self.db.get_conversation(conversation_id)
        if conv is None:
            return ChunkSyncResult()
        messages = self.db.get_conversation_messages(
            conversation_id, include_system=False, include_deleted=False
        )
        desired = (
            self.chunker.chunk(conversation_id, conv["name"], bool(conv["is_group"]), messages)
            if messages
            else []
        )
        existing = self.db.chunk_hashes_for_conversation(conversation_id)
        desired_ids = {c.chunk_id for c in desired}

        res = ChunkSyncResult()
        for chunk in desired:
            prev = existing.get(chunk.chunk_id)
            if prev is None:
                self.db.insert_chunk(chunk, commit=False)
                res.added += 1
            elif prev != chunk.content_hash:
                self.db.insert_chunk(chunk, commit=False)
                res.updated += 1
            else:
                res.unchanged += 1
        for orphan in existing.keys() - desired_ids:
            self.db.delete_chunk(orphan, commit=False)
            res.deleted += 1
        self.db.commit()
        return res

    def sync_all(self) -> ChunkSyncResult:
        total = ChunkSyncResult()
        for cid in self.db.conversation_ids():
            total = total + self.sync_conversation(cid)
        return total
