"""SQLite store — the authoritative record of every message.

The vector index is derived data and can always be rebuilt from here.
Chunk / embedding tables are created now (schema stability) but only
populated from increment 2.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.storage.models import Chunk, Conversation, Message

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    source          TEXT NOT NULL,
    is_group        INTEGER NOT NULL DEFAULT 0,
    first_ts        TEXT,
    last_ts         TEXT,
    message_count   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS participants (
    conversation_id TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    is_me           INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (conversation_id, display_name)
);

CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    seq             INTEGER NOT NULL,   -- global order in conversation; recomputed on ingest
    local_ord       INTEGER NOT NULL,   -- position within its own export; set once, never changed
    timestamp       TEXT NOT NULL,
    sender          TEXT NOT NULL,
    sender_raw      TEXT NOT NULL,
    is_from_me      INTEGER NOT NULL DEFAULT 0,
    is_system       INTEGER NOT NULL DEFAULT 0,
    media_type      TEXT,
    edited          INTEGER NOT NULL DEFAULT 0,
    deleted         INTEGER NOT NULL DEFAULT 0,
    text            TEXT NOT NULL,
    source          TEXT NOT NULL,
    src_file        TEXT,
    src_line_start  INTEGER,
    src_line_end    INTEGER,
    ingested_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_conv_seq ON messages (conversation_id, seq);
CREATE INDEX IF NOT EXISTS ix_messages_sender   ON messages (sender);
CREATE INDEX IF NOT EXISTS ix_messages_ts       ON messages (timestamp);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    seq_start       INTEGER NOT NULL,
    seq_end         INTEGER NOT NULL,
    ts_start        TEXT NOT NULL,
    ts_end          TEXT NOT NULL,
    participants    TEXT NOT NULL,
    text            TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_conv ON chunks (conversation_id);

CREATE TABLE IF NOT EXISTS chunk_messages (
    chunk_id   TEXT NOT NULL,
    message_id TEXT NOT NULL,
    ord        INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, message_id)
);

CREATE TABLE IF NOT EXISTS chunk_participants (
    chunk_id     TEXT NOT NULL,
    display_name TEXT NOT NULL,
    PRIMARY KEY (chunk_id, display_name)
);
CREATE INDEX IF NOT EXISTS ix_chunk_participants_name ON chunk_participants (display_name);

CREATE TABLE IF NOT EXISTS embedding_meta (
    chunk_id        TEXT PRIMARY KEY,
    content_hash    TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    faiss_id        INTEGER,
    created_at      TEXT NOT NULL
);
"""


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def row_to_chunk(r: sqlite3.Row, message_ids: Optional[list[str]] = None) -> Chunk:
    return Chunk(
        chunk_id=r["chunk_id"],
        conversation_id=r["conversation_id"],
        seq_start=r["seq_start"],
        seq_end=r["seq_end"],
        ts_start=datetime.fromisoformat(r["ts_start"]),
        ts_end=datetime.fromisoformat(r["ts_end"]),
        participants=json.loads(r["participants"]),
        message_ids=message_ids if message_ids is not None else [],
        text=r["text"],
        content_hash=r["content_hash"],
        token_estimate=r["token_estimate"],
    )


def row_to_message(r: sqlite3.Row) -> Message:
    """Rebuild a typed :class:`Message` from a ``messages`` row."""
    return Message(
        conversation_id=r["conversation_id"],
        conversation_name="",  # not stored per-row; join conversations if needed
        timestamp=datetime.fromisoformat(r["timestamp"]),
        sender=r["sender"],
        text=r["text"],
        source=r["source"],
        sender_raw=r["sender_raw"],
        is_from_me=bool(r["is_from_me"]),
        is_system=bool(r["is_system"]),
        media_type=r["media_type"],
        edited=bool(r["edited"]),
        deleted=bool(r["deleted"]),
        src_file=r["src_file"] or "",
        src_line_start=r["src_line_start"] or 0,
        src_line_end=r["src_line_end"] or 0,
        message_id=r["message_id"],
        seq=r["seq"],
    )


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.init_schema()

    # --- lifecycle ------------------------------------------------------
    def init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def _migrate(self) -> None:
        """Bring a pre-v2 database forward in place (no real data exists yet, but
        a dev DB created during increment 1 would otherwise break on insert)."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(messages)").fetchall()}
        if cols and "local_ord" not in cols:
            self.conn.execute(
                "ALTER TABLE messages ADD COLUMN local_ord INTEGER NOT NULL DEFAULT 0"
            )
            for (cid,) in self.conn.execute(
                "SELECT DISTINCT conversation_id FROM messages"
            ).fetchall():
                self._recompute_seq(cid)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes -------------------------------------------------------
    def upsert_conversation(self, conv: Conversation, me_names: Iterable[str] = ()) -> None:
        """Upsert conversation identity + participants. Time span / message_count are
        owned by :meth:`refresh_conversation_stats` (called from ``insert_messages``)."""
        self.conn.execute(
            """
            INSERT INTO conversations
                (conversation_id, name, source, is_group, first_ts, last_ts,
                 message_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                name=excluded.name,
                source=excluded.source,
                is_group=MAX(conversations.is_group, excluded.is_group)
            """,
            (
                conv.conversation_id,
                conv.name,
                conv.source,
                int(conv.is_group),
                _iso(conv.first_ts),
                _iso(conv.last_ts),
                conv.message_count,
                _now(),
            ),
        )
        me_lower = {m.strip().lower() for m in me_names if m.strip()}
        for p in conv.participants:
            self.conn.execute(
                """
                INSERT INTO participants (conversation_id, display_name, is_me)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id, display_name) DO UPDATE SET
                    is_me = MAX(participants.is_me, excluded.is_me)
                """,
                (conv.conversation_id, p, int(p.lower() in me_lower)),
            )
        self.conn.commit()

    def insert_messages(self, messages: Sequence[Message]) -> int:
        """Insert messages (exact-duplicate ids ignored), then rebuild the global
        ``seq`` ordering and refresh conversation stats for every affected
        conversation. Returns the number of new rows. Idempotent."""
        before = self._count("messages")
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO messages
                (message_id, conversation_id, seq, local_ord, timestamp, sender,
                 sender_raw, is_from_me, is_system, media_type, edited, deleted, text,
                 source, src_file, src_line_start, src_line_end, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    m.message_id, m.conversation_id, m.seq, m.seq, _iso(m.timestamp),
                    m.sender, m.sender_raw, int(m.is_from_me), int(m.is_system),
                    m.media_type, int(m.edited), int(m.deleted), m.text, m.source,
                    m.src_file, m.src_line_start, m.src_line_end, _now(),
                )
                for m in messages
            ],
        )
        for conv_id in {m.conversation_id for m in messages}:
            self._recompute_seq(conv_id)
            self.refresh_conversation_stats(conv_id)
        self.conn.commit()
        return self._count("messages") - before

    def _recompute_seq(self, conversation_id: str) -> None:
        """Rebuild dense 0..n ``seq`` for a conversation across all imports.

        Order: timestamp, then arrival (``ingested_at``), then the message's
        position within its own export (``local_ord``), then id as a last resort.
        This keeps same-minute runs from one export in their original order and
        appends later imports after earlier ones.
        """
        self.conn.execute(
            """
            WITH ordered AS (
                SELECT message_id,
                       ROW_NUMBER() OVER (
                           ORDER BY timestamp, ingested_at, local_ord, message_id
                       ) - 1 AS rn
                FROM messages
                WHERE conversation_id = :cid
            )
            UPDATE messages
               SET seq = (SELECT rn FROM ordered WHERE ordered.message_id = messages.message_id)
             WHERE conversation_id = :cid
            """,
            {"cid": conversation_id},
        )

    def refresh_conversation_stats(self, conversation_id: str) -> None:
        self.conn.execute(
            """
            UPDATE conversations SET
                message_count = (SELECT COUNT(*)   FROM messages WHERE conversation_id = :cid),
                first_ts      = (SELECT MIN(timestamp) FROM messages WHERE conversation_id = :cid),
                last_ts       = (SELECT MAX(timestamp) FROM messages WHERE conversation_id = :cid)
            WHERE conversation_id = :cid
            """,
            {"cid": conversation_id},
        )

    def insert_chunk(self, chunk: Chunk, *, commit: bool = True) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO chunks
                (chunk_id, conversation_id, seq_start, seq_end, ts_start, ts_end,
                 participants, text, content_hash, token_estimate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id, chunk.conversation_id, chunk.seq_start, chunk.seq_end,
                _iso(chunk.ts_start), _iso(chunk.ts_end), json.dumps(chunk.participants),
                chunk.text, chunk.content_hash, chunk.token_estimate, _now(),
            ),
        )
        self.conn.execute("DELETE FROM chunk_messages WHERE chunk_id = ?", (chunk.chunk_id,))
        self.conn.executemany(
            "INSERT INTO chunk_messages (chunk_id, message_id, ord) VALUES (?, ?, ?)",
            [(chunk.chunk_id, mid, i) for i, mid in enumerate(chunk.message_ids)],
        )
        self.conn.execute("DELETE FROM chunk_participants WHERE chunk_id = ?", (chunk.chunk_id,))
        self.conn.executemany(
            "INSERT OR IGNORE INTO chunk_participants (chunk_id, display_name) VALUES (?, ?)",
            [(chunk.chunk_id, p) for p in chunk.participants],
        )
        if commit:
            self.conn.commit()

    def delete_chunk(self, chunk_id: str, *, commit: bool = True) -> None:
        for table in ("chunk_messages", "chunk_participants", "embedding_meta", "chunks"):
            self.conn.execute(f"DELETE FROM {table} WHERE chunk_id = ?", (chunk_id,))
        if commit:
            self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    # --- chunk reads ------------------------------------------------
    def chunk_hashes_for_conversation(self, conversation_id: str) -> dict[str, str]:
        return {
            r["chunk_id"]: r["content_hash"]
            for r in self.conn.execute(
                "SELECT chunk_id, content_hash FROM chunks WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
        }

    def _chunk_message_ids(self, chunk_id: str) -> list[str]:
        return [
            r["message_id"]
            for r in self.conn.execute(
                "SELECT message_id FROM chunk_messages WHERE chunk_id = ? ORDER BY ord",
                (chunk_id,),
            ).fetchall()
        ]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        r = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if r is None:
            return None
        return row_to_chunk(r, self._chunk_message_ids(chunk_id))

    def get_chunks_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        order = {cid: i for i, cid in enumerate(ids)}
        chunks = [c for cid in ids if (c := self.get_chunk(cid)) is not None]
        return sorted(chunks, key=lambda c: order.get(c.chunk_id, 1 << 30))

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY conversation_id, seq_start").fetchall()
        return [row_to_chunk(r, self._chunk_message_ids(r["chunk_id"])) for r in rows]

    def count_chunks(self) -> int:
        return self._count("chunks")

    # --- reads -------------------------------------------------------
    def _count(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def list_conversations(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM conversations ORDER BY last_ts DESC"
        ).fetchall()

    def get_conversation(self, conversation_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()

    def get_message(self, message_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM messages WHERE message_id = ?", (message_id,)
        ).fetchone()

    def get_messages_by_ids(self, ids: Sequence[str]) -> list[sqlite3.Row]:
        if not ids:
            return []
        qs = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM messages WHERE message_id IN ({qs})", tuple(ids)
        ).fetchall()
        order = {mid: i for i, mid in enumerate(ids)}
        return sorted(rows, key=lambda r: order.get(r["message_id"], 1 << 30))

    def get_messages(
        self,
        *,
        conversation_id: Optional[str] = None,
        sender: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        contains: Optional[str] = None,
        include_system: bool = True,
        limit: Optional[int] = None,
    ) -> list[sqlite3.Row]:
        clauses, params = [], []
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if sender:
            clauses.append("LOWER(sender) = ?")
            params.append(sender.lower())
        if start:
            clauses.append("timestamp >= ?")
            params.append(start.isoformat())
        if end:
            clauses.append("timestamp <= ?")
            params.append(end.isoformat())
        if contains:
            clauses.append("text LIKE ?")
            params.append(f"%{contains}%")
        if not include_system:
            clauses.append("is_system = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM messages {where} ORDER BY conversation_id, seq"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, tuple(params)).fetchall()

    def conversation_ids(self) -> list[str]:
        return [
            r["conversation_id"]
            for r in self.conn.execute(
                "SELECT conversation_id FROM conversations ORDER BY conversation_id"
            ).fetchall()
        ]

    def get_conversation_messages(
        self,
        conversation_id: str,
        *,
        include_system: bool = False,
        include_deleted: bool = True,
    ) -> list[Message]:
        """Typed, seq-ordered messages for one conversation — the chunker's input."""
        clauses = ["conversation_id = ?"]
        params: list = [conversation_id]
        if not include_system:
            clauses.append("is_system = 0")
        if not include_deleted:
            clauses.append("deleted = 0")
        rows = self.conn.execute(
            f"SELECT * FROM messages WHERE {' AND '.join(clauses)} ORDER BY seq",
            tuple(params),
        ).fetchall()
        return [row_to_message(r) for r in rows]

    def conversation_window(
        self, conversation_id: str, seq_start: int, seq_end: int
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ? AND seq BETWEEN ? AND ?
            ORDER BY seq
            """,
            (conversation_id, seq_start, seq_end),
        ).fetchall()

    def context_window(
        self, conversation_id: str, seq_start: int, seq_end: int, *, before: int = 0, after: int = 0
    ) -> list[Message]:
        """Typed messages spanning [seq_start - before, seq_end + after] (§14)."""
        rows = self.conversation_window(
            conversation_id, max(0, seq_start - before), seq_end + after
        )
        return [row_to_message(r) for r in rows]

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) n, MIN(timestamp) lo, MAX(timestamp) hi FROM messages"
        ).fetchone()
        senders = [
            r["sender"]
            for r in self.conn.execute(
                "SELECT DISTINCT sender FROM messages WHERE is_system = 0 AND sender != '' "
                "ORDER BY sender"
            ).fetchall()
        ]
        return {
            "messages": row["n"],
            "conversations": self._count("conversations"),
            "participants": senders,
            "date_range": (row["lo"], row["hi"]),
        }
