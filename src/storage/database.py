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

SCHEMA_VERSION = 1

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
    seq             INTEGER NOT NULL,
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
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes -------------------------------------------------------
    def upsert_conversation(self, conv: Conversation, me_names: Iterable[str] = ()) -> None:
        self.conn.execute(
            """
            INSERT INTO conversations
                (conversation_id, name, source, is_group, first_ts, last_ts,
                 message_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                name=excluded.name,
                is_group=excluded.is_group,
                first_ts=MIN(conversations.first_ts, excluded.first_ts),
                last_ts=MAX(conversations.last_ts, excluded.last_ts),
                message_count=excluded.message_count
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
                INSERT OR IGNORE INTO participants (conversation_id, display_name, is_me)
                VALUES (?, ?, ?)
                """,
                (conv.conversation_id, p, int(p.lower() in me_lower)),
            )
        self.conn.commit()

    def insert_messages(self, messages: Sequence[Message]) -> int:
        """Insert messages, ignoring exact-duplicate ids. Returns rows added."""
        before = self._count("messages")
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO messages
                (message_id, conversation_id, seq, timestamp, sender, sender_raw,
                 is_from_me, is_system, media_type, edited, deleted, text, source,
                 src_file, src_line_start, src_line_end, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    m.message_id, m.conversation_id, m.seq, _iso(m.timestamp), m.sender,
                    m.sender_raw, int(m.is_from_me), int(m.is_system), m.media_type,
                    int(m.edited), int(m.deleted), m.text, m.source, m.src_file,
                    m.src_line_start, m.src_line_end, _now(),
                )
                for m in messages
            ],
        )
        self.conn.commit()
        return self._count("messages") - before

    def insert_chunk(self, chunk: Chunk) -> None:
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
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunk_messages (chunk_id, message_id, ord) VALUES (?, ?, ?)",
            [(chunk.chunk_id, mid, i) for i, mid in enumerate(chunk.message_ids)],
        )
        self.conn.commit()

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
