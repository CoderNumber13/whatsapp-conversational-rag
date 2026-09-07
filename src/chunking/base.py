"""Chunker interface + shared rendering.

A chunk is a contiguous, conversation-bounded run of messages with a stable id
and a text representation that keeps per-message ids inline so answers can cite
individual messages.

``chunk_id`` is derived from the ids of its first and last member messages
(plus the chunker version and strategy params) — never from ``seq``, which the
DB recomputes on every import. Bump ``CHUNKER_VERSION`` whenever the rendered
text format changes, to force a global re-chunk + re-embed.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from src.storage.models import Chunk, Message, stable_id

CHUNKER_VERSION = "v1"

# Bump when the embedding *input* recipe below changes: stored vectors were
# built from the old recipe and must be regenerated. Tracked separately from
# CHUNKER_VERSION because it does not change chunk ids or the stored text.
EMBED_TEXT_VERSION = 2


def make_chunk_id(
    conversation_id: str,
    first_message_id: str,
    last_message_id: str,
    *strategy_params: object,
) -> str:
    return stable_id(
        CHUNKER_VERSION, conversation_id, first_message_id, last_message_id, *strategy_params
    )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def render_message_line(m: Message) -> str:
    """`[YYYY-MM-DD HH:MM] Sender: body  [m:<id>]` — one message, one line."""
    who = m.sender or ("system" if m.is_system else "unknown")
    if m.deleted:
        body = "(message deleted)"
    elif m.text:
        body = m.text.replace("\n", " ⏎ ")
    elif m.media_type:
        body = f"[{m.media_type}]"
    else:
        body = ""
    return f"[{m.timestamp:%Y-%m-%d %H:%M}] {who}: {body}  [m:{m.message_id}]".rstrip()


def render_chunk_text(conversation_name: str, is_group: bool, messages: Sequence[Message]) -> str:
    kind = "group" if is_group else "direct"
    participants = ", ".join(sorted({m.sender for m in messages if m.sender}))
    span_lo = min(m.timestamp for m in messages)
    span_hi = max(m.timestamp for m in messages)
    header = (
        f"Conversation: {conversation_name} ({kind}) | Participants: {participants}\n"
        f"Dates: {span_lo:%Y-%m-%d %H:%M} - {span_hi:%Y-%m-%d %H:%M}\n"
    )
    return header + "\n".join(render_message_line(m) for m in messages)


_M_TAG_RE = re.compile(r"[ \t]*\[m:[0-9a-f]+\]")
_LINE_TS_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*", re.M)


def render_embedding_text(text: str) -> str:
    """The retrieval view of a chunk: same content, without display scaffolding.

    ``render_chunk_text`` is built for the LLM prompt and the citation parser, so
    every line carries an ISO timestamp and an ``[m:<id>]`` tag. Those are ~60%
    of the tokens and carry no meaning for similarity, and they push a 12-message
    chunk well past all-MiniLM-L6-v2's 256-token input limit — where the tail is
    silently discarded, sometimes taking the only mention of the subject with it.

    Only the embedding input is affected. The stored ``chunk.text`` still goes to
    the prompt verbatim, so citations are unchanged. Per-message timestamps
    remain retrievable via metadata filters and the chunk's own date header.
    """
    return _LINE_TS_RE.sub("", _M_TAG_RE.sub("", text))


def build_chunk(
    conversation_id: str,
    conversation_name: str,
    is_group: bool,
    messages: Sequence[Message],
    *strategy_params: object,
) -> Chunk:
    ms = list(messages)
    text = render_chunk_text(conversation_name, is_group, ms)
    return Chunk(
        chunk_id=make_chunk_id(conversation_id, ms[0].message_id, ms[-1].message_id, *strategy_params),
        conversation_id=conversation_id,
        seq_start=ms[0].seq,
        seq_end=ms[-1].seq,
        ts_start=ms[0].timestamp,
        ts_end=ms[-1].timestamp,
        participants=sorted({m.sender for m in ms if m.sender}),
        message_ids=[m.message_id for m in ms],
        text=text,
        token_estimate=estimate_tokens(text),
    )


class Chunker(Protocol):
    name: str

    def chunk(
        self, conversation_id: str, conversation_name: str, is_group: bool, messages: Sequence[Message]
    ) -> list[Chunk]:
        ...
