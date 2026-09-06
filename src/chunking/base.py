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

from typing import Protocol, Sequence

from src.storage.models import Chunk, Message, stable_id

CHUNKER_VERSION = "v1"


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
