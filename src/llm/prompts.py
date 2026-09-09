"""The grounded-answer prompt.

Rules enforced on the model: use only the excerpts, cite every factual
sentence with ``[m:<id>]`` copied verbatim, mark inferences, and emit the
literal token ``NOT_FOUND`` when the excerpts don't contain the answer.
"""

from __future__ import annotations

from typing import Iterable

from src.storage.models import Message

NOT_FOUND_TOKEN = "NOT_FOUND"


def is_not_found(text: str) -> bool:
    """True if the model's reply is a 'not found' signal, tolerating spacing /
    punctuation / casing variants ('NOTFOUND', 'not found.', 'NOT_FOUND')."""
    squashed = "".join(ch for ch in text.upper() if ch.isalnum())
    return squashed in {"NOTFOUND", "NOTFOUNDINCHATS"} or (
        len(text.strip()) <= 40 and squashed.startswith("NOTFOUND")
    )


SYSTEM_PROMPT = (
    "You answer questions about the user's own chat history.\n"
    "Rules:\n"
    "1. Use ONLY the numbered excerpts below. Do not use outside knowledge, and "
    "do not fill a gap in the excerpts with anything you know or assume.\n"
    "2. End every factual sentence with one or more citations of the form "
    "[m:<id>], copied exactly from the excerpts that support it.\n"
    "3. If the excerpts do not contain the answer, reply with exactly "
    f"{NOT_FOUND_TOKEN} and nothing else. Partial or merely related excerpts "
    "are not an answer — prefer NOT_FOUND over a guess.\n"
    "4. Separate what was explicitly said from your own inference. Prefix any "
    "inferred sentence with 'Likely: '.\n"
    "5. Be concise. Do not quote message ids anywhere except inside [m:<id>] "
    "citations. Never invent names, dates, numbers, events, or ids.\n"
    "6. NEVER present a password, passcode, PIN, OTP, API key, account number "
    "or any other credential unless an excerpt states in words that the value "
    "IS that credential. A value that merely appears near the topic — a handle, "
    "an email address, a code, a link — is NOT a credential. Do not guess one, "
    "do not offer a candidate, and do not label such a guess 'Likely'. If asked "
    f"for a credential the excerpts do not explicitly state, reply "
    f"{NOT_FOUND_TOKEN}.\n"
)


def _render_message(m: Message) -> str:
    who = m.sender or ("system" if m.is_system else "unknown")
    if m.deleted:
        body = "(message deleted)"
    elif m.text:
        body = m.text.replace("\n", " / ")
    elif m.media_type:
        body = f"[{m.media_type}]"
    else:
        body = ""
    return f"[{m.timestamp:%Y-%m-%d %H:%M}] {who}: {body} [m:{m.message_id}]"


def build_context_block(retrieved: Iterable) -> str:
    """`retrieved` is an iterable of RetrievedChunk (duck-typed to avoid an
    import cycle with the retrieval package)."""
    parts: list[str] = []
    for i, rc in enumerate(retrieved, 1):
        conv = rc.conversation
        kind = "group" if conv.get("is_group") else "direct"
        lines = "\n".join(_render_message(m) for m in rc.ordered_messages())
        parts.append(
            f"### Excerpt {i} — {conv.get('name', '?')} ({kind}), "
            f"relevance {rc.score:.2f}\n{lines}"
        )
    return "\n\n".join(parts)


def build_user_prompt(question: str, retrieved: Iterable, *, filters_note: str = "") -> str:
    ctx = build_context_block(retrieved)
    note = f"\n(Retrieval was restricted to: {filters_note})\n" if filters_note else ""
    return (
        f"Question: {question}\n{note}\n"
        f"Excerpts:\n{ctx}\n\n"
        "Answer (with [m:<id>] citations, or the single token "
        f"{NOT_FOUND_TOKEN}):"
    )
