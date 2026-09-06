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
    "You answer questions about the user's own chat history using only the "
    "message excerpts you are given.\n\n"
    "The excerpts are real messages already retrieved as relevant to the "
    "question. Read them carefully — if they contain the answer, give it.\n\n"
    "Rules:\n"
    "1. Use ONLY the excerpts. No outside knowledge, no guessing.\n"
    "2. After each factual sentence, put the citation(s) that support it: the "
    "[m:<id>] tag(s) copied verbatim from the excerpt lines you used.\n"
    "3. Prefix any sentence that is your inference (not explicitly stated) with "
    "'Likely: '.\n"
    "4. Only if NONE of the excerpts address the question, reply with exactly "
    f"{NOT_FOUND_TOKEN} (underscore included) and nothing else.\n"
    "5. Be concise. Never write a message id except inside a [m:<id>] citation.\n\n"
    "Example excerpt line:\n"
    "[2026-08-17 09:16] Rahul: Joining date is 2 September. [m:9efd65ab13262592]\n"
    "Example answer:\n"
    "Rahul's joining date is 2 September [m:9efd65ab13262592].\n"
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
