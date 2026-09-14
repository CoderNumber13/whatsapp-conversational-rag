"""The grounded-answer prompt.

Rules enforced on the model: use only the excerpts, cite every factual
sentence with ``[m:<id>]`` copied verbatim, mark inferences, and emit the
literal token ``NOT_FOUND`` when the excerpts don't contain the answer.
"""

from __future__ import annotations

import re
from typing import Iterable

from src.storage.models import Message

NOT_FOUND_TOKEN = "NOT_FOUND"


# The sentinel the prompt asks for, allowing NOT_FOUND / NOTFOUND but NOT the
# spaced prose form: an answer may legitimately contain 'not found'
# ("the parcel was not found"), and that is content, not a refusal.
_NOT_FOUND_SENTINEL_RE = re.compile(r"\bNOT_?FOUND\b", re.IGNORECASE)

# A refusal wrapped in a sentence is still a refusal, but a long answer that
# merely mentions the phrase is not. Smaller local models often reply
# "The final answer is NOT_FOUND." instead of the bare token; treating that as
# an answer would surface the raw sentinel to the user as if it were content.
_MAX_REFUSAL_CHARS = 120


def is_not_found(text: str) -> bool:
    """True if the model's reply is a 'not found' signal.

    Tolerates spacing, punctuation and casing ('NOTFOUND', 'not found.',
    'NOT_FOUND'), and the token embedded in a short sentence. A reply that
    cites a source is never a refusal, however it is phrased.
    """
    stripped = text.strip()
    squashed = "".join(ch for ch in stripped.upper() if ch.isalnum())
    if squashed in {"NOTFOUND", "NOTFOUNDINCHATS"}:
        return True
    if len(stripped) <= 40 and squashed.startswith("NOTFOUND"):
        return True
    # e.g. "The final answer is NOT_FOUND." -- a refusal the model dressed up.
    return (
        len(stripped) <= _MAX_REFUSAL_CHARS
        and bool(_NOT_FOUND_SENTINEL_RE.search(stripped))
        and "[m:" not in stripped
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
