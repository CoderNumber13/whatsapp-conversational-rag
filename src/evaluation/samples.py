"""The synthetic sample corpus, materialised on demand.

The four sample chats are **generated**, not stored. They used to be tracked as
``.txt`` files under ``data/sample/synthetic_chats/``, which was a liability for
a public repository: files named "WhatsApp Chat with Amma.txt" read as somebody's
private conversation whether or not they are invented, and a repo whose entire
premise is "your chats are never committed" should not ship anything that looks
like the opposite.

``scripts/generate_synthetic_chats.py`` produces them deterministically and
byte-for-byte identically to the files that were tracked, so every published
benchmark number in ``docs/BASELINE.md`` still reproduces exactly. Nothing about
the corpus changed; only where it comes from.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "synthetic_chats"


def ensure_sample_corpus(dest: Path | None = None) -> Path:
    """Return a directory holding the synthetic chats, generating them if absent.

    Idempotent and cheap: if the files are already there it does nothing, so
    callers can invoke it freely before reading the directory.
    """
    target = Path(dest) if dest is not None else SAMPLE_DIR
    if target.is_dir() and any(target.glob("*.txt")):
        return target

    from scripts.generate_synthetic_chats import write_all

    target.mkdir(parents=True, exist_ok=True)
    write_all(target)
    return target
