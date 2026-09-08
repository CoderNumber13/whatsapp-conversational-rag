"""Lexical (BM25) search over the same chunks the vector index holds.

Phase 2, step 1. Standalone: nothing fuses this with the dense retriever yet, so
it can be measured on its own against the recorded vector baseline.

Why lexical retrieval is here at all: the benchmark shows dense retrieval scoring
0.0% Recall@1 on the ``exact_term`` category, and failing to retrieve ``TCS`` at
all within the top 10 of 144 chunks. Rare proper nouns and acronyms carry little
signal in a 384-dimension sentence embedding but are exactly what an IDF-weighted
term match is good at.

Index lifetime
--------------
The BM25 index is built in memory from the ``chunks`` table when the searcher is
opened, not persisted. For corpora of this size that costs milliseconds, and it
removes a whole staleness problem: there is no second artifact that can drift
from SQLite the way the FAISS index can. Revisit if the corpus grows enough for
build time to matter.

Scores are BM25 sums, not similarities: unbounded, corpus-relative, and NOT
comparable with the cosine scores from ``VectorSearch``. Compare ranks.

Known behaviour worth expecting rather than debugging: BM25Okapi's IDF is
``log(N-n+0.5) - log(n+0.5)``, which is exactly 0 for a term appearing in half
the chunks and negative beyond that. A query made only of such terms scores zero
everywhere and returns nothing — correct, since those terms carry no
discriminating evidence, but it is why BM25 is weak on common-word questions
("Is the pay any good?") and strong on rare ones ("CGPA"). MIN_RETRIEVAL_SCORE
is calibrated for cosine and is meaningless against these scores; the abstain
gate is left untouched here.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from src.chunking.base import render_embedding_text
from src.config import CONFIG, Config
from src.retrieval.base import ScoredChunk
from src.storage.database import Database

# Tokens are runs of letters/digits, optionally joined by . _ - or / so that
# "gpt-4", "test.user" and "2026-08-21" survive as single terms. A leading "@"
# or "#" is dropped, which makes "@sample7handle" and "sample7handle" match --
# handles are written both ways.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")

# Deliberately small. BM25's IDF already discounts ubiquitous words, and an
# aggressive stopword list is how "What is my gmail password?" loses the terms
# that make it answerable. These are the ones that add nothing but noise.
_STOPWORDS = frozenset("""
a an the of to in on at for and or is are was were be been being do does did
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, split into terms, drop a few stopwords.

    Shared by indexing and querying — they must tokenize identically or terms
    silently fail to match.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Search:
    """Lexical searcher satisfying ``ChunkSearcher``.

    Chunk ids are preserved end to end, so results line up with the vector
    searcher's for direct comparison.
    """

    def __init__(
        self,
        db: Database,
        chunk_ids: Sequence[str],
        corpus_tokens: Sequence[Sequence[str]],
        config: Config = CONFIG,
    ) -> None:
        self.db = db
        self.config = config
        self.chunk_ids = list(chunk_ids)
        self._tokens = [list(t) for t in corpus_tokens]
        self._bm25 = None
        if self._tokens:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokens)

    @classmethod
    def open(cls, db: Database, config: Config = CONFIG) -> "BM25Search":
        """Build the index from the chunks table.

        Indexes the same *embedding view* the dense retriever uses: without
        stripping them, each chunk contributes a dozen ``[m:<16-hex>]`` tags,
        which bloat the term space and skew BM25's length normalisation.
        """
        chunks = db.all_chunks()
        ids = [c.chunk_id for c in chunks]
        tokens = [tokenize(render_embedding_text(c.text)) for c in chunks]
        return cls(db, ids, tokens, config)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(
        self,
        query: str,
        k: int,
        candidate_chunk_ids: Optional[set[str]] = None,
    ) -> list[ScoredChunk]:
        if self._bm25 is None or not query.strip():
            return []
        terms = tokenize(query)
        if not terms:                      # e.g. "???" or pure stopwords
            return []
        if candidate_chunk_ids is not None and not candidate_chunk_ids:
            return []

        scores = self._bm25.get_scores(terms)

        ranked = [
            (cid, float(score))
            for cid, score in zip(self.chunk_ids, scores)
            if score > 0.0
            and (candidate_chunk_ids is None or cid in candidate_chunk_ids)
        ]
        # Sort by score, then chunk_id: ties would otherwise resolve by corpus
        # order, which changes whenever chunks are re-inserted. Ranking must be
        # deterministic for the benchmark to mean anything.
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))

        out: list[ScoredChunk] = []
        for cid, score in ranked[:k]:
            chunk = self.db.get_chunk(cid)
            if chunk is not None:
                out.append(ScoredChunk(chunk=chunk, score=score))
        return out
