"""End-to-end pipeline: ingest exports, then answer questions with citations.

    ingest(paths)  -> parse -> normalize -> SQLite -> chunk -> embed
    answer(q)      -> retrieve -> (abstain if weak) -> LLM -> validate citations

Abstention is enforced before any LLM call: if nothing is retrieved or the top
cosine score is below ``MIN_RETRIEVAL_SCORE`` the pipeline returns a
"not found" answer without spending a model call. Citations returned by the
model are validated against the retrieved context; unknown ids are dropped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from src.chunking.service import ChunkingService, ChunkSyncResult
from src.config import CONFIG, Config
from src.ingestion.normalizer import WhatsAppExportSource
from src.llm.base import LLMClient
from src.llm.factory import get_llm
from src.llm.prompts import NOT_FOUND_TOKEN, SYSTEM_PROMPT, build_user_prompt
from src.retrieval.indexer import EmbeddingIndexer, IndexSyncResult
from src.retrieval.retriever import RetrievalFilters, RetrievedChunk, Retriever
from src.storage.database import Database

logger = logging.getLogger("conversation_memory.pipeline")

_CITE_RE = re.compile(r"\[m:([0-9a-f]+)\]")


@dataclass
class IngestReport:
    files: int = 0
    messages_added: int = 0
    conversations: int = 0
    chunks: ChunkSyncResult = field(default_factory=ChunkSyncResult)
    index: IndexSyncResult = field(default_factory=IndexSyncResult)


@dataclass
class Citation:
    message_id: str
    conversation_id: str
    conversation_name: str
    sender: str
    timestamp: datetime
    text: str
    src_file: str
    src_line_start: int
    src_line_end: int


@dataclass
class Answer:
    text: str
    supported: bool
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    llm_model: str
    top_score: float
    abstained: bool = False


class RagPipeline:
    def __init__(self, config: Config = CONFIG, *, db: Optional[Database] = None,
                 llm: Optional[LLMClient] = None) -> None:
        self.config = config
        self.db = db or Database(config.db_path)
        self._llm = llm
        self._retriever: Optional[Retriever] = None

    # --- ingest -----------------------------------------------------
    def ingest(
        self, paths: Sequence[str | Path], me_names: Optional[Sequence[str]] = None
    ) -> IngestReport:
        me = list(me_names) if me_names is not None else self.config.me_names
        src = WhatsAppExportSource(paths, me_names=me)
        src.ingest()
        convs = src.get_conversations()
        for conv in convs:
            self.db.upsert_conversation(conv, me_names=me)
        added = self.db.insert_messages(src.get_messages())

        chunks = ChunkingService(self.db, config=self.config).sync_all()
        index = EmbeddingIndexer(self.db, config=self.config).sync()

        self._retriever = None  # index changed; rebuild lazily
        return IngestReport(
            files=len(list(paths)) if not isinstance(paths, (str, Path)) else 1,
            messages_added=added,
            conversations=len(convs),
            chunks=chunks,
            index=index,
        )

    # --- answer -----------------------------------------------------
    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever.open(self.db, self.config)
        return self._retriever

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm(self.config)
        return self._llm

    def answer(
        self,
        question: str,
        filters: Optional[RetrievalFilters] = None,
        *,
        k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Answer:
        k = k or self.config.retrieval_top_k
        floor = self.config.min_retrieval_score if min_score is None else min_score
        retrieved = self.retriever.retrieve(question, k=k, filters=filters)
        top = retrieved[0].score if retrieved else 0.0
        logger.debug("retrieved %d chunks, top score %.3f", len(retrieved), top)

        if not retrieved or top < floor:
            return Answer(
                text="I couldn't find anything about that in your chats.",
                supported=False,
                citations=[],
                retrieved=retrieved,
                llm_model="(none)",
                top_score=top,
                abstained=True,
            )

        valid_ids = self._context_ids(retrieved)
        filters_note = _filters_note(filters)
        user_prompt = build_user_prompt(question, retrieved, filters_note=filters_note)
        resp = self.llm.complete(SYSTEM_PROMPT, user_prompt, max_tokens=self.config.llm_max_tokens)

        if resp.text.strip().rstrip(".").upper() == NOT_FOUND_TOKEN:
            return Answer(
                text="I couldn't find anything about that in your chats.",
                supported=False,
                citations=[],
                retrieved=retrieved,
                llm_model=resp.model,
                top_score=top,
            )

        citations = self._collect_citations(resp.text, valid_ids)
        return Answer(
            text=resp.text.strip(),
            supported=bool(citations),
            citations=citations,
            retrieved=retrieved,
            llm_model=resp.model,
            top_score=top,
        )

    # --- helpers -------------------------------------------------
    @staticmethod
    def _context_ids(retrieved: list[RetrievedChunk]) -> set[str]:
        ids: set[str] = set()
        for rc in retrieved:
            ids.update(m.message_id for m in rc.ordered_messages())
        return ids

    def _collect_citations(self, text: str, valid_ids: set[str]) -> list[Citation]:
        out: list[Citation] = []
        seen: set[str] = set()
        for mid in _CITE_RE.findall(text):
            if mid in seen or mid not in valid_ids:
                continue
            seen.add(mid)
            row = self.db.get_message(mid)
            if row is None:
                continue
            conv = self.db.get_conversation(row["conversation_id"])
            out.append(
                Citation(
                    message_id=mid,
                    conversation_id=row["conversation_id"],
                    conversation_name=conv["name"] if conv else "",
                    sender=row["sender"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    text=row["text"],
                    src_file=row["src_file"] or "",
                    src_line_start=row["src_line_start"] or 0,
                    src_line_end=row["src_line_end"] or 0,
                )
            )
        return out

    def close(self) -> None:
        self.db.close()


def _filters_note(filters: Optional[RetrievalFilters]) -> str:
    if not filters:
        return ""
    bits = []
    for k, v in filters.as_kwargs().items():
        bits.append(f"{k}={v:%Y-%m-%d}" if isinstance(v, datetime) else f"{k}={v}")
    return ", ".join(bits)
