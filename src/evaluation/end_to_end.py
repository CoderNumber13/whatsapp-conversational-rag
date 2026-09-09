"""End-to-end grounded answering evaluation.

Measures the whole pipeline — vector + BM25 + RRF + cross-encoder + LLM — rather
than retrieval alone.

The success criterion is deliberately strict, and deliberately not "the answer
contains the right string". A response counts as a grounded success only when
all three hold:

    1. the required evidence was actually retrieved into the context,
    2. the answer states the gold fact,
    3. the answer carries at least one citation, and its citations point at
       messages that were in the supplied context.

An answer can satisfy (2) alone by guessing — that is precisely the failure the
credential questions exist to catch — so (2) is never sufficient on its own.

Questions are partitioned three ways, because "correct" means something
different in each:

    WITH_EVIDENCE     answerable, evidence retrieved  -> should answer, grounded
    WITHOUT_EVIDENCE  answerable, evidence NOT retrieved -> should REFUSE
    ABSENT            no answer exists at all         -> should REFUSE

Answering a WITHOUT_EVIDENCE question is fabrication, not success, however
plausible the text: the pipeline had nothing to ground it on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.pipeline.rag_pipeline import NOT_FOUND_MESSAGE

WITH_EVIDENCE = "with_evidence"
WITHOUT_EVIDENCE = "without_evidence"
ABSENT = "absent"
GROUPS = (WITH_EVIDENCE, WITHOUT_EVIDENCE, ABSENT)


@dataclass
class AnswerObservation:
    qid: str
    category: str
    question: str
    group: str

    retrieved_chunk_ids: list[str]
    evidence_rank: Optional[int]
    evidence_retrieved: bool

    answer_text: str
    abstained: bool
    supported: bool
    cited_message_ids: list[str]
    context_message_ids: set[str] = field(repr=False, default_factory=set)
    expected_message_ids: set[str] = field(repr=False, default_factory=set)

    contains_gold: bool = False
    citations_valid: bool = False
    latency_ms: float = 0.0
    llm_model: str = ""
    # set only when the LLM call failed after retries; such a row is neither a
    # success nor an abstention and is reported separately
    error: Optional[str] = None

    @property
    def cited_evidence(self) -> bool:
        """Did it cite at least one message from the chunk that actually
        answers the question?"""
        return bool(self.expected_message_ids & set(self.cited_message_ids))

    @property
    def grounded_success(self) -> bool:
        """The strict criterion: evidence present, fact stated, properly cited."""
        return (
            self.group == WITH_EVIDENCE
            and not self.abstained
            and self.contains_gold
            and self.citations_valid
            and bool(self.cited_message_ids)
        )

    @property
    def correct_abstention(self) -> bool:
        """Refusing when there was nothing to ground an answer on."""
        return self.group in (WITHOUT_EVIDENCE, ABSENT) and self.abstained

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def fabrication(self) -> bool:
        """Answered a question the pipeline had no evidence for.

        The serious failure: for ABSENT there is nothing in the corpus, and for
        WITHOUT_EVIDENCE nothing relevant reached the context, so any substantive
        answer was invented.
        """
        return (self.group in (WITHOUT_EVIDENCE, ABSENT)
                and not self.abstained and not self.failed)

    @property
    def leaked_gold(self) -> bool:
        """Stated the gold value despite having no evidence for it — the
        credential fabrication case, called out separately because it is the
        most damaging form."""
        return self.group == WITHOUT_EVIDENCE and self.contains_gold

    @property
    def unsupported(self) -> bool:
        """Produced prose but no usable citation, so nothing ties it to the
        conversation."""
        return not self.abstained and not self.failed and not self.cited_message_ids


@dataclass
class GroupReport:
    group: str
    n: int = 0
    answered: int = 0
    abstained: int = 0
    grounded_success: int = 0
    contains_gold: int = 0
    cited_evidence: int = 0
    citations_valid: int = 0
    unsupported: int = 0
    failures: int = 0
    fabrications: int = 0
    leaked_gold: int = 0

    def _rate(self, num: int) -> Optional[float]:
        return (num / self.n) if self.n else None

    @property
    def grounded_answer_rate(self) -> Optional[float]:
        return self._rate(self.answered)

    @property
    def grounded_accuracy(self) -> Optional[float]:
        """Of the questions answered, how many were grounded successes."""
        return (self.grounded_success / self.answered) if self.answered else None

    @property
    def correct_abstention_rate(self) -> Optional[float]:
        return self._rate(self.abstained)

    @property
    def fabrication_rate(self) -> Optional[float]:
        return self._rate(self.fabrications)

    @property
    def unsupported_rate(self) -> Optional[float]:
        return self._rate(self.unsupported)

    @property
    def citation_correctness(self) -> Optional[float]:
        """Of the answers given, how many cited a message that genuinely
        supports the question."""
        return (self.cited_evidence / self.answered) if self.answered else None


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def contains_any(text: str, keys: Sequence[str]) -> bool:
    hay = _norm(text)
    return any(_norm(k) in hay for k in keys if k)


def _answer_with_retry(answerer, question: str, k, attempts: int, on_retry):
    """Hosted models return transient 503s under load. A 44-question run makes
    that near-certain, so retry with backoff rather than losing the whole run —
    and surface the failure instead of silently scoring it as an abstention."""
    from src.llm.base import LLMError

    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return answerer.answer(question, k=k), None
        except LLMError as exc:
            last = exc
            if attempt < attempts:
                on_retry(attempt, exc)
                time.sleep(min(2 ** attempt, 30))
    return None, last


def observe_answers(
    answerer,
    questions: Sequence,
    expected: dict[str, set[str]],
    answer_keys,
    *,
    k: Optional[int] = None,
    attempts: int = 4,
    on_retry=lambda attempt, exc: None,
    pace_s: float = 0.0,
) -> list[AnswerObservation]:
    """Run every question end to end and record what happened.

    ``pace_s`` spaces the calls. Hosted free tiers rate-limit per minute, and a
    44-question burst reliably trips that; pacing turns a run that dies on 429s
    into one that simply takes longer.
    """
    out: list[AnswerObservation] = []
    for i, q in enumerate(questions):
        if pace_s and i:
            time.sleep(pace_s)
        want = expected[q.qid]
        t0 = time.perf_counter()
        ans, error = _answer_with_retry(answerer, q.question, k, attempts, on_retry)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if ans is None:
            # An LLM failure is not an abstention and not a success; record it
            # so it cannot be quietly counted as either.
            out.append(AnswerObservation(
                qid=q.qid, category=q.category, question=q.question,
                group=(ABSENT if q.is_absent else WITHOUT_EVIDENCE),
                retrieved_chunk_ids=[], evidence_rank=None,
                evidence_retrieved=False, answer_text="", abstained=False,
                supported=False, cited_message_ids=[],
                latency_ms=latency_ms, llm_model="(error)",
                error=str(error),
            ))
            continue

        rank = next((i for i, rc in enumerate(ans.retrieved, 1)
                     if want & set(rc.chunk.message_ids)), None)
        context_ids = {m.message_id for rc in ans.retrieved
                       for m in rc.ordered_messages()}
        cited = [c.message_id for c in ans.citations]

        if q.is_absent:
            group = ABSENT
        elif rank is None:
            group = WITHOUT_EVIDENCE
        else:
            group = WITH_EVIDENCE

        out.append(AnswerObservation(
            qid=q.qid, category=q.category, question=q.question, group=group,
            retrieved_chunk_ids=[rc.chunk.chunk_id for rc in ans.retrieved],
            evidence_rank=rank, evidence_retrieved=rank is not None,
            answer_text=ans.text,
            # a refusal is the pipeline's exact refusal text, from either the
            # retrieval gate or the model's NOT_FOUND. Prose with no usable
            # citation is NOT an abstention — it is an unsupported answer.
            abstained=(ans.text.strip() == NOT_FOUND_MESSAGE),
            supported=ans.supported, cited_message_ids=cited,
            context_message_ids=context_ids,
            expected_message_ids=set(want),
            contains_gold=contains_any(ans.text, answer_keys(q.qid)),
            # every citation resolves to a message that was in the context
            citations_valid=bool(cited) and all(c in context_ids for c in cited),
            latency_ms=latency_ms, llm_model=ans.llm_model,
        ))
    return out


def report(observations: Sequence[AnswerObservation]) -> dict[str, GroupReport]:
    reports = {g: GroupReport(group=g) for g in GROUPS}
    for o in observations:
        r = reports[o.group]
        r.n += 1
        if o.failed:
            r.failures += 1
            continue
        if o.abstained:
            r.abstained += 1
        else:
            r.answered += 1
        r.grounded_success += int(o.grounded_success)
        r.contains_gold += int(o.contains_gold)
        r.cited_evidence += int(o.cited_evidence)
        r.citations_valid += int(o.citations_valid)
        r.unsupported += int(o.unsupported)
        r.fabrications += int(o.fabrication)
        r.leaked_gold += int(o.leaked_gold)
    return reports
