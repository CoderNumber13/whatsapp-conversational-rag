"""Retrieval strictness: how much evidence the stack gathers before answering.

A user asking about a credential, a rare name or a product code is asking about
something with weak semantic similarity to anything. The benchmark shows this
directly: ``exact_term`` questions score 0.0% Recall@1 under dense retrieval,
and the credential questions are not reachable at all. Giving the user a way to
say "look harder" is reasonable.

What this control must NOT be
-----------------------------
It is **not** a confidence threshold, and it deliberately exposes no number that
looks like one. The four stages produce four incomparable score scales:

===============  =========================================  ====================
stage            score                                      comparable to?
===============  =========================================  ====================
vector           cosine similarity, bounded [-1, 1]         ``MIN_RETRIEVAL_SCORE``
BM25             unbounded corpus-relative sum (to ~20.8)   nothing else
RRF              a rank-fusion artifact, ~0.016-0.03        nothing at all
cross-encoder    unbounded logit, ~[-11, +9], mostly < 0    nothing else
===============  =========================================  ====================

`MIN_RETRIEVAL_SCORE` (0.25) is a **cosine** floor. Comparing it against an RRF
value would abstain on everything; against a cross-encoder logit, likewise.
Experiments 5 and 6 further showed that no cross-encoder threshold — absolute or
margin-based — separates answerable from unanswerable questions well enough to
gate on. So strictness moves **candidate depth**, which is scale-free, and never
a threshold.

What it does move
-----------------
Three depths, each a count of chunks:

``rrf_candidates``
    how deep the two retrievers are fused before ranking. Counter-intuitively,
    *smaller* helps rare evidence: RRF rewards agreement, so a deeper pool
    supplies more two-retriever chunks that outrank anything only one retriever
    can see.
``rerank_candidates``
    how many fused candidates the cross-encoder actually reads.
``evidence_chunks``
    how many reranked chunks are handed to the grounded prompt.

Widening these can only *add* evidence to the context. It cannot make the model
answer: the grounded prompt still refuses unless an excerpt explicitly supports
the answer, and rule 6 still forbids presenting a nearby handle, address, code
or URL as a credential. Permissive changes what the model gets to read, never
what it is allowed to claim.

Balanced is derived from the live config rather than hardcoded, so it *is* the
production default by construction and cannot drift away from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import CONFIG, Config

STRICT = "strict"
BALANCED = "balanced"
PERMISSIVE = "permissive"
MODES = (STRICT, BALANCED, PERMISSIVE)

DEFAULT_MODE = BALANCED


@dataclass(frozen=True)
class StrictnessProfile:
    """Candidate depths for one mode. Every field is a count of chunks."""

    name: str
    rrf_candidates: int
    rerank_candidates: int
    evidence_chunks: int
    description: str = ""

    def __post_init__(self) -> None:
        for field_name in ("rrf_candidates", "rerank_candidates", "evidence_chunks"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be >= 1")


# Permissive uses a SHALLOWER fusion pool, which looks backwards and is not.
#
# Measured (docs/BASELINE.md, experiment 7): widening the RRF pool makes the
# credential evidence *harder* to reach, not easier. RRF scores agreement, so a
# deeper pool produces more chunks that both retrievers return, and every one of
# them outranks a chunk only one retriever can see. The credential chunk is
# exactly that: visible to the dense retriever alone. Deepening the pool to 100
# left X1 at rank 26 and X3 unreachable; shrinking it to 10 brought all three
# into reach (X1 14, X2 8, X3 13) while *improving* MRR and halving latency.
#
# So permissiveness is "how much evidence reaches the model", and the pool depth
# is set to serve that rather than raised for its own sake.
#
# Pool depths are absolute for permissive because the relationship is not
# monotonic — scaling the configured value would encode the wrong intuition.
_POOL_DEPTHS = {
    #            rrf pool,        rerank depth
    STRICT: (None, None),        # production pools; strictness is in the evidence
    BALANCED: (None, None),      # None => take the configured production value
    PERMISSIVE: (10, 10),        # measured; see above
}

# Evidence chunks handed to the grounded prompt, relative to the configured
# MAX_CONTEXT_CHUNKS. This is the axis the control is really about.
_EVIDENCE_SCALING = {STRICT: 0.6, BALANCED: 1.0, PERMISSIVE: 3.0}

_DESCRIPTIONS = {
    STRICT: "Fewer candidates. Fastest, and least likely to surface a "
            "marginal chunk. Best when questions are well-phrased.",
    BALANCED: "The production default, and the configuration every published "
              "benchmark number was measured with.",
    PERMISSIVE: "Passes far more evidence to the model, and fuses a shallower "
                "pool so a chunk only one retriever found is not buried by "
                "consensus. Use for credentials, rare names and codes. The "
                "grounding and abstention rules are unchanged.",
}


def profile_for(mode: str = DEFAULT_MODE, config: Config = CONFIG) -> StrictnessProfile:
    """Resolve a mode to concrete depths against the live configuration."""
    key = (mode or DEFAULT_MODE).strip().lower()
    if key not in MODES:
        raise ValueError(f"unknown strictness {mode!r} (want one of {MODES})")

    rrf_pool, rerank_pool = _POOL_DEPTHS[key]
    return StrictnessProfile(
        name=key,
        rrf_candidates=rrf_pool if rrf_pool is not None else config.rrf_candidates,
        rerank_candidates=(rerank_pool if rerank_pool is not None
                           else config.rerank_candidates),
        evidence_chunks=max(1, round(config.max_context_chunks
                                     * _EVIDENCE_SCALING[key])),
        description=_DESCRIPTIONS[key],
    )


def all_profiles(config: Config = CONFIG) -> dict[str, StrictnessProfile]:
    return {m: profile_for(m, config) for m in MODES}
