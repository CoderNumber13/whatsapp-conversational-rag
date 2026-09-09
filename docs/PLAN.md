# Implementation Plan — Conversation Memory RAG

The canonical spec for this project. `README.md` describes what *exists*; this
file describes what is *intended* and in what order. Keep the status markers
current — they are the single source of truth for "what's next".

Status legend: **DONE** · **PARTIAL** · **TODO**

---

## 1. Objective

A privacy-first, local-first system for asking natural-language questions about
your own WhatsApp conversation history. Exports in → grounded, citation-backed
answers out.

Long-term: evolve from a static WhatsApp RAG into a continuously updating
personal conversational-memory engine.

**Hard constraint:** no WhatsApp scraping, no programmatic account access. The
only ingestion source is the user's own exported `.txt` files.

## 2. Final goal

More than a chatbot — a **privacy-first Conversational Memory Engine** turning
raw conversational data into searchable memory, semantic + temporal retrieval,
entity/relationship knowledge, conversation reconstruction, evidence-backed
answers, and eventually agentic personal memory.

The first public version demonstrates the full pipeline on synthetic data
without exposing anyone's private conversations.

---

## 3. Phase status

| Phase | Scope | Status |
|---|---|---|
| **1 — MVP** | parser → schema → SQLite → chunking → embeddings → vector search → LLM answers → citations → Streamlit UI | **DONE** |
| **2 — Better retrieval** | conversation-aware chunking, metadata filtering, keyword/BM25, hybrid retrieval, reranking, context reconstruction | **DONE** (semantic chunking deferred) |
| **3 — Graph RAG** | entity + relationship extraction, graph construction, community detection, community summaries, local + global graph retrieval | **TODO** |
| **4 — Agent** | query understanding, retrieval/conversation tools, agent planning, multi-step retrieval | **TODO** |
| **5 — Evaluation** | benchmark questions, baseline, Recall@K, strategy comparison, faithfulness, latency | **PARTIAL** — retrieval baseline done, see [BASELINE.md](BASELINE.md) |
| **6 — Productionization** | privacy controls, config, logging, error handling, Docker, docs, synthetic dataset, clean git | **PARTIAL** |

## Status as of 2026-09-09 (commit after f04efd9)

### Complete

- **Phase 1** end to end, plus the Phase 2 retrieval stack:
  `vector + BM25 → RRF → cross-encoder → top-N chunks → Gemini → cited answer`.
- A 44-question benchmark on a 1269-message / 144-chunk synthetic corpus, with
  every stage measured against it ([BASELINE.md](BASELINE.md)).
- **End-to-end grounded answering.** On the measured subset: grounded-answer
  accuracy 100%, citation correctness 100%, **fabrication rate 0%**.
- **The Streamlit demo now runs the stack the experiments measured.** It
  previously ran vector-only, so the demo and the evidence disagreed.
- Production OpenMP mitigation, in one documented place (`src/runtime.py`).

### Remaining

- Semantic chunking (Phase 2's last item) — deferred, never measured.
- Phase 3 knowledge graph, Phase 4 agent — not started.
- Phase 5: retrieval evaluation is done; **answer** evaluation is only partially
  run (see the quota limitation below).
- Phase 6: Docker, and the private-data cleanup listed below.

### Known limitations

**The Gmail credential case (X1/X2/X3) is unsolved and is not a ranking bug.**
The credential is a bare token that nothing in the corpus identifies as a
password, so it shares no term or meaning with any phrasing of the question.
Dense retrieval, BM25, RRF and a cross-encoder each fail on it independently,
and RRF and the cross-encoder each rank it *lower* than vector search alone did.
There is no evidence in the corpus to rank on. **Treated as a permanent
limitation of retrieval, not something to tune around.**

What *is* fixed is the consequence: the system now **refuses instead of
fabricating**. X1/X2/X3 abstain, and none states the credential. Earlier, at a
loosened retrieval floor, the model volunteered *"Likely: the password is …"* by
inferring a nearby handle; system-prompt rule 6 forbids presenting any value as
a credential unless an excerpt says in words that it is one, and forbids hedging
such a guess.

**Cross-encoder scores are not globally calibrated.** They rank well within a
query but their magnitude tracks phrasing as much as evidence quality. At a
threshold with zero false acceptances, 37% of answerable questions are wrongly
refused — including P8, whose *correct* chunk at rank 1 scores below every
question that has no answer. Margins (top1−top2, top1−top3) were tested as an
alternative and are not better. **Therefore no cross-encoder threshold exists in
the pipeline**, and `MIN_RETRIEVAL_SCORE` (a cosine value, 0.25) must never be
compared against a cross-encoder logit. Abstention is the grounded prompt's
decision. P5 is the proof this matters: its evidence sits at rank 4 with a score
of −10.81 and it is answered correctly and cited.

**FAISS/torch OpenMP conflict.** `faiss-cpu` (PyPI) links LLVM's OpenMP runtime,
torch links Intel's. The second to initialise aborts the process with exit code
3 and no traceback. Import ordering, eager loading, `omp_set_num_threads(1)` and
restricting to single-query search were each measured and none avoids it; only
`KMP_DUPLICATE_LIB_OK=TRUE` does. It is therefore set in exactly one place,
`src/runtime.py`, before the libraries load, with the evidence recorded there.
**The proper fix is environmental** — `conda install -c pytorch faiss-cpu` links
the same OpenMP as torch and removes the conflict entirely; the guard then
becomes a no-op. A regression test asserts both that the workload survives with
the guard and that it still fails without it, so the workaround can be deleted
the day it stops being necessary.

**LLM quota.** Gemini's free tier allows 20 requests per day *per model*, so the
full 44-question end-to-end benchmark cannot be run in one sitting. Results are
from subsets; the harness records quota failures explicitly rather than scoring
them as abstentions.

### Phase 2 breakdown

| Item | Status | Where |
|---|---|---|
| Conversation-aware chunking | **DONE** | `src/chunking/` (`fixed_count`, `time_window`) |
| Metadata filtering | **DONE** | `RetrievalFilters` in `src/retrieval/retriever.py` |
| Conversation-context reconstruction | **DONE** | `CONTEXT_WINDOW_MESSAGES` |
| Semantic chunker (3rd strategy) | **TODO** | `src/chunking/semantic_chunker.py` |
| Keyword / BM25 search | **TODO** | `src/retrieval/keyword_search.py` |
| Hybrid retrieval (fuse vector + keyword + metadata) | **TODO** | `src/retrieval/hybrid_search.py` |
| Reranking layer | **TODO** | `src/retrieval/reranker.py` |

Hybrid retrieval matters because vector search alone is weak on names, exact
dates, project names, company names, and specific phrases.

**Measured evidence (2026-09-07 investigation).** On the real 141-message
corpus, asking for a credential that *is* present returned "not found". Two
distinct causes, one fixed and one deferred:

1. *Fixed.* Chunks were embedded from their display text, so `[m:<id>]` tags and
   per-line ISO timestamps made up ~60% of every vector's input and pushed 13 of
   15 chunks past all-MiniLM-L6-v2's 256-token limit. One gmail chunk had the
   word "gmail" truncated away entirely. Embedding the stripped view raised the
   target chunk 0.1806 → 0.2391 and cut truncation to 1 of 15.
2. *Deferred to Phase 2.* The credential is an unlabelled bare token — nothing
   in the chat says "password". It shares no term with any natural phrasing of
   the question, so cosine similarity cannot reach it at any threshold. **This
   is the acceptance test for hybrid retrieval:** on the real corpus, "what is
   my gmail password" must surface the chunk holding that token.

Also measured: `MIN_RETRIEVAL_SCORE=0.25` is not a valid relevance gate here. An
irrelevant query ("trip to Japan") scored **0.3307** while two answerable gmail
questions scored **0.2253** and **0.2391** — the floor abstains on real matches
and admits irrelevant ones. Lowering it to 0.10 made the LLM assert a password
by inferring one from an adjacent token. Leave the floor alone until hybrid
retrieval + reranking give scores worth thresholding on.

Reranking is a distinct modular stage so approaches can be swapped:
`query → vector + keyword retrieval → candidate pool → reranker → top chunks`.

---

## 4. Architecture

```
WhatsApp TXT exports
  → Chat Parser
  → Normalized Message Schema
  → Conversation Store (SQLite, authoritative)
  → Conversation-aware Chunking
  → Embedding Generation
  → Vector Index
  → Metadata / Structured DB
  → Hybrid Retrieval        ← Phase 2
  → Reranking               ← Phase 2
  → Context Construction
  → LLM
  → Answer + Evidence/Citations
```

WhatsApp is only *one* ingestion source. Future: Discord, Telegram, Slack,
generic JSON/CSV. Downstream retrieval code must never depend on the original
WhatsApp text format.

### Ingestion interface

```python
class ChatSource:
    def ingest(self): ...
    def get_messages(self): ...
```

Implementations: `WhatsAppExportSource` (now), `DiscordSource`,
`GenericJSONSource` (later). Adding a source must not require modifying the
core RAG pipeline.

Future continuous ingestion is event-oriented:
`new message → normalize → store → detect affected chunk → update embedding →
update index → update graph`.

### Normalized message

`message_id`, `conversation_id`, `conversation_name`, `timestamp`, `sender`,
`text`, `source`, optional media metadata.

---

## 5. Component requirements

### Parser
Handles both `[08/25/26, 9:14 PM] Rahul: ...` and `25/08/26, 21:14 - Rahul: ...`.
Must survive: multiline messages, differing date formats, senderless and system
messages, emoji/Unicode, media placeholders, commas and colons in text, names
with spaces. Unit-tested.

### Storage
SQLite is authoritative — **the vector DB is never the only source of truth.**
Must efficiently answer: all messages from a sender; all messages in a
conversation; messages between two dates; messages containing a term;
chronological reconstruction.

### Chunking
Never embed individual messages blindly — "yeah", "do it", "tomorrow" carry
meaning only in context. Document at least three strategies (fixed-count,
time-window, semantic). A chunk retains conversation id, participants, start
and end timestamps, ordered messages, text representation. Ordering is never
destroyed.

### Embeddings
Behind an interface (`embed_text`, `embed_documents`) so models can be swapped.
Never regenerate embeddings for unchanged chunks.

### Knowledge graph (Phase 3)
Entities: people, organizations, companies, places, projects, events,
technologies, dates. Relations: `discussed`, `works_at`, `told`, `attended`.
NetworkX initially. Connects entities ↔ messages/chunks ↔ conversations.
**Do not assume every extracted entity or relationship is correct — store
provenance linking every graph fact back to its source messages.**

Community detection (Louvain/Leiden) groups entities into themes (placements,
college, projects, gaming, travel). Community summaries are LLM-generated and
must store the source relationships/chunks used to produce them.

Retrieval splits into **local** (specific fact lookup via entities) and
**global** (broad thematic questions via community summaries). The system must
distinguish the two.

### Query understanding (Phase 4)
Classify whether the question is about a person, conversation, date/time, topic,
specific fact, broad summary, relationship, decision, commitment, or an opinion
changing over time — and extract deterministic filters.
**Do not use an LLM for simple deterministic filters that can be extracted
reliably.**

### Agent (Phase 4)
Tools: `search_messages`, `search_person`, `search_conversation`,
`search_date_range`, `search_topic`, `get_conversation_context`,
`search_entities`, `get_entity_relationships`.
**Not to be built before basic retrieval is reliable.**

### Answer generation
The prompt must instruct the model to answer only from retrieved evidence, not
invent facts, state when evidence is insufficient, distinguish direct statements
from inference, and cite source messages. The system must separate directly
supported facts, reasonable summaries, and not-found. It must not expose private
data unrelated to the question.

### Citations
Every claim traceable to source messages via stable identifiers
(`conversation_id`, `message_id`, `timestamp`). The UI lets the user inspect
supporting messages.

### Evaluation (Phase 5)
50–100 manually curated questions covering fact / person / date / topic
retrieval, multi-hop, summarization, decision tracking, temporal questions, each
with expected source messages.

- Retrieval: Recall@1/5/10, MRR
- Generation: faithfulness, answer relevance, citation correctness
- System: ingestion time, embedding time, retrieval latency, total latency, storage

Compare: baseline vector → +metadata → hybrid → hybrid+rerank → graph+hybrid.
**Do not invent benchmark numbers.**

### UI
Streamlit. Ingestion view (message/conversation counts, date range,
participants), chat view (question → answer → sources), and a retrieval
inspection panel (chunks, scores, metadata, source messages) for debugging and
demonstrating the system.

---

## 6. Privacy requirements

- Real conversations are **never** committed. `.gitignore`, private data dirs,
  synthetic data for demos.
- Local-first processing.
- Provider-swappable LLM (local / OpenAI / other) — never hard-code one provider
  throughout the codebase.
- **No API keys in source.** Environment variables and config files only;
  `.env` is git-ignored and only `.env.example` (placeholders) is tracked.

## 7. Technology stack

Python · SQLite + JSON · sentence-transformers, spaCy, regex, optionally NLTK ·
FAISS or Chroma · NetworkX (+ Leiden/Louvain) · Ollama/local model with hosted
API support · Streamlit · pytest · git · Docker later.

Avoid frameworks before understanding the underlying components. LangChain /
LlamaIndex must **not** be a mandatory dependency of every component — prefer
modular Python components with clear interfaces.

## 8. Engineering principles

Prioritize correctness over feature count.

Do **not**: scrape WhatsApp · store real chats in git · hard-code API keys ·
send whole chat history to an LLM · rely solely on vector similarity · build the
agent before retrieval works · use a framework abstraction you don't understand.

Every retrieval result has provenance. Every answer is grounded in retrieved
evidence. Every major component is independently testable. Parameters are
configurable. Architectural decisions are documented.
