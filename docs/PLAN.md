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
| **2 — Better retrieval** | conversation-aware chunking, metadata filtering, keyword/BM25, hybrid retrieval, reranking, context reconstruction | **PARTIAL** |
| **3 — Graph RAG** | entity + relationship extraction, graph construction, community detection, community summaries, local + global graph retrieval | **TODO** |
| **4 — Agent** | query understanding, retrieval/conversation tools, agent planning, multi-step retrieval | **TODO** |
| **5 — Evaluation** | benchmark questions, baseline, Recall@K, strategy comparison, faithfulness, latency | **TODO** |
| **6 — Productionization** | privacy controls, config, logging, error handling, Docker, docs, synthetic dataset, clean git | **PARTIAL** |

### Phase 2 breakdown (current phase)

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
