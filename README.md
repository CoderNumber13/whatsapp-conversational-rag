# Conversation Memory RAG

A privacy-first, local-first system for asking natural-language questions about
your own WhatsApp conversation history. WhatsApp `.txt` exports go in; grounded,
citation-backed answers come out. WhatsApp is just the first ingestion source —
the pipeline works on a normalized, source-independent message schema.

> **Status: Phase 1 complete (MVP).** Parser → normalized schema → SQLite →
> conversation-aware chunking → embeddings → FAISS → retrieval (+ metadata
> filters + context reconstruction) → grounded LLM answers with per-message
> citations → Streamlit UI. Phase 2 (BM25 / hybrid / reranking) and Phase 3
> (knowledge graph) are next.

## Privacy

- Real chat data is **never** committed. `data/private/` and loose `.txt` drops
  are git-ignored; only the synthetic set under `data/sample/` is tracked.
- No WhatsApp scraping, no account access — only user-exported `.txt` files.
- LLM provider is configurable (`ollama` / `openai` / `mock`); no keys in source.
- The FAISS index is derived data: it can always be rebuilt from SQLite.

## Setup

Requires **Python 3.12** (3.11 also fine). The installed 3.14 lacks ML wheels.

```bash
# 1. environment
conda create -n convmem python=3.12 -y
conda activate convmem
pip install -r requirements.txt

# 2. local LLM
#    install Ollama for Windows, then:
ollama pull llama3.1:8b
#    Low RAM (Ollama "failed to allocate buffer"): set OLLAMA_NUM_CTX=2048 in
#      .env, or use a small model: ollama pull llama3.2:3b  (OLLAMA_MODEL=llama3.2:3b)
#    CUDA runner crash ("shared object initialization failed" / exit 0xc0000409,
#      old NVIDIA driver vs Ollama's bundled CUDA): set OLLAMA_NUM_GPU=0 in .env
#      to force CPU, or update the NVIDIA driver.

# 3. config
cp .env.example .env        # then edit ME_NAMES to your WhatsApp display name(s)

# 4. tests  (uses a mock embedder + mock LLM — no model needed)
pytest

# 5. run the app
streamlit run app.py
```

In the app: click **Load sample** (synthetic chats) or upload your own `.txt`
exports, then ask questions. Every answer shows its **Sources** (the exact
messages cited) and a **Retrieval inspection** panel (each retrieved chunk with
its score and whether it was cited).

### Use it from Python

```python
from src.pipeline.rag_pipeline import RagPipeline
from src.retrieval.retriever import RetrievalFilters

pipe = RagPipeline()
pipe.ingest(["path/to/WhatsApp Chat with Rahul.txt"], me_names=["Me"])

ans = pipe.answer("What did Rahul tell me about his internship?")
print(ans.text, ans.supported)
for c in ans.citations:
    print(c.timestamp, c.sender, c.text)

# with metadata filters
pipe.answer("what did we decide?", filters=RetrievalFilters(sender="Rahul"))
```

`answer()` **abstains without calling the LLM** when nothing is retrieved above
`MIN_RETRIEVAL_SCORE`; when the model can't answer from the excerpts it returns
`supported=False`; citations the model invents are dropped.

## Project layout

```
src/
  ingestion/   base.py (ChatSource) · whatsapp_parser.py · normalizer.py
  storage/     models.py (normalized schema) · database.py (SQLite = source of truth)
  chunking/    base.py · message_chunker.py · time_chunker.py · service.py
  embeddings/  base.py · sentence_transformer.py · mock.py · factory.py
  retrieval/   vector_store.py (FAISS) · indexer.py · vector_search.py · retriever.py
  llm/         base.py · ollama_client.py · openai_client.py · mock.py · prompts.py
  pipeline/    rag_pipeline.py
  graph/ query/ agent/ evaluation/     ← Phase 3+ (scaffolded)
app.py                                 ← Streamlit UI
scripts/generate_synthetic_chats.py
tests/                                 ← 127 tests
```

## Pipeline

```
.txt export
  → whatsapp_parser (format only)
  → normalizer (identity, stable ids, ordering, dedup)
  → SQLite  (authoritative: messages, conversations, chunks, embedding_meta)
  → ChunkingService     conversation-bounded windows, [m:<id>]-tagged text
  → EmbeddingIndexer    all-MiniLM-L6-v2 → FAISS (IndexIDMap2/IP), staleness by content_hash
  → Retriever           metadata filter → vector search → context window (§14)
  → RagPipeline.answer   abstain-if-weak → grounded prompt → LLM → validated citations
```

### Design notes

- **`ME_NAMES`** — WhatsApp labels your own messages with your profile/contact
  name, which varies between exports. Set it so "what did *I* promise" works.
- **Date order** — `DD/MM` vs `MM/DD` is inferred per file (looking for a day
  > 12); genuinely ambiguous files default to day-first and are flagged.
- **Timestamps** are stored naive/local — WhatsApp exports carry no timezone.
- **`message_id`** = hash of `(conversation_id, timestamp, sender, text, occurrence)`.
  The `occurrence` index keeps genuinely repeated short messages ("ok"/"ok" in
  one Android minute) distinct; identical re-exports still dedupe.
- **`seq`** is a DB-owned cache, recomputed across all imports after every
  ingest — so re-uploading an export with older history prepended stays ordered.
- **`chunk_id`** derives from its first/last member message ids (+ chunker
  version), never from `seq`, so re-ingest only re-embeds genuinely changed chunks.
- **`content_hash` / `embedding_model`** on every chunk drive incremental
  re-embedding; a chunk keeps one `faiss_id` for life.
- **Rebuild:** delete `data/private/index/` and call `EmbeddingIndexer(db).rebuild()`
  (or just re-ingest) — answers are unchanged.

## Configuration

All via environment / `.env` (see `.env.example`). Key knobs: `EMBEDDING_MODEL`
(`mock` skips the download), `LLM_PROVIDER` (`ollama` | `openai` | `mock`),
`OLLAMA_NUM_CTX`, `CHUNK_STRATEGY` (`fixed_count` | `time_window`),
`CHUNK_SIZE_MESSAGES` / `CHUNK_OVERLAP_MESSAGES`, `RETRIEVAL_TOP_K`,
`MIN_RETRIEVAL_SCORE`, `CONTEXT_WINDOW_MESSAGES`.
