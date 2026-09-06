# Conversation Memory RAG

A privacy-first, local-first system for asking natural-language questions about
your own WhatsApp conversation history. WhatsApp `.txt` exports go in; grounded,
citation-backed answers come out. WhatsApp is just the first ingestion source —
the pipeline works on a normalized, source-independent message schema.

> **Status: Phase 1, increment 1** — WhatsApp parser, normalized schema, SQLite
> store, synthetic data, tests. Embeddings, vector search, the LLM answer layer
> and the Streamlit UI land in the next increment.

## Privacy

- Real chat data is **never** committed. `data/private/` and loose `.txt` drops
  are git-ignored; only the synthetic set under `data/sample/` is tracked.
- No WhatsApp scraping, no account access — only user-exported `.txt` files.
- LLM provider is configurable (`ollama` / `openai` / `mock`); no keys in source.

## Setup

Requires **Python 3.12** (3.11 also fine). The installed 3.14 lacks ML wheels.

```bash
# 1. environment
conda create -n convmem python=3.12 -y
conda activate convmem
pip install -r requirements.txt

# 2. local LLM (used from increment 2)
#    install Ollama for Windows, then:
ollama pull llama3.1:8b

# 3. config
cp .env.example .env        # then edit ME_NAMES to your WhatsApp display name(s)

# 4. synthetic sample data
python scripts/generate_synthetic_chats.py

# 5. tests
pytest
```

## Project layout

```
src/
  ingestion/   base.py (ChatSource) · whatsapp_parser.py · normalizer.py
  storage/     models.py (normalized schema) · database.py (SQLite, source of truth)
  chunking/    embeddings/ retrieval/ graph/ query/ llm/ agent/ evaluation/ pipeline/
               ^ scaffolded, populated in later increments
scripts/generate_synthetic_chats.py
tests/
```

## How a WhatsApp export becomes memory

```
.txt export → whatsapp_parser (format only) → normalizer (identity, ids,
ordering, dedup) → Message objects → SQLite (authoritative)
→ [next] conversation-aware chunks → embeddings → FAISS
→ [next] hybrid retrieval → LLM → answer + message-level citations
```

### Design notes

- **`ME_NAMES`** — WhatsApp labels your own messages with your profile/contact
  name, which varies between exports. Set it so "what did *I* promise" works.
- **Date order** — `DD/MM` vs `MM/DD` is inferred per file (looking for a day
  > 12); genuinely ambiguous files default to day-first and are flagged.
- **Timestamps** are stored naive/local — WhatsApp exports carry no timezone.
- **`content_hash` / `embedding_model`** columns exist from the first schema so
  re-embedding can be skipped when a chunk is unchanged.
- **`message_id`** = hash of `(conversation_id, timestamp, sender, text)` — this
  is also the cross-export dedup key.
