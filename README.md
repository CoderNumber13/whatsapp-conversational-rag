# Conversation Memory RAG

A privacy-first, local-first system for asking natural-language questions about
your own WhatsApp conversation history. WhatsApp `.txt` exports go in; grounded,
citation-backed answers come out. WhatsApp is just the first ingestion source —
the pipeline works on a normalized, source-independent message schema.

> **Status: Phase 1 + Phase 2 retrieval complete.** Parser → normalized schema → SQLite →
> conversation-aware chunking → embeddings → FAISS → retrieval (+ metadata
> filters + context reconstruction) → grounded LLM answers with per-message
> citations → Streamlit UI, plus BM25 + RRF hybrid retrieval and cross-encoder
> reranking. Phase 3 (knowledge graph) is next.
>
> The system **refuses rather than guesses** when retrieval finds no supporting
> evidence — see the known limitations in [`docs/PLAN.md`](docs/PLAN.md).
>
> Full roadmap and component requirements: [`docs/PLAN.md`](docs/PLAN.md).

## Privacy

- Real chat data is **never** committed. `data/private/` and every loose `.txt`
  are git-ignored. **No chat file of any kind is tracked** — even the synthetic
  sample set is generated on first use by
  `scripts/generate_synthetic_chats.py`, so nothing resembling a private
  conversation ships in the repository.
- No WhatsApp scraping, no account access — only user-exported `.txt` files.
- LLM provider is configurable (`gemini` / `ollama` / `openai` / `mock`); no keys
  in source. Only `.env.example` (placeholders) is tracked — `.env` is ignored.
- The FAISS index is derived data: it can always be rebuilt from SQLite.

## Setup

Requires **Python 3.12** (3.11 also fine). The installed 3.14 lacks ML wheels.

```bash
# 1. environment
conda create -n convmem python=3.12 -y
conda activate convmem          # REQUIRED: base Python 3.14 has no faiss/torch wheels
pip install -r requirements.txt

# 1b. RECOMMENDED: replace pip's faiss with the conda build.
#     faiss-cpu from PyPI links LLVM's OpenMP while torch links Intel's; two
#     OpenMP runtimes in one process abort it (exit 3, no traceback). The app
#     works around this in src/runtime.py, but the clean fix is one runtime:
# pip uninstall -y faiss-cpu && conda install -y -c pytorch faiss-cpu

# 2. config
cp .env.example .env        # then edit ME_NAMES to your WhatsApp display name(s)

# 3. tests  (uses a mock embedder + mock LLM — no model or API key needed)
pytest                      # 445 tests; full inventory in docs/TESTS.md

# 4. run the app
streamlit run app.py        # or, from cmd, just:  run_app.bat
```

**On Windows, prefer `run_app.bat`.** Anaconda's base environment has
`streamlit` but not `faiss`, so a plain `streamlit run app.py` without
activating `convmem` starts the app and then fails at ingestion with
`ModuleNotFoundError: No module named 'faiss'`. Base is Python 3.14, which has
no faiss wheel, so installing it there is not an option. The launcher names the
environment's interpreter directly, so there is nothing to remember; the app
also preflights at startup and tells you which interpreter it is running under
if a required package is missing.

> **Always activate `convmem` first.** The system Anaconda base is Python 3.14,
> which has no `faiss` wheel — ingestion fails with `ModuleNotFoundError: faiss`.

### Choosing an LLM

Set `LLM_PROVIDER` in `.env`. **The default is `ollama`** — fully local, so no
API key is needed and no conversation leaves your machine. Switch to `gemini`
for faster answers and more reliable citations, at the cost of sending
excerpts to Google.

| Provider | Config | Notes |
|---|---|---|
| `ollama` | `OLLAMA_MODEL`, `OLLAMA_NUM_CTX` | **Default.** Fully local/offline, no key, no quota. Slower (20-40s on an 8B model) and cites less consistently. |
| `gemini` | `GEMINI_API_KEY` ([get one](https://aistudio.google.com/apikey)), `GEMINI_MODEL` | Faster and cites reliably. Free tier allows ~20 requests/day per model. Pin an explicit model, not `gemini-flash-latest`. |
| `openai` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Also accepts OpenAI-compatible endpoints. |
| `mock` | — | Deterministic; used by the test suite. |

Gemini model names retire. If you see `404 ... no longer available to new
users`, list what your key can actually reach and update `GEMINI_MODEL`:

```bash
python -c "import requests;from src.config import CONFIG;\
print([m['name'] for m in requests.get(f'{CONFIG.gemini_base_url}/models',\
headers={'x-goog-api-key':CONFIG.gemini_api_key}).json()['models']\
if 'generateContent' in m.get('supportedGenerationMethods',[])])"
```

**Ollama troubleshooting** — `ollama pull llama3.1:8b` first. On
`failed to allocate buffer`, set `OLLAMA_NUM_CTX=2048` or pull `llama3.2:3b`
(weaker; may over-abstain). On a CUDA crash (exit `0xc0000409`, NVIDIA driver
older than Ollama's bundled CUDA), set `OLLAMA_NUM_GPU=0` to force CPU.

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

### The full retrieval stack

`RagPipeline` alone is vector-only. The stack the benchmarks measure — and what
the Streamlit app runs — is assembled by `GroundedAnswerer`:

```python
from src.runtime import init_native_runtimes
init_native_runtimes()          # must precede faiss/torch; see src/runtime.py

from src.pipeline.full_stack import GroundedAnswerer
from src.pipeline.rag_pipeline import RagPipeline

answerer = GroundedAnswerer(RagPipeline())    # vector + BM25 -> RRF -> reranker
ans = answerer.answer("What did Rahul tell me about his internship?")
```

It deliberately passes **no score floor**: `MIN_RETRIEVAL_SCORE` is a cosine
threshold and the stack's final scores are cross-encoder logits. Abstention is
decided by the grounded prompt instead. See `docs/BASELINE.md` for why.

### Retrieval strictness

The sidebar offers **Strict / Balanced / Permissive**. It adjusts *candidate
depth* — how many chunks are fused, reranked and handed to the model — and is
deliberately **not** a confidence threshold: the four stages produce four
incomparable score scales.

```python
answerer = GroundedAnswerer(RagPipeline(), strictness="permissive")
```

Use **Permissive** for credentials, rare names and codes, where the right chunk
ranks poorly. It is the only setting that gets the credential evidence into the
model's context — and the model still refuses to call an unlabelled token a
password, because permissiveness changes what it may *read*, never what it may
*claim*. **Balanced is the default** and the configuration every published
benchmark number was measured with.

## Project layout

```
src/
  ingestion/   base.py (ChatSource) · whatsapp_parser.py · normalizer.py
  storage/     models.py (normalized schema) · database.py (SQLite = source of truth)
  chunking/    base.py · message_chunker.py · time_chunker.py · service.py
  embeddings/  base.py · sentence_transformer.py · mock.py · factory.py
  retrieval/   base.py (ChunkSearcher) · vector_store.py (FAISS) · indexer.py
               vector_search.py · keyword_search.py (BM25) · hybrid_search.py (RRF)
               reranker.py (cross-encoder) · retriever.py
  llm/         base.py · factory.py · gemini_client.py · ollama_client.py
               openai_client.py · mock.py · prompts.py
  pipeline/    rag_pipeline.py · full_stack.py (the assembled stack)
               strictness.py (Strict/Balanced/Permissive depths)
  evaluation/  dataset.py · corpus.py · metrics.py · runner.py · sweep.py
               calibration.py · margins.py · end_to_end.py
  runtime.py                           ← FAISS/torch OpenMP guard
  graph/ query/ agent/                 ← Phase 3+ (scaffolded)
app.py                                 ← Streamlit UI
scripts/generate_synthetic_chats.py    <- sample chats (generated, not tracked)
tests/                                 ← 445 tests (see docs/TESTS.md)
```

## Pipeline

```
.txt export
  → whatsapp_parser (format only)
  → normalizer (identity, stable ids, ordering, dedup)
  → SQLite  (authoritative: messages, conversations, chunks, embedding_meta)
  → ChunkingService     conversation-bounded windows, [m:<id>]-tagged text
  → EmbeddingIndexer    all-MiniLM-L6-v2 → FAISS (IndexIDMap2/IP), staleness by content_hash
  → Retriever           metadata filter → search → context window (§14)
  → GroundedAnswerer    vector + BM25 → RRF → cross-encoder → top-N chunks
  → RagPipeline.answer   grounded prompt → LLM → validated citations
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
(`mock` skips the download), `LLM_PROVIDER` (`ollama` | `gemini` | `openai` |
`mock`), `OLLAMA_NUM_CTX`, `CHUNK_STRATEGY` (`fixed_count` | `time_window`),
`CHUNK_SIZE_MESSAGES` / `CHUNK_OVERLAP_MESSAGES`, `RETRIEVAL_TOP_K`,
`MIN_RETRIEVAL_SCORE`, `CONTEXT_WINDOW_MESSAGES`.
