# WhatsApp Conversational RAG

A privacy-first, local-first system for asking natural-language questions about your own WhatsApp conversation history.

You provide WhatsApp `.txt` exports and ask questions through a Streamlit app. Answers include the source messages used to support them.

## Privacy

- Real chat data is not committed to the repository. `data/private/` and `.txt` files are git-ignored.
- No WhatsApp scraping or account access is used. You provide your own exported `.txt` files.
- The LLM provider is configurable: `ollama`, `gemini`, `openai`, or `mock`.
- API keys are stored in `.env`, which is git-ignored.
- The FAISS index can be rebuilt from the SQLite database.

## Setup

Requires **Python 3.12**. Python 3.11 also works.

### 1. Create the environment

```bash
conda create -n convmem python=3.12 -y
conda activate convmem
pip install -r requirements.txt
```

On Windows, keep the `convmem` environment activated when running the app. The project uses FAISS and PyTorch, which may not work correctly with newer Python versions.

If you run into FAISS/OpenMP issues on Windows, use the conda build:

```bash
pip uninstall -y faiss-cpu
conda install -y -c pytorch faiss-cpu
```

### 2. Configure the project

Copy the example environment file:

```bash
cp .env.example .env
```

Then open `.env` and set your WhatsApp display name in `ME_NAMES`.

Do not put API keys directly in the source code or commit your `.env` file to GitHub.

### 3. Run the tests

The test suite uses mock models, so no API key is required:

```bash
pytest
```

### 4. Start the app

```bash
streamlit run app.py
```

On Windows, you can also use:

```text
run_app.bat
```

## Choosing an LLM

Set `LLM_PROVIDER` in `.env`.

### Ollama

Ollama is the default option and runs locally. No API key is required.

Install Ollama, then download a model:

```bash
ollama pull llama3.1:8b
```

If your laptop has limited RAM, you can use a smaller model:

```bash
ollama pull llama3.2:3b
```

You can also reduce the context size in `.env`:

```env
OLLAMA_NUM_CTX=2048
```

### Gemini

Use Gemini if you prefer a hosted model.

Set these values in `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=your_model_name
```

Keep the API key in `.env`. Never commit it to GitHub.

### OpenAI

You can also use an OpenAI-compatible endpoint:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_base_url
```

### Mock

The mock provider is mainly useful for testing:

```env
LLM_PROVIDER=mock
```

## Using the app

Start the Streamlit app and either:

1. Click **Load sample** to generate synthetic conversations.
2. Upload your own WhatsApp `.txt` export.
3. Ask questions about the conversation.

The app displays the source messages used for each answer.

For local use, Ollama is the simplest option because your conversation data stays on your machine.

## Using it from Python

You can also use the RAG pipeline directly:

```python
from src.pipeline.rag_pipeline import RagPipeline
from src.retrieval.retriever import RetrievalFilters

pipe = RagPipeline()

pipe.ingest(
    ["path/to/WhatsApp Chat with Rahul.txt"],
    me_names=["Me"]
)

ans = pipe.answer("What did Rahul tell me about his internship?")

print(ans.text)
print(ans.supported)

for citation in ans.citations:
    print(citation.timestamp, citation.sender, citation.text)

# Optional metadata filter
ans = pipe.answer(
    "What did we decide?",
    filters=RetrievalFilters(sender="Rahul")
)
```

## Retrieval

The Streamlit app uses:

- Vector search with FAISS
- BM25 keyword search
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Metadata filters
- Conversation context reconstruction
- Grounded LLM answers with citations

If there is not enough supporting information, the system can abstain instead of guessing.

The sidebar has three retrieval settings:

- **Strict**
- **Balanced**
- **Permissive**

Balanced is the default.

## Project structure

```text
src/
├── ingestion/       WhatsApp parsing and normalization
├── storage/         SQLite database
├── chunking/        Conversation-aware chunking
├── embeddings/      Embedding models
├── retrieval/       FAISS, BM25, hybrid search, reranking
├── llm/             LLM providers and prompts
├── pipeline/        RAG pipeline and full retrieval stack
├── evaluation/      Evaluation and benchmark code
└── runtime.py       Runtime setup

app.py               Streamlit UI
scripts/              Utility scripts
tests/                Test suite
```

## Configuration

Configuration is stored in `.env`.

Common settings include:

```env
EMBEDDING_MODEL=
LLM_PROVIDER=ollama
OLLAMA_NUM_CTX=
CHUNK_STRATEGY=fixed_count
CHUNK_SIZE_MESSAGES=
CHUNK_OVERLAP_MESSAGES=
RETRIEVAL_TOP_K=
MIN_RETRIEVAL_SCORE=
CONTEXT_WINDOW_MESSAGES=
ME_NAMES=
```

For the complete list of settings, see `.env.example`.

## Troubleshooting

### `ModuleNotFoundError: faiss`

Make sure the `convmem` environment is active:

```bash
conda activate convmem
```

Then run:

```bash
streamlit run app.py
```

### Ollama fails to load a model

Try a smaller model:

```bash
ollama pull llama3.2:3b
```

If the model runs out of memory, reduce:

```env
OLLAMA_NUM_CTX=2048
```

### CUDA crash with Ollama

If Ollama crashes while using an NVIDIA GPU, you can force CPU execution:

```env
OLLAMA_NUM_GPU=0
```

## Rebuilding the FAISS index

The FAISS index is derived from the SQLite data. If needed, delete:

```text
data/private/index/
```

and re-ingest your chats.

## License

See the repository license for usage and distribution terms.
