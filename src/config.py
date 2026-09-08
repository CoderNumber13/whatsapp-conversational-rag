"""Central configuration.

Values come from environment variables (optionally loaded from a local ``.env``),
with safe defaults so the parser/storage layer works with zero setup.
Nothing here is provider-specific beyond a name the caller can switch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Config:
    # Identity
    me_names: list[str] = field(default_factory=lambda: _get_list("ME_NAMES", ["Me"]))

    # Storage
    db_path: Path = field(
        default_factory=lambda: (REPO_ROOT / _get("DB_PATH", "data/private/conversation_memory.db"))
    )
    index_dir: Path = field(
        default_factory=lambda: (REPO_ROOT / _get("INDEX_DIR", "data/private/index"))
    )

    # Embeddings (increment 2)
    embedding_model: str = field(
        default_factory=lambda: _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_device: str = field(default_factory=lambda: _get("EMBEDDING_DEVICE", "cpu"))

    # LLM (increment 2)
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "ollama"))
    ollama_host: str = field(default_factory=lambda: _get("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _get("OLLAMA_MODEL", "llama3.1:8b"))
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: _get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(default_factory=lambda: _get("OPENAI_MODEL", "gpt-4o-mini"))
    gemini_api_key: str = field(default_factory=lambda: _get("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _get("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_base_url: str = field(
        default_factory=lambda: _get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
    )

    embedding_batch_size: int = field(
        default_factory=lambda: _get_int("EMBEDDING_BATCH_SIZE", 64)
    )

    # LLM generation
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.0))
    llm_max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 800))
    llm_timeout_s: int = field(default_factory=lambda: _get_int("LLM_TIMEOUT_S", 120))
    # llama3.1 defaults to 2048 in Ollama; 4096 gives RAG-prompt headroom without
    # inflating the KV-cache allocation on low-RAM machines. Drop to 2048 (or use a
    # 3B model) if Ollama fails to allocate; raise to 8192 with RAM to spare.
    ollama_num_ctx: int = field(default_factory=lambda: _get_int("OLLAMA_NUM_CTX", 4096))
    # -1 = let Ollama decide (default). Set to 0 to force CPU-only, e.g. when the
    # CUDA runner crashes ("shared object initialization failed" / 0xc0000409)
    # because the NVIDIA driver is older than Ollama's bundled CUDA runtime.
    ollama_num_gpu: int = field(default_factory=lambda: _get_int("OLLAMA_NUM_GPU", -1))

    # Chunking
    chunk_strategy: str = field(default_factory=lambda: _get("CHUNK_STRATEGY", "fixed_count"))
    chunk_size_messages: int = field(default_factory=lambda: _get_int("CHUNK_SIZE_MESSAGES", 12))
    chunk_overlap_messages: int = field(
        default_factory=lambda: _get_int("CHUNK_OVERLAP_MESSAGES", 3)
    )
    chunk_time_window_minutes: int = field(
        default_factory=lambda: _get_int("CHUNK_TIME_WINDOW_MINUTES", 45)
    )
    min_messages_per_chunk: int = field(
        default_factory=lambda: _get_int("MIN_MESSAGES_PER_CHUNK", 4)
    )

    # Retrieval / answering
    retrieval_top_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_K", 8))
    min_retrieval_score: float = field(
        default_factory=lambda: _get_float("MIN_RETRIEVAL_SCORE", 0.25)
    )
    context_window_messages: int = field(
        default_factory=lambda: _get_int("CONTEXT_WINDOW_MESSAGES", 3)
    )
    max_context_chunks: int = field(default_factory=lambda: _get_int("MAX_CONTEXT_CHUNKS", 8))

    # Reciprocal Rank Fusion. RRF_K damps the influence of top ranks: the
    # contribution of a result at rank r is 1/(RRF_K + r), so a large K flattens
    # the curve and lets agreement between retrievers outweigh any single
    # retriever's confidence. 60 is the value from Cormack et al. (2009), which
    # introduced RRF, and is the near-universal default.
    rrf_k: int = field(default_factory=lambda: _get_int("RRF_K", 60))
    # How deep to read each retriever before fusing. Must exceed the final k, or
    # fusion has nothing to disagree about; 50 keeps every candidate either
    # retriever ranked plausibly while staying cheap.
    rrf_candidates: int = field(default_factory=lambda: _get_int("RRF_CANDIDATES", 50))

    # Paths
    uploads_dir: Path = field(
        default_factory=lambda: (REPO_ROOT / _get("UPLOADS_DIR", "data/private/uploads"))
    )

    # Logging
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))
    log_redact_text: bool = field(default_factory=lambda: _get_bool("LOG_REDACT_TEXT", True))

    @classmethod
    def reload(cls) -> "Config":
        """Rebuild from the current environment (``.env`` is re-read on import only)."""
        return cls()


CONFIG = Config()
