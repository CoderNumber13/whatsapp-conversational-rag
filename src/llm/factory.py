from __future__ import annotations

from src.config import CONFIG, Config
from src.llm.base import LLMClient, LLMError


def get_llm(config: Config = CONFIG) -> LLMClient:
    provider = config.llm_provider.lower()
    if provider == "mock":
        from src.llm.mock import MockLLM

        return MockLLM()
    if provider == "ollama":
        from src.llm.ollama_client import OllamaClient

        return OllamaClient(
            host=config.ollama_host,
            model=config.ollama_model,
            num_ctx=config.ollama_num_ctx,
            temperature=config.llm_temperature,
            timeout_s=config.llm_timeout_s,
            num_gpu=config.ollama_num_gpu,
        )
    if provider == "openai":
        from src.llm.openai_client import OpenAIClient

        return OpenAIClient(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.openai_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            timeout_s=config.llm_timeout_s,
        )
    raise LLMError(f"unknown LLM_PROVIDER: {config.llm_provider!r}")
