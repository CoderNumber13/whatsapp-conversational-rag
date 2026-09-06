"""Local Ollama chat client.

Sets ``options.num_ctx`` explicitly — Ollama otherwise defaults llama3.1 to a
2048-token context and silently truncates RAG prompts.
"""

from __future__ import annotations

import requests

from src.llm.base import LLMClient, LLMError, LLMResponse


class OllamaClient(LLMClient):
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        num_ctx: int = 8192,
        temperature: float = 0.0,
        timeout_s: int = 120,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str, **opts) -> LLMResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": opts.get("temperature", self.temperature),
                "num_predict": opts.get("max_tokens", -1),
            },
        }
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout_s
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMError(f"Ollama timed out after {self.timeout_s}s.") from e
        except requests.exceptions.HTTPError as e:
            raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}") from e
        except ValueError as e:
            raise LLMError("Ollama returned a non-JSON response.") from e

        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise LLMError(f"Unexpected Ollama response shape: {data!r}") from e

        usage = {
            k: data[k]
            for k in ("prompt_eval_count", "eval_count", "total_duration")
            if k in data
        }
        return LLMResponse(text=text, model=self.model, usage=usage)
