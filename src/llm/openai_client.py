"""OpenAI-compatible chat client.

Works against api.openai.com or any compatible endpoint (``OPENAI_BASE_URL``)
— local servers included. The key is read from config/env only; never logged.
"""

from __future__ import annotations

import requests

from src.llm.base import LLMClient, LLMError, LLMResponse


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: int = 120,
    ) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str, **opts) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": opts.get("temperature", self.temperature),
            "max_tokens": opts.get("max_tokens", self.max_tokens),
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"Cannot reach the LLM endpoint at {self.base_url}.") from e
        except requests.exceptions.Timeout as e:
            raise LLMError(f"LLM request timed out after {self.timeout_s}s.") from e
        except requests.exceptions.HTTPError as e:
            raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}") from e
        except ValueError as e:
            raise LLMError("LLM returned a non-JSON response.") from e

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {data!r}") from e
        return LLMResponse(text=text, model=self.model, usage=data.get("usage", {}))
