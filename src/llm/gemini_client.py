"""Google Gemini (Generative Language API) chat client.

Uses the native ``:generateContent`` endpoint. The API key goes in the
``x-goog-api-key`` header (not the URL) and is never logged.

Get a key at https://aistudio.google.com/apikey
"""

from __future__ import annotations

import requests

from src.llm.base import LLMClient, LLMError, LLMResponse


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: int = 120,
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set (get one at https://aistudio.google.com/apikey).")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str, **opts) -> LLMResponse:
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": opts.get("temperature", self.temperature),
                "maxOutputTokens": opts.get("max_tokens", self.max_tokens),
            },
        }
        url = f"{self.base_url}/models/{self.model}:generateContent"
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"x-goog-api-key": self.api_key},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"Cannot reach the Gemini API at {self.base_url}.") from e
        except requests.exceptions.Timeout as e:
            raise LLMError(f"Gemini request timed out after {self.timeout_s}s.") from e
        except requests.exceptions.HTTPError as e:
            raise LLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}") from e
        except ValueError as e:
            raise LLMError("Gemini returned a non-JSON response.") from e

        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise LLMError(f"Gemini blocked the prompt: {block}")

        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"Gemini returned no candidates: {data!r}")

        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            reason = cand.get("finishReason", "unknown")
            raise LLMError(f"Gemini returned an empty answer (finishReason={reason}).")

        usage = data.get("usageMetadata", {})
        return LLMResponse(text=text, model=self.model, usage=usage)
