"""Deterministic LLM stand-in for tests and offline runs.

Contract-compatible with the grounded prompt: it copies through the first
message-id citation it finds in the prompt, or answers ``NOT_FOUND`` when the
context carries none.
"""

from __future__ import annotations

import re

from src.llm.base import LLMClient, LLMResponse

_CITE = re.compile(r"\[m:[0-9a-f]+\]")


class MockLLM(LLMClient):
    model = "mock-llm"

    def complete(self, system: str, user: str, **opts) -> LLMResponse:
        cites = _CITE.findall(user)
        if not cites:
            return LLMResponse(text="NOT_FOUND", model=self.model)
        first = cites[0]
        return LLMResponse(
            text=f"Based on the retrieved messages, here is the answer {first}.",
            model=self.model,
        )
