"""LLM client interface — one method, provider-agnostic.

No provider is imported here. Concrete clients live in sibling modules and are
chosen by :func:`src.llm.factory.get_llm` from ``LLM_PROVIDER``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    """Raised for connection / HTTP / response-shape problems."""


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMClient(ABC):
    model: str

    @abstractmethod
    def complete(self, system: str, user: str, **opts) -> LLMResponse:
        """Single-turn completion. ``opts`` may carry temperature / max_tokens."""
