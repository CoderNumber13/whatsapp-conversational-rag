"""LLM adapters, factory and the grounded prompt (no real model calls)."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.config import Config
from src.llm.base import LLMError
from src.llm.factory import get_llm
from src.llm.mock import MockLLM
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import NOT_FOUND_TOKEN, build_user_prompt
from src.storage.models import Message


class _FakeRC:
    def __init__(self, msgs, name="Rahul", is_group=False, score=0.5):
        self._msgs = msgs
        self.conversation = {"name": name, "is_group": is_group}
        self.score = score

    def ordered_messages(self):
        return self._msgs


def _msg(text, mid_seed):
    return Message(
        conversation_id="c",
        conversation_name="Rahul",
        timestamp=datetime(2026, 8, 17, 9, 16),
        sender="Rahul",
        text=text,
        message_id=f"{mid_seed:016x}",
        seq=mid_seed,
    )


# --- MockLLM ---------------------------------------------------------

def test_mock_llm_not_found_without_citations():
    assert MockLLM().complete("sys", "no citations here").text == NOT_FOUND_TOKEN


def test_mock_llm_echoes_first_citation():
    out = MockLLM().complete("sys", "context [m:00000000000000ab] more [m:00000000000000cd]")
    assert "[m:00000000000000ab]" in out.text


# --- factory ------------------------------------------------------

def test_factory_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert isinstance(get_llm(Config.reload()), MockLLM)


def test_factory_ollama_builds_without_network(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    client = get_llm(Config.reload())
    assert isinstance(client, OllamaClient)
    assert client.num_ctx == 4096


def test_factory_openai_without_key_errors(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(LLMError):
        get_llm(Config.reload())


def test_factory_unknown_provider_errors(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "banana")
    with pytest.raises(LLMError):
        get_llm(Config.reload())


# --- prompt -----------------------------------------------------

def test_build_user_prompt_shape():
    rc = _FakeRC([_msg("I got the Microsoft internship offer", 0xAB),
                  _msg("Joining date is 2 September", 0xCD)])
    prompt = build_user_prompt("When does Rahul join?", [rc], filters_note="sender=Rahul")
    assert "Question: When does Rahul join?" in prompt
    assert "Excerpt 1 — Rahul (direct)" in prompt
    assert "[m:00000000000000ab]" in prompt and "[m:00000000000000cd]" in prompt
    assert "sender=Rahul" in prompt
    assert NOT_FOUND_TOKEN in prompt


# --- ollama error handling -------------------------------------

def test_ollama_connection_error_is_wrapped():
    client = OllamaClient(host="http://127.0.0.1:9", model="llama3.1:8b", timeout_s=2)
    with pytest.raises(LLMError) as ei:
        client.complete("sys", "user")
    assert "ollama serve" in str(ei.value).lower()
