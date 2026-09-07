"""LLM adapters, factory and the grounded prompt (no real model calls)."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.config import Config
from src.llm.base import LLMError
from src.llm.factory import get_llm
from src.llm.gemini_client import GeminiClient
from src.llm.mock import MockLLM
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import NOT_FOUND_TOKEN, build_user_prompt
from src.storage.models import Message


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(response=self)


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


# --- gemini ---------------------------------------------------------

def test_factory_gemini_builds_with_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = get_llm(Config.reload())
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-2.5-flash"


def test_factory_gemini_without_key_errors(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(LLMError):
        get_llm(Config.reload())


def test_gemini_parses_response_and_sends_key_in_header(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _FakeResp(
            {
                "candidates": [{"content": {"parts": [{"text": "Rahul joins 2 Sept [m:abc]."}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"totalTokenCount": 42},
            }
        )

    monkeypatch.setattr("src.llm.gemini_client.requests.post", fake_post)
    out = GeminiClient(api_key="k", model="gemini-2.5-flash").complete("SYS", "USER")
    assert out.text == "Rahul joins 2 Sept [m:abc]."
    assert out.usage == {"totalTokenCount": 42}
    assert captured["headers"]["x-goog-api-key"] == "k"
    assert "gemini-2.5-flash:generateContent" in captured["url"]
    assert captured["body"]["system_instruction"]["parts"][0]["text"] == "SYS"


def test_gemini_blocked_prompt_raises(monkeypatch):
    monkeypatch.setattr(
        "src.llm.gemini_client.requests.post",
        lambda *a, **k: _FakeResp({"promptFeedback": {"blockReason": "SAFETY"}}),
    )
    with pytest.raises(LLMError, match="blocked"):
        GeminiClient(api_key="k").complete("s", "u")


def test_gemini_empty_answer_raises(monkeypatch):
    monkeypatch.setattr(
        "src.llm.gemini_client.requests.post",
        lambda *a, **k: _FakeResp({"candidates": [{"content": {"parts": []},
                                                   "finishReason": "MAX_TOKENS"}]}),
    )
    with pytest.raises(LLMError, match="empty"):
        GeminiClient(api_key="k").complete("s", "u")


def test_gemini_connection_error_wrapped(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr("src.llm.gemini_client.requests.post", boom)
    with pytest.raises(LLMError, match="Cannot reach"):
        GeminiClient(api_key="k").complete("s", "u")


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
