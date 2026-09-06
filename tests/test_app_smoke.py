"""Streamlit UI smoke test via AppTest (mock embedder + mock LLM)."""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = "app.py"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    import streamlit as st

    st.cache_resource.clear()  # don't reuse a pipeline from a previous test
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-64")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cm.db"))
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "-1")
    monkeypatch.setenv("CHUNK_SIZE_MESSAGES", "6")
    monkeypatch.setenv("CHUNK_OVERLAP_MESSAGES", "2")
    monkeypatch.setenv("MIN_MESSAGES_PER_CHUNK", "2")


def _click(at, key):
    for b in at.button:
        if b.key == key:
            return b.click().run()
    raise AssertionError(f"button {key!r} not found")


def test_app_boots_empty(env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("No data yet" in i.value for i in at.info)


def test_load_sample_then_ask(env):
    at = AppTest.from_file(APP, default_timeout=90).run()
    _click(at, "btn_sample")
    assert not at.exception
    # data is loaded: the "No data yet" gate is gone and metrics are populated
    assert not any("No data yet" in i.value for i in at.info)
    assert any(m.label == "Conversations" and str(m.value) == "4" for m in at.metric)

    at.text_input(key="question").set_value("What did Rahul say about the Microsoft internship?")
    submit = at.get("form_submit_button") or [b for b in at.button if b.label == "Ask"]
    submit[0].click().run()
    assert not at.exception
    joined = " ".join(m.value for m in at.markdown)
    assert "### Answer" in joined and "#### Sources" in joined
