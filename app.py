"""Streamlit UI for the Conversation Memory RAG.

    streamlit run app.py

Upload WhatsApp .txt exports (or load the synthetic sample set), then ask
questions. Every answer shows its supporting messages and the raw retrieval
so you can see why it said what it said.
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import requests
import streamlit as st

from src.config import Config
from src.pipeline.rag_pipeline import RagPipeline
from src.retrieval.retriever import RetrievalFilters

SAMPLE_DIR = Path(__file__).parent / "data" / "sample" / "synthetic_chats"

st.set_page_config(page_title="Conversation Memory", page_icon="💬", layout="wide")


# --- resources ------------------------------------------------------

@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline(Config.reload())


def llm_status(cfg: Config) -> tuple[bool, str]:
    if cfg.llm_provider == "mock":
        return True, "mock LLM"
    if cfg.llm_provider == "ollama":
        try:
            r = requests.get(f"{cfg.ollama_host.rstrip('/')}/api/tags", timeout=3)
            names = [m["name"] for m in r.json().get("models", [])]
            ok = cfg.ollama_model in names
            return ok, (
                f"ollama · {cfg.ollama_model}"
                if ok
                else f"ollama up, but '{cfg.ollama_model}' not pulled"
            )
        except Exception:
            return False, f"ollama unreachable at {cfg.ollama_host}"
    if cfg.llm_provider == "gemini":
        return bool(cfg.gemini_api_key), (
            f"gemini · {cfg.gemini_model}" if cfg.gemini_api_key else "GEMINI_API_KEY not set"
        )
    if cfg.llm_provider == "openai":
        return bool(cfg.openai_api_key), (
            f"openai · {cfg.openai_model}" if cfg.openai_api_key else "OPENAI_API_KEY not set"
        )
    return False, f"unknown provider {cfg.llm_provider!r}"


pipe = get_pipeline()
cfg = pipe.config
db = pipe.db


# --- sidebar: ingestion + corpus ---------------------------------

with st.sidebar:
    st.header("Data")

    up = st.file_uploader("WhatsApp .txt exports", type="txt", accept_multiple_files=True)
    me_names = st.text_input(
        "Your name(s) in these chats", value=", ".join(cfg.me_names), key="me_names"
    )
    col_a, col_b = st.columns(2)

    if col_a.button("Ingest uploads", disabled=not up, use_container_width=True, key="btn_ingest"):
        cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in up:
            p = cfg.uploads_dir / f.name
            p.write_bytes(f.getbuffer())
            paths.append(p)
        with st.spinner(f"Ingesting {len(paths)} file(s)…"):
            rep = pipe.ingest(paths, me_names=[s.strip() for s in me_names.split(",") if s.strip()])
        st.success(
            f"+{rep.messages_added} messages · {rep.conversations} conversations · "
            f"chunks +{rep.chunks.added}/~{rep.chunks.updated} · "
            f"embedded {rep.index.embedded + rep.index.reembedded}"
        )
        get_pipeline.clear()  # drop cached retriever so the new index is picked up
        st.rerun()

    if col_b.button("Load sample", use_container_width=True, key="btn_sample"):
        files = sorted(SAMPLE_DIR.glob("*.txt"))
        with st.spinner("Ingesting synthetic sample chats…"):
            rep = pipe.ingest(files, me_names=["Me"])
        st.success(f"Loaded {rep.conversations} sample conversations.")
        get_pipeline.clear()
        st.rerun()

    st.divider()
    stats = db.stats()
    st.metric("Messages", f"{stats['messages']:,}")
    st.metric("Conversations", stats["conversations"])
    lo, hi = stats["date_range"]
    if lo:
        st.caption(f"📅 {lo[:10]} → {hi[:10]}")
    if stats["participants"]:
        st.caption("👥 " + ", ".join(stats["participants"][:12]))

    st.divider()
    ok, msg = llm_status(cfg)
    st.caption(("🟢 " if ok else "🔴 ") + msg)
    st.caption(f"🔤 embeddings: {cfg.embedding_model}")

    with st.expander("Retrieval settings"):
        top_k = st.slider("top-k chunks", 1, 20, cfg.retrieval_top_k)
        min_score = st.slider("min cosine score (abstain below)", 0.0, 1.0, cfg.min_retrieval_score, 0.01)


# --- main: ask ----------------------------------------------------

st.title("💬 Conversation Memory")

if db.count_chunks() == 0:
    st.info("No data yet — upload exports or click **Load sample** in the sidebar.")
    st.stop()

convs = db.list_conversations()
conv_by_label = {f"{c['name']}{' (group)' if c['is_group'] else ''}": c["conversation_id"] for c in convs}

with st.form("ask"):
    question = st.text_input(
        "Ask about your conversations",
        placeholder="What did Rahul tell me about his internship?",
        key="question",
    )
    fc1, fc2, fc3, fc4 = st.columns(4)
    conv_label = fc1.selectbox("Conversation", ["(any)"] + list(conv_by_label), key="f_conv")
    sender = fc2.text_input("From sender", key="f_sender")
    d_from = fc3.date_input("From date", value=None, key="f_from")
    d_to = fc4.date_input("To date", value=None, key="f_to")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    filters = RetrievalFilters(
        conversation_id=conv_by_label.get(conv_label) if conv_label != "(any)" else None,
        sender=sender.strip() or None,
        date_from=datetime.combine(d_from, time.min) if d_from else None,
        date_to=datetime.combine(d_to, time.max) if d_to else None,
    )
    has_filters = any(v is not None for v in filters.as_kwargs().values())
    with st.spinner("Retrieving and answering…"):
        try:
            ans = pipe.answer(
                question,
                filters=filters if has_filters else None,
                k=top_k,
                min_score=min_score,
            )
        except Exception as e:  # LLM down, etc.
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

    if ans.abstained:
        st.warning(ans.text + f"  \n_(nothing retrieved above the score floor; best was {ans.top_score:.2f})_")
    elif not ans.supported:
        st.warning(ans.text)
    else:
        st.markdown(f"### Answer\n{ans.text}")
        st.caption(f"model: {ans.llm_model} · top score {ans.top_score:.2f} · {len(ans.citations)} citation(s)")

    if ans.citations:
        st.markdown("#### Sources")
        for c in ans.citations:
            st.markdown(
                f"**[{c.timestamp:%Y-%m-%d %H:%M}] {c.sender}** · _{c.conversation_name}_  \n"
                f"{c.text or '_(media / no text)_'}  \n"
                f"<sub>`{c.src_file}:{c.src_line_start}` · `m:{c.message_id}`</sub>",
                unsafe_allow_html=True,
            )

    with st.expander(f"Retrieval inspection — {len(ans.retrieved)} chunk(s)"):
        cited_ids = {c.message_id for c in ans.citations}
        for i, rc in enumerate(ans.retrieved, 1):
            used = "✅ cited" if cited_ids & {m.message_id for m in rc.messages} else "—"
            st.markdown(
                f"**#{i} · score {rc.score:.3f} · {rc.conversation['name']} · "
                f"seq {rc.chunk.seq_start}–{rc.chunk.seq_end} · {used}**"
            )
            st.code("\n".join(m.display_line() for m in rc.ordered_messages()), language=None)
