"""Streamlit UI for the Conversation Memory RAG.

    streamlit run app.py

Upload WhatsApp .txt exports (or load the synthetic sample set), then ask
questions. Every answer shows its supporting messages and the raw retrieval
so you can see why it said what it said.
"""

from __future__ import annotations

# MUST be first: faiss-cpu and torch link different OpenMP runtimes and the
# second to initialise aborts the process (exit 3, no traceback). See
# src/runtime.py for the evidence and the proper environment-level fix.
from src.runtime import init_native_runtimes  # isort:skip

init_native_runtimes()  # noqa: E402

from datetime import datetime, time  # noqa: E402
from pathlib import Path  # noqa: E402

import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import Config  # noqa: E402
from src.llm.base import LLMError  # noqa: E402
from src.pipeline.full_stack import GroundedAnswerer  # noqa: E402
from src.pipeline.rag_pipeline import RagPipeline  # noqa: E402
from src.retrieval.retriever import RetrievalFilters  # noqa: E402

SAMPLE_DIR = Path(__file__).parent / "data" / "sample" / "synthetic_chats"

st.set_page_config(page_title="Conversation Memory", page_icon="💬", layout="wide")


# --- resources ------------------------------------------------------

@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline(Config.reload())


@st.cache_resource
def get_answerer(_pipe: RagPipeline, rerank: bool) -> GroundedAnswerer:
    """Full stack: vector + BM25 -> RRF -> (cross-encoder) -> LLM.

    Cached because the cross-encoder is a ~80MB model load. Built on first
    question rather than at import so the app starts promptly.
    """
    return GroundedAnswerer(_pipe, rerank=rerank)


def llm_status(cfg: Config) -> tuple[bool, str]:
    if cfg.llm_provider == "mock":
        return True, "mock LLM (canned answers)"
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
            f"gemini · {cfg.gemini_model}"
            if cfg.gemini_api_key
            else "GEMINI_API_KEY not set — add it to .env"
        )
    if cfg.llm_provider == "openai":
        return bool(cfg.openai_api_key), (
            f"openai · {cfg.openai_model}"
            if cfg.openai_api_key
            else "OPENAI_API_KEY not set — add it to .env"
        )
    return False, f"unknown provider {cfg.llm_provider!r}"


def friendly_llm_error(exc: Exception) -> str:
    """Turn provider failures into something a demo audience can act on."""
    text = str(exc)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return ("The model's request quota is exhausted. Free Gemini tiers allow "
                "only a handful of requests per day per model — wait, switch "
                "GEMINI_MODEL in .env, or use a paid key.")
    if "503" in text or "UNAVAILABLE" in text:
        return "The model is temporarily overloaded. Try again in a moment."
    if "404" in text or "NOT_FOUND" in text:
        return ("That model name is not available to this API key. Update "
                "GEMINI_MODEL in .env (see README for how to list valid names).")
    if "not set" in text.lower() or "api_key" in text.lower():
        return "No API key configured. Set GEMINI_API_KEY in .env and restart."
    return text


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
        try:
            with st.spinner(f"Ingesting {len(paths)} file(s)…"):
                rep = pipe.ingest(
                    paths, me_names=[s.strip() for s in me_names.split(",") if s.strip()]
                )
        except Exception as e:
            st.error(f"Ingestion failed — {type(e).__name__}: {e}")
        else:
            if rep.messages_added == 0:
                st.warning(
                    "No new messages were parsed. The file may be empty, already "
                    "ingested, or not a WhatsApp export."
                )
            else:
                st.success(
                    f"+{rep.messages_added} messages · {rep.conversations} conversations · "
                    f"chunks +{rep.chunks.added}/~{rep.chunks.updated} · "
                    f"embedded {rep.index.embedded + rep.index.reembedded}"
                )
            # the index changed, so drop the cached searchers (not the DB)
            get_answerer.clear()
            st.rerun()

    if col_b.button("Load sample", use_container_width=True, key="btn_sample"):
        files = sorted(SAMPLE_DIR.glob("*.txt"))
        if not files:
            st.error(f"No sample exports found in {SAMPLE_DIR}")
        else:
            with st.spinner("Ingesting synthetic sample chats…"):
                rep = pipe.ingest(files, me_names=["Me"])
            st.success(f"Loaded {rep.conversations} sample conversations.")
            get_answerer.clear()
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
        top_k = st.slider("evidence chunks sent to the LLM", 1, 20,
                          cfg.max_context_chunks)
        rerank = st.checkbox(
            "cross-encoder reranking", value=True,
            help="Reorders candidates by reading query and chunk together. "
                 "Loads a ~80MB model on first use and adds ~0.3s per query.",
        )
        st.caption(
            "Retrieval: vector + BM25 → RRF"
            + (" → cross-encoder" if rerank else "")
            + ". Whether to answer is decided by the grounded prompt, not by a "
              "score threshold — retrieval scores are not comparable across "
              "stages, so there is no slider for it."
        )


# --- main: ask ----------------------------------------------------

st.title("💬 Conversation Memory")

if db.count_chunks() == 0:
    st.info("No data yet — upload exports or click **Load sample** in the sidebar.")
    st.stop()

convs = db.list_conversations()
conv_by_label = {
    f"{c['name']}{' (group)' if c['is_group'] else ''}": c["conversation_id"]
    for c in convs
}

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
    submitted = st.form_submit_button("Ask", type="primary", disabled=not ok)

if not ok:
    st.error(f"Can't answer yet: {msg}")

if submitted and question.strip():
    filters = RetrievalFilters(
        conversation_id=conv_by_label.get(conv_label) if conv_label != "(any)" else None,
        sender=sender.strip() or None,
        date_from=datetime.combine(d_from, time.min) if d_from else None,
        date_to=datetime.combine(d_to, time.max) if d_to else None,
    )
    has_filters = any(v is not None for v in filters.as_kwargs().values())
    try:
        with st.spinner("Loading retrieval stack…" if rerank else "Preparing…"):
            answerer = get_answerer(pipe, rerank)
        with st.spinner("Retrieving and answering…"):
            ans = answerer.answer(
                question, filters=filters if has_filters else None, k=top_k
            )
    except LLMError as e:
        st.error(friendly_llm_error(e))
        st.stop()
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

    if not ans.supported:
        # Either the model emitted NOT_FOUND, or every citation it produced was
        # invented and therefore dropped. Both mean: not grounded, so don't
        # present it as an answer.
        st.warning(ans.text)
        st.caption(
            "The system refuses rather than guessing when the retrieved "
            "conversations don't support an answer."
        )
    else:
        st.markdown(f"### Answer\n{ans.text}")
        st.caption(f"model: {ans.llm_model} · {len(ans.citations)} citation(s)")

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
        st.caption(
            "Scores come from the last retrieval stage and are only comparable "
            "within this list: cosine for vector-only, an RRF fusion value, or a "
            "cross-encoder logit (unbounded, often negative) when reranking is on."
        )
        cited_ids = {c.message_id for c in ans.citations}
        for i, rc in enumerate(ans.retrieved, 1):
            used = "✅ cited" if cited_ids & {m.message_id for m in rc.messages} else "—"
            st.markdown(
                f"**#{i} · score {rc.score:.3f} · {rc.conversation['name']} · "
                f"seq {rc.chunk.seq_start}–{rc.chunk.seq_end} · {used}**"
            )
            st.code("\n".join(m.display_line() for m in rc.ordered_messages()), language=None)
