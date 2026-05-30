"""
app.py – Streamlit UI for the arXiv ReAct Agent.

Run:
    streamlit run app.py
"""

import json
import time
from pathlib import Path

import streamlit as st

from rag import config
from rag.processing.chunker import process_papers, save_chunks, load_chunks
from rag.retrieval.dense import Retriever
from rag.retrieval.bm25 import BM25Retriever
from rag.agent import ReActAgent, ConversationMemory, ResearchMemory

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="arXiv Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header  { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1280px; }
.stApp { background: #07070f; }

section[data-testid="stSidebar"] {
    background: #0a0a14;
    border-right: 1px solid #1a1a2a;
}
section[data-testid="stSidebar"] * { color: #c0bdd4 !important; }

.agent-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 700;
    color: #e8e4ff; letter-spacing: -0.02em;
}
.agent-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #5a5870;
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 4px;
}

/* Chat bubbles */
.msg-user {
    display: flex; justify-content: flex-end; margin: 1rem 0;
}
.msg-user-bubble {
    background: #2a284a; border: 1px solid #3a3860;
    color: #e0ddf0; padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%; font-size: 0.95rem; line-height: 1.6;
}
.msg-assistant { display: flex; margin: 1rem 0; gap: 10px; align-items: flex-start; }
.msg-avatar {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, #7c6fff, #c084fc);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0; margin-top: 2px;
}
.msg-bubble {
    background: #111120; border: 1px solid #1e1e30;
    color: #c0bdd4; padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 82%; font-size: 0.93rem; line-height: 1.75;
}

/* Scratchpad */
.scratchpad-step {
    background: #0d0d1a; border: 1px solid #1a1a2e;
    border-radius: 8px; padding: 12px; margin: 6px 0;
    font-size: 12px; font-family: 'Inter', sans-serif;
}
.step-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 2px;
}
.step-thought { color: #9d93c4; }
.step-action  { color: #7c6fff; font-weight: 600; }
.step-obs     { color: #5a8a6a; font-size: 11px; margin-top: 4px; }

/* Sources */
.source-chip {
    display: inline-block; background: #131325;
    border: 1px solid #252540; border-radius: 6px;
    padding: 4px 10px; margin: 3px 3px 3px 0;
    font-size: 11px; color: #8b88a8;
}
.source-score { font-family: 'JetBrains Mono', monospace; color: #7c6fff; font-size: 10px; margin-left: 5px; }

/* Pills */
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: #111120; border: 1px solid #1e1e30;
    border-radius: 20px; padding: 4px 11px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #5a5870; margin-right: 6px;
}
.pill-val { color: #9d93c4; font-weight: 500; }

/* Critique badge */
.badge-pass  { background: #0d2018; border: 1px solid #1d4030; color: #4ade80; border-radius: 6px; padding: 3px 10px; font-size: 11px; }
.badge-retry { background: #201208; border: 1px solid #3d2010; color: #fb923c; border-radius: 6px; padding: 3px 10px; font-size: 11px; }

/* Sub-question tags */
.subq-tag {
    display: inline-block; background: #12122a;
    border: 1px solid #2a2a50; border-radius: 6px;
    padding: 3px 9px; font-size: 11px; color: #7c6fff; margin: 2px;
}

/* Note card */
.note-card {
    background: #0d0d1a; border: 1px solid #1a1a2e;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
    font-size: 12px; color: #8b88a8;
}
.note-key { color: #c0bdd4; font-weight: 600; font-size: 13px; }

/* Paper list */
.paper-item {
    background: #0d0d1a; border: 1px solid #1a1a2e;
    border-radius: 8px; padding: 9px 13px; margin-bottom: 5px;
    font-size: 12px; color: #8b88a8;
}
.paper-title { color: #c0bdd4; font-weight: 500; font-size: 13px; }
.paper-meta  { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #3a3860; margin-top: 2px; }

/* Input */
.stTextInput > div > div > input {
    background: #0d0d1a !important; border: 1px solid #252540 !important;
    border-radius: 12px !important; color: #e0ddf0 !important;
    font-size: 0.95rem !important; padding: 13px 17px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c6fff !important;
    box-shadow: 0 0 0 2px rgba(124,111,255,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #3a3860 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":       [],
        "agent":          None,
        "conv_memory":    None,
        "res_memory":     None,
        "corpus_ready":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Cached resource loaders ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_chunks():
    cache = config.DATA_DIR / f"chunks_{config.DEFAULT_CHUNK}.json"
    if cache.exists():
        return load_chunks(cache)
    meta = config.DATA_DIR / "metadata.json"
    if not meta.exists():
        return []
    with open(meta) as f:
        papers = json.load(f)
    chunks = process_papers(papers, chunk_size=config.DEFAULT_CHUNK)
    save_chunks(chunks, cache)
    return chunks


@st.cache_resource(show_spinner=False)
def _get_dense_retriever(n_chunks: int):
    chunks = _get_chunks()
    return Retriever.build(
        model_key=config.MODEL_KEY,
        chunks=chunks,
        chunk_size=config.DEFAULT_CHUNK,
        index_dir=config.RESULTS_DIR / "indices",
    )


@st.cache_resource(show_spinner=False)
def _get_bm25_retriever(n_chunks: int):
    return BM25Retriever(_get_chunks())


@st.cache_data(show_spinner=False)
def _get_metadata():
    p = config.DATA_DIR / "metadata.json"
    return json.load(open(p)) if p.exists() else []


# ── Bootstrap agent ────────────────────────────────────────────────────────────
chunks = _get_chunks()
corpus_ready = len(chunks) > 0

if corpus_ready and st.session_state.agent is None:
    with st.spinner("Loading BGE retriever (first run may take ~30s)…"):
        dense = _get_dense_retriever(len(chunks))
    with st.spinner("Building BM25 index…"):
        bm25 = _get_bm25_retriever(len(chunks))

    conv_mem = ConversationMemory()
    res_mem  = ResearchMemory()
    st.session_state.conv_memory  = conv_mem
    st.session_state.res_memory   = res_mem
    st.session_state.agent        = ReActAgent(dense, bm25, conv_mem, res_mem)
    st.session_state.corpus_ready = True


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8e4ff;margin-bottom:2px;">⚙ Agent Config</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860;text-transform:uppercase;letter-spacing:.1em;margin-bottom:1rem;">BGE · ReAct · Gemini 2.5</p>',
        unsafe_allow_html=True,
    )

    st.markdown("**Model:** `BAAI/bge-large-en`")
    st.markdown("**Chunk size:** `512 tokens` (benchmark winner)")
    st.markdown("**Max steps:** `8`")
    st.markdown("**Backend:** Gemini 2.5 Flash Lite")

    st.divider()

    # ── Research notes ────────────────────────────────────────────────────────
    st.markdown("**📌 Research Notes**")
    res_mem: ResearchMemory = st.session_state.res_memory

    if res_mem:
        notes = res_mem.all_notes()
        if notes:
            for note in notes[-5:]:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"""
                    <div class="note-card">
                        <div class="note-key">{note.key}</div>
                        <div>{note.content[:80]}{'…' if len(note.content) > 80 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("✕", key=f"del_{note.key}"):
                        res_mem.delete_note(note.key)
                        st.rerun()
        else:
            st.caption("No notes yet. Ask the agent to save findings.")

        with st.expander("➕ Add Note"):
            note_key  = st.text_input("Label", placeholder="e.g. BGE vs MiniLM")
            note_body = st.text_area("Finding", placeholder="BGE outperforms…", height=80)
            if st.button("Save Note", use_container_width=True):
                if note_key and note_body:
                    res_mem.save_note(note_key, note_body)
                    st.rerun()

    st.divider()

    # ── Paper browser ──────────────────────────────────────────────────────────
    with st.expander("📚 Browse 150 Papers"):
        papers = _get_metadata()
        if papers:
            search = st.text_input("Filter", placeholder="Search…", label_visibility="collapsed", key="paper_filter")
            filtered = [p for p in papers if search.lower() in p["title"].lower()] if search else papers[:30]
            for p in filtered[:20]:
                st.markdown(f"""
                <div class="paper-item">
                    <div class="paper-title">{p['title'][:65]}{'…' if len(p['title'])>65 else ''}</div>
                    <div class="paper-meta">{p['paper_id']} · {p['published'][:10]}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.conv_memory:
                st.session_state.conv_memory.clear()
            st.rerun()
    with col_b:
        if st.button("🔄 Reset Agent", use_container_width=True):
            st.session_state.agent = None
            st.session_state.messages = []
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
col_title, col_pills = st.columns([3, 2])
with col_title:
    st.markdown('<h1 class="agent-title">arXiv Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="agent-sub">ReAct · BGE · Gemini 2.5 · 150 ML papers</p>', unsafe_allow_html=True)
with col_pills:
    if corpus_ready:
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:flex-end;padding-top:10px;flex-wrap:wrap;gap:5px">
            <span class="pill">📄 <span class="pill-val">{len(_get_metadata())}</span> papers</span>
            <span class="pill">🧩 <span class="pill-val">{len(chunks):,}</span> chunks</span>
            <span class="pill">⚡ <span class="pill-val">BGE</span></span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #1a1a2a;margin:12px 0 20px'>", unsafe_allow_html=True)


# ── Chat rendering ─────────────────────────────────────────────────────────────
chat = st.container()

with chat:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1rem">
            <div style="font-size:2.5rem;margin-bottom:10px">🤖</div>
            <p style="font-family:Syne,sans-serif;font-size:1.1rem;color:#8b88a8;">
                A research agent that reasons step-by-step over arXiv ML papers
            </p>
            <p style="font-size:11px;color:#3a3860;font-family:JetBrains Mono,monospace;">
                ReAct loop · Multi-hop planning · Self-critique · Conversation memory
            </p>
        </div>
        """, unsafe_allow_html=True)

        examples = [
            ("🔍", "What are the most effective approaches to RLHF in 2026?"),
            ("⚖️", "Compare LoRA and prefix tuning on parameter efficiency"),
            ("📊", "What evaluation benchmarks are used for LLM reliability?"),
            ("🧠", "How do recent papers handle catastrophic forgetting?"),
        ]
        cols = st.columns(2)
        for i, (icon, ex) in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"{icon} {ex}", key=f"ex_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": ex})
                    st.rerun()

    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="msg-user-bubble">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Main answer bubble
                st.markdown(f"""
                <div class="msg-assistant">
                    <div class="msg-avatar">🤖</div>
                    <div class="msg-bubble">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

                # Sub-questions (if complex query was decomposed)
                if msg.get("sub_questions"):
                    tags = "".join(f'<span class="subq-tag">{q}</span>' for q in msg["sub_questions"])
                    st.markdown(
                        f'<div style="margin-left:44px;margin-top:-8px;margin-bottom:8px">'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860;">DECOMPOSED INTO: </span>{tags}</div>',
                        unsafe_allow_html=True,
                    )

                # Critique badge
                if msg.get("critique_verdict"):
                    badge_class = "badge-pass" if msg["critique_verdict"] == "pass" else "badge-retry"
                    badge_text  = "✅ Self-critique: passed" if msg["critique_verdict"] == "pass" else "🔄 Refined after critique"
                    st.markdown(
                        f'<div style="margin-left:44px;margin-bottom:6px">'
                        f'<span class="{badge_class}">{badge_text}</span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860;margin-left:10px">'
                        f'{msg.get("total_steps", 0)} steps · {msg.get("latency_ms", 0):.0f}ms</span></div>',
                        unsafe_allow_html=True,
                    )

                # Scratchpad expander
                if msg.get("scratchpad"):
                    with st.expander(f"🔍 Agent Reasoning ({len(msg['scratchpad'])} steps)", expanded=False):
                        for i, step in enumerate(msg["scratchpad"], 1):
                            action_color = "#4ade80" if step["action"] == "finish" else "#7c6fff"
                            obs_html = ""
                            if step.get("observation"):
                                obs_preview = step["observation"][:300]
                                obs_html = f'<div class="step-obs">📋 {obs_preview}{"…" if len(step["observation"]) > 300 else ""}</div>'
                            st.markdown(f"""
                            <div class="scratchpad-step">
                                <div class="step-label" style="color:#3a3860">Step {i}</div>
                                <div class="step-thought">💭 {step["thought"]}</div>
                                <div class="step-action" style="margin-top:4px;color:{action_color}">
                                    ⚡ {step["action"]}({step["action_input"][:80]}{"…" if len(step["action_input"])>80 else ""})
                                </div>
                                {obs_html}
                            </div>
                            """, unsafe_allow_html=True)

                # Sources
                if msg.get("sources"):
                    chips = "".join([
                        f'<span class="source-chip">[{i+1}] {s["title"][:48]}{"…" if len(s["title"])>48 else ""}'
                        f'<span class="source-score">{s["score"]:.3f}</span></span>'
                        for i, s in enumerate(msg["sources"][:8])
                    ])
                    st.markdown(
                        f'<div style="margin-left:44px;margin-top:4px">'
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860;margin-bottom:5px;">SOURCES</div>'
                        f'{chips}</div>',
                        unsafe_allow_html=True,
                    )


# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

if not corpus_ready:
    st.error("⚠ No corpus found. Run the data collection script first.", icon="⚠️")
else:
    with st.form("chat_form", clear_on_submit=True):
        c_inp, c_btn = st.columns([8, 1])
        with c_inp:
            user_input = st.text_input(
                "q",
                placeholder="Ask the agent anything about ML research…",
                label_visibility="collapsed",
            )
        with c_btn:
            submitted = st.form_submit_button("➤", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()


# ── Agent processing ───────────────────────────────────────────────────────────
if (
    corpus_ready
    and st.session_state.agent is not None
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    query = st.session_state.messages[-1]["content"]
    agent: ReActAgent = st.session_state.agent

    with st.status("🤖 Agent is thinking…", expanded=True) as status:
        st.write("🗺 Planning query…")
        # Run agent (blocking — Streamlit re-renders after)
        response = agent.run(query)
        st.write(f"✅ Done in {response.total_steps} steps ({response.latency_ms:.0f}ms)")
        status.update(label="Done", state="complete")

    # Serialize scratchpad for storage (dataclasses → dicts)
    scratchpad_dicts = [
        {
            "thought":      s.thought,
            "action":       s.action,
            "action_input": s.action_input,
            "observation":  s.observation,
            "is_final":     s.is_final,
        }
        for s in response.scratchpad
    ]

    st.session_state.messages.append({
        "role":            "assistant",
        "content":         response.answer,
        "scratchpad":      scratchpad_dicts,
        "sources":         response.sources,
        "sub_questions":   response.sub_questions,
        "critique_verdict": response.critique.verdict if response.critique else "pass",
        "total_steps":     response.total_steps,
        "latency_ms":      response.latency_ms,
    })
    st.rerun()
