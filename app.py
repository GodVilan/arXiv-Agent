"""
app.py – Immersive Claude-style UI for arXiv-Agent.
"""
import json
import time
import re
from pathlib import Path
from io import BytesIO

import streamlit as st

from rag import config
from rag.processing.chunker import process_papers, save_chunks, load_chunks
from rag.retrieval.dense import Retriever
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.embeddings import EmbeddingModel
from rag.sources.session_index import SessionIndex
from rag.sources.source_router import SourceRouter, SourceConfig
from rag.sources.arxiv_fetcher import search_arxiv, fetch_paper_chunks
from rag.sources.pdf_uploader import process_upload
from rag.agent import (
    ReActAgent, LiteratureAgent,
    ConversationMemory, ResearchMemory,
)
from rag.sources.project_manager import ProjectManager
from rag.agent.exporter import export_to_docx, render_latex
from rag.agent.citation_formatter import CitationFormatter, PaperMeta
from rag.agent.literature_agent import LiteratureReview, Theme, ThemeSection

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="arXiv Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Instructions/Review Serialization Helpers ───────────────────────────
def serialize_review(review) -> str:
    sections_list = []
    for s in review.sections:
        sections_list.append({
            "theme": {
                "title": s.theme.title,
                "search_query": s.theme.search_query,
                "description": s.theme.description
            },
            "content": s.content,
            "sources": s.sources
        })
        
    references_list = []
    for r in review.references:
        references_list.append({
            "paper_id": r.paper_id,
            "title": r.title,
            "authors": r.authors,
            "abstract": r.abstract,
            "published": r.published,
            "url": r.url,
            "pdf_url": r.pdf_url,
            "source": r.source,
            "venue": r.venue,
            "citation_count": r.citation_count,
            "doi": r.doi
        })
        
    data = {
        "is_literature_review": True,
        "topic": review.topic,
        "citation_format": review.citation_format,
        "introduction": review.introduction,
        "sections": sections_list,
        "gaps": review.gaps,
        "future_work": review.future_work,
        "conclusion": review.conclusion,
        "references": references_list,
        "total_papers": review.total_papers,
        "latency_ms": review.latency_ms
    }
    return json.dumps(data)


def deserialize_review(json_str: str):
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict) or not data.get("is_literature_review"):
            return None
            
        sections = []
        for s in data["sections"]:
            theme = Theme(
                title=s["theme"]["title"],
                search_query=s["theme"]["search_query"],
                description=s["theme"].get("description", "")
            )
            sections.append(ThemeSection(
                theme=theme,
                content=s["content"],
                sources=s.get("sources", [])
            ))
            
        references = []
        for r in data["references"]:
            references.append(PaperMeta(
                paper_id=r["paper_id"],
                title=r["title"],
                authors=r.get("authors") or [],
                abstract=r.get("abstract", ""),
                published=r.get("published", ""),
                url=r.get("url", ""),
                pdf_url=r.get("pdf_url", ""),
                source=r.get("source", "corpus"),
                venue=r.get("venue"),
                citation_count=r.get("citation_count", 0),
                doi=r.get("doi")
            ))
            
        return LiteratureReview(
            topic=data["topic"],
            citation_format=data["citation_format"],
            introduction=data["introduction"],
            sections=sections,
            gaps=data["gaps"],
            future_work=data["future_work"],
            conclusion=data["conclusion"],
            references=references,
            total_papers=data.get("total_papers", 0),
            latency_ms=data.get("latency_ms", 0.0)
        )
    except Exception:
        return None


def hydrate_citations(content: str, papers_by_id: dict) -> str:
    """
    Finds `<span class="citation" data-paper-id="PAPER_ID">` in text and adds
    an inline toggle-able detail card right after it, powered client-side.
    """
    def repl(match):
        full_span = match.group(0)
        paper_id = match.group(1).strip()
        citation_anchor = match.group(2)
        
        p = papers_by_id.get(paper_id)
        if not p:
            return full_span
            
        title = p.get("title", "Unknown Title")
        # Authors format support (could be list of strings or string)
        auths = p.get("authors")
        if isinstance(auths, list):
            authors = ", ".join(auths[:2])
        elif isinstance(auths, str):
            authors = auths
        else:
            authors = "Unknown Authors"
            
        abstract = p.get("abstract", "No abstract available.")
        pdf_url = p.get("pdf_url") or p.get("url") or f"https://arxiv.org/abs/{paper_id}"
        
        clean_abs = abstract.replace('"', '&quot;').replace("'", "&apos;").replace("\n", " ").strip()
        if len(clean_abs) > 260:
            clean_abs = clean_abs[:260] + "..."
            
        card_html = f"""
        <span class="citation-wrapper" style="display:inline; position:relative;">
            <span class="citation" onclick="toggleCitationCard(this)" style="cursor:pointer; color:#7c6fff; font-weight:600; text-decoration:underline;">{citation_anchor}</span>
            <div class="citation-card" style="display:none; background:#0a0a14; border:1px solid rgba(124, 111, 255, 0.3); border-radius:8px; padding:12px; margin-top:6px; margin-bottom:6px; font-size:12px; box-shadow:0 4px 16px rgba(0,0,0,0.6); line-height:1.45; color:#c4c1dc; z-index:100; max-width:450px;">
                <div style="font-weight:700; color:#e8e4ff; margin-bottom:4px; font-size:12.5px;">📋 {title}</div>
                <div style="font-size:10px; color:#a8a5c4; margin-bottom:6px; font-style:italic;">By {authors}</div>
                <div style="margin-bottom:8px; font-size:11px; color:#8e8aa8; line-height:1.4;">{clean_abs}</div>
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:10px; font-family:monospace; margin-top:4px; border-top:1px solid #16162a; padding-top:6px;">
                    <span style="color:#7c6fff; font-weight:600;">ID: {paper_id}</span>
                    <a href="{pdf_url}" target="_blank" style="color:#c084fc; text-decoration:none; font-weight:700;">📄 Open Paper ↗</a>
                </div>
            </div>
        </span>
        """
        return card_html

    # Matches `<span class="citation" data-paper-id="PAPER_ID">(.*?)</span>`
    return re.sub(r'<span class="citation" data-paper-id="(.+?)">(.*?)</span>', repl, content)


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});"></script>

<script>
function toggleCitationCard(element) {
    var card = element.nextElementSibling;
    if (card.style.display === "none" || card.style.display === "") {
        card.style.display = "block";
        card.style.animation = "slideDown 0.25s ease-out forwards";
    } else {
        card.style.display = "none";
    }
}
</script>

<style>
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(0); }
}
.citation {
    cursor: pointer !important;
    color: #7c6fff !important;
    text-decoration: underline !important;
    font-weight: 600 !important;
    transition: color 0.2s ease !important;
}
.citation:hover {
    color: #c084fc !important;
}

@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
#MainMenu,footer,header{ visibility:hidden; }
.block-container{ padding-top:1.5rem; max-width:1400px; }
.stApp{ background:#06060c; }

/* Custom Sleek Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #07070f; }
::-webkit-scrollbar-thumb { background: #1a1a2e; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #7c6fff; }

/* Sidebar Premium Overhaul */
section[data-testid="stSidebar"] {
    background: #080811 !important;
    border-right: 1px solid #16162a !important;
    padding-top: 1rem !important;
}
section[data-testid="stSidebar"] * {
    color: #a8a5c4 !important;
}

/* Premium Sidebar Buttons (Claude-style) */
div[data-testid="stSidebar"] .stButton>button {
    background: rgba(17, 17, 32, 0.4) !important;
    border: 1px solid rgba(124, 111, 255, 0.1) !important;
    color: #a8a5c4 !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-align: left !important;
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    width: 100% !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(124, 111, 255, 0.08) !important;
    border-color: rgba(124, 111, 255, 0.35) !important;
    color: #e8e4ff !important;
    transform: translateY(-0.5px) !important;
    box-shadow: 0 4px 12px rgba(124, 111, 255, 0.12) !important;
}

/* Primary Sidebar Action Buttons (like New Chat / Create Workspace) */
div[data-testid="stSidebar"] .stButton>button[key^="new_"] {
    background: linear-gradient(135deg, rgba(124, 111, 255, 0.12), rgba(192, 132, 252, 0.05)) !important;
    border: 1px solid rgba(124, 111, 255, 0.25) !important;
    color: #e8e4ff !important;
    font-weight: 600 !important;
}
div[data-testid="stSidebar"] .stButton>button[key^="new_"]:hover {
    background: linear-gradient(135deg, rgba(124, 111, 255, 0.2), rgba(192, 132, 252, 0.1)) !important;
    border-color: rgba(124, 111, 255, 0.5) !important;
    box-shadow: 0 4px 16px rgba(124, 111, 255, 0.2) !important;
}

.agent-title{ font-family:'Syne',sans-serif; font-size:2rem; font-weight:700; color:#e8e4ff; letter-spacing:-0.02em; }
.agent-sub{ font-family:'JetBrains Mono',monospace; font-size:10px; color:#4a4870; text-transform:uppercase; letter-spacing:.12em; margin-top:4px; }

/* Fade In Keyframes */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* High-End Chat Bubbles */
.msg-user{ display:flex; justify-content:flex-end; margin:1.2rem 0; animation: fadeIn 0.4s ease forwards; }
.msg-user-bubble{ background: rgba(36, 34, 62, 0.65); border: 1px solid rgba(124, 111, 255, 0.25); backdrop-filter: blur(10px); color:#e8e4ff; padding:12px 18px; border-radius:18px 18px 4px 18px; max-width:85%; font-size:.95rem; line-height:1.65; box-shadow: 0 4px 24px rgba(0,0,0,0.3); }
.msg-assistant{ display:flex; margin:1.2rem 0; gap:12px; align-items:flex-start; animation: fadeIn 0.4s ease forwards; width: 100%; }
.msg-avatar{ width:36px; height:36px; border-radius:10px; background:linear-gradient(135deg,#7c6fff,#c084fc); display:flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0; box-shadow: 0 0 12px rgba(124,111,255,0.4); }
.msg-bubble{ background: rgba(13, 13, 24, 0.55); border: 1px solid rgba(124, 111, 255, 0.14); backdrop-filter: blur(10px); color:#c4c1dc; padding:14px 20px; border-radius:4px 18px 18px 18px; width:100%; font-size:.94rem; line-height:1.75; box-shadow: 0 4px 24px rgba(0,0,0,0.35); }

.scratchpad-step{ background: rgba(10, 10, 22, 0.65); border: 1px solid rgba(124, 111, 255, 0.08); border-radius:8px; padding:12px; margin:6px 0; font-size:12px; }
.step-thought{ color:#a297cb; }
.step-obs{ color:#5f9470; font-size:11px; margin-top:4px; }

.source-chip{ display:inline-block; background:#121223; border:1px solid #20203a; border-radius:6px; padding:4px 10px; margin:3px 3px 3px 0; font-size:11px; color:#8e8aa8; transition: all 0.2s ease; }
.source-chip:hover { border-color: #7c6fff; color: #e8e4ff; }
.source-score{ font-family:'JetBrains Mono',monospace; color:#7c6fff; font-size:10px; margin-left:5px; }
.badge-corpus{ background:#0b192c; border:1px solid #162a4a; color:#5da2fa; border-radius:4px; padding:2px 7px; font-size:10px; }
.badge-arxiv { background:#180b2a; border:1px solid #33164a; color:#be7ffa; border-radius:4px; padding:2px 7px; font-size:10px; }
.badge-upload{ background:#0a1e16; border:1px solid #143a2b; color:#42db7a; border-radius:4px; padding:2px 7px; font-size:10px; }
.badge-pass  { background:#0a1e16; border:1px solid #173b2c; color:#42db7a; border-radius:6px; padding:3px 10px; font-size:11px; }
.badge-retry { background:#1e1008; border:1px solid #3a1e0f; color:#fa8d37; border-radius:6px; padding:3px 10px; font-size:11px; }

.pill{ display:inline-flex; align-items:center; gap:5px; background: rgba(13, 13, 24, 0.6); border: 1px solid rgba(124, 111, 255, 0.08); border-radius:20px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:10px; color:#8e8aa8; margin-right:6px; }
.pill-val{ color:#7c6fff; font-weight:500; }

/* Interactive Hoverable Cards */
.paper-card{ background: rgba(11, 11, 22, 0.5); border: 1px solid rgba(124, 111, 255, 0.1); border-radius:10px; padding:12px 16px; margin-bottom:8px; transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }
.paper-card:hover { border-color: #7c6fff !important; box-shadow: 0 0 15px rgba(124, 111, 255, 0.18); transform: translateY(-1px); }
.paper-title{ color:#c4c1dc; font-weight:600; font-size:14px; }
.paper-meta{ font-family:'JetBrains Mono',monospace; font-size:10px; color:#4e4b78; margin-top:3px; }

.note-card{ background: rgba(11, 11, 22, 0.55); border: 1px solid rgba(124, 111, 255, 0.08); border-radius:8px; padding:10px 14px; margin-bottom:8px; font-size:12px; color:#8e8aa8; transition: all 0.25s ease; }
.note-card:hover { border-color: #7c6fff; }
.note-key{ color:#c4c1dc; font-weight:600; font-size:13px; }

.stTextInput>div>div>input{ background: rgba(11, 11, 22, 0.6) !important; border:1px solid rgba(124, 111, 255, 0.13) !important; border-radius:12px !important; color:#e8e4ff !important; font-size:.95rem !important; padding:13px 17px !important; backdrop-filter: blur(5px); transition: all 0.25s ease; }
.stTextInput>div>div>input:focus{ border-color:#7c6fff !important; box-shadow:0 0 12px rgba(124,111,255,.18) !important; }
.stTextInput>div>div>input::placeholder{ color:#3e3b68 !important; }

.stTextArea>div>div>textarea{ background: rgba(11, 11, 22, 0.6) !important; border:1px solid rgba(124, 111, 255, 0.13) !important; border-radius:12px !important; color:#e8e4ff !important; font-size:.95rem !important; padding:13px 17px !important; backdrop-filter: blur(5px); transition: all 0.25s ease; }
.stTextArea>div>div>textarea:focus{ border-color:#7c6fff !important; box-shadow:0 0 12px rgba(124,111,255,.18) !important; }

.review-box{ background: rgba(11, 11, 22, 0.5); border: 1px solid rgba(124, 111, 255, 0.14); backdrop-filter: blur(12px); border-radius:12px; padding:24px; font-size:.93rem; line-height:1.8; color:#c4c1dc; box-shadow: 0 8px 32px rgba(0,0,0,0.35); }

/* Glowing Active Tab Styling Override */
div[data-baseweb="tab-list"] { gap: 12px; border-bottom: 1px solid #16162a !important; }
button[data-baseweb="tab"] { background: transparent !important; color: #8e8aa8 !important; border: none !important; font-family: 'Syne', sans-serif; font-size: 14px !important; font-weight: 600 !important; padding: 10px 16px !important; transition: all 0.3s ease !important; }
button[data-baseweb="tab"]:hover { color: #e8e4ff !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #e8e4ff !important; text-shadow: 0 0 10px rgba(124, 111, 255, 0.35); }
div[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #7c6fff, #c084fc) !important; height: 2px !important; }

/* Custom Popover Content Borders */
div[data-testid="stPopoverBody"] {
    background: #0d0d19 !important;
    border: 1px solid rgba(124, 111, 255, 0.2) !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

/* 1. Borderless Form Containers */
div[data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* 2. Glassmorphic Expanders */
div[data-testid="stExpander"] {
    background: rgba(13, 13, 24, 0.45) !important;
    border: 1px solid rgba(124, 111, 255, 0.08) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    overflow: hidden !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(124, 111, 255, 0.2) !important;
    box-shadow: 0 4px 20px rgba(124, 111, 255, 0.08) !important;
}
div[data-testid="stExpander"] details {
    border: none !important;
}
div[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #e8e4ff !important;
    padding: 10px 16px !important;
    font-size: 13px !important;
}
div[data-testid="stExpander"] summary:hover {
    color: #7c6fff !important;
}

/* 3. Glowing Gradient Submit Buttons */
.stFormSubmitButton>button {
    background: linear-gradient(135deg, #7c6fff, #c084fc) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-family: 'Syne', sans-serif !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(124, 111, 255, 0.2) !important;
    width: 100% !important;
}
.stFormSubmitButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(124, 111, 255, 0.4) !important;
    color: #ffffff !important;
}

/* 4. Premium Outline Download Buttons */
.stDownloadButton>button {
    background: rgba(124, 111, 255, 0.05) !important;
    border: 1px solid rgba(124, 111, 255, 0.25) !important;
    color: #c4c1dc !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton>button:hover {
    background: rgba(124, 111, 255, 0.12) !important;
    border-color: #7c6fff !important;
    color: #e8e4ff !important;
    box-shadow: 0 4px 12px rgba(124, 111, 255, 0.15) !important;
}
/* 5. Sidebar Flexbox Alignment Rules */
div[data-testid="stSidebar"] [data-testid="stColumn"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="stSidebar"] [data-testid="stColumn"] button {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
div[data-testid="stSidebar"] [data-testid="stColumn"] div[data-testid="stPopover"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
div[data-testid="stSidebar"] [data-testid="stColumn"] div[data-testid="stPopover"]>button {
    padding: 6px 10px !important;
}

/* 6. Slim, borderless micro-buttons for workspace unlinks/deletes */
section:not([data-testid="stSidebar"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(2) [data-testid="stColumn"] button {
    background: transparent !important;
    border: none !important;
    color: #6b68a0 !important;
    padding: 0 !important;
    margin: 0 !important;
    font-size: 11px !important;
    min-height: unset !important;
    height: 20px !important;
    width: 20px !important;
    box-shadow: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1 !important;
    transition: all 0.2s ease !important;
}
section:not([data-testid="stSidebar"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(2) [data-testid="stColumn"] button:hover {
    color: #ff5c5c !important;
    background: rgba(255, 92, 92, 0.1) !important;
    border-radius: 4px !important;
}

/* 7. Align chat expanders under assistant bubbles */
div[data-testid="stColumn"]:nth-of-type(1) div[data-testid="stExpander"] {
    margin-left: 48px !important;
    background: rgba(13, 13, 24, 0.45) !important;
    border: 1px solid rgba(124, 111, 255, 0.08) !important;
}

/* 8. Styled glassmorphic native border container */
div[data-testid="stColumn"]:nth-of-type(1) div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(11, 11, 24, 0.35) !important;
    border: 1px solid rgba(124, 111, 255, 0.12) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-left: 48px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
}
/* Neutralize nesting inner padding on Streamlit vertical blocks */
div[data-testid="stColumn"]:nth-of-type(1) div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-left: 0 !important;
    backdrop-filter: none !important;
    box-shadow: none !important;
}
/* Sleek, uniform compact square buttons and popovers in nested chat columns and header */
div[class*="st-key-star_proj_main_"] button,
div[class*="st-key-config_proj_main_"] button,
div[class*="st-key-settings_proj_main_"] button,
div[class*="st-key-pin_note_"] button,
div[class*="st-key-branch_"] button,
div[class*="st-key-dl_md_"] button,
div[class*="st-key-copy_"] button,
div[class*="st-key-regen_point_"] button {
    padding: 0 !important;
    min-height: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    width: 36px !important;
    font-size: 16px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 8px !important;
    white-space: nowrap !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages":      [],
    "react_agent":   None,
    "lit_agent":     None,
    "conv_memory":   None,
    "res_memory":    None,
    "router":        None,
    "session_idx":   None,
    "bm25":          None,
    "corpus_ready":  False,
    "lit_review":    None,
    "lit_progress":  [],
    "added_papers":  [],
    
    # Project & Thread Workspace Keys
    "pm":            None,
    "active_research_id": None,
    "active_conversation_id": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_chunks():
    cache = config.DATA_DIR / f"chunks_{config.DEFAULT_CHUNK}.json"
    if cache.exists():
        return load_chunks(cache)
    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        papers = json.load(f)
    chunks = process_papers(papers, chunk_size=config.DEFAULT_CHUNK)
    save_chunks(chunks, cache)
    return chunks

@st.cache_resource(show_spinner=False)
def _load_retriever(n: int):
    chunks = _load_chunks()
    return Retriever.build(
        chunks=chunks,
        chunk_size=config.DEFAULT_CHUNK,
        index_dir=config.RESULTS_DIR / "indices",
    )

@st.cache_resource(show_spinner=False)
def _load_bm25(n: int):
    return BM25Retriever(_load_chunks())

@st.cache_resource(show_spinner=False)
def _load_emb_model():
    return EmbeddingModel()

@st.cache_data(show_spinner=False)
def _load_metadata():
    p = config.DATA_DIR / "metadata.json"
    return json.load(open(p)) if p.exists() else []

def submit_query(pm, conversation_id: str, text: str,
                 use_corpus, use_arxiv, use_session) -> None:
    """
    Single entry point for every Q&A composer (landing card + in-thread chat_input).
    Persists the active source toggles, forces Q&A mode, records the user turn, and
    flags the agent to run on the next rerun. Caller is responsible for st.rerun().
    """
    pm.update_conversation_toggles(
        conversation_id, int(bool(use_corpus)), int(bool(use_arxiv)), int(bool(use_session))
    )
    pm.update_conversation_mode(conversation_id, "qa")
    pm.add_message(conversation_id, "user", text.strip())
    st.session_state["trigger_agent_run"] = True


def _auto_title_thread(conversation_id: str, query: str):
    """Automatically generate a neat short title for the conversation thread based on the first query."""
    try:
        from google.genai import types
        from rag.llm import get_client
        client = get_client()
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=f"Summarize this question into a very short, clean title (3-4 words max, no quotes, no period):\n'{query}'",
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=10
            )
        )
        title = resp.text.strip().replace('"', '').replace("'", "")
        if title and st.session_state.pm:
            st.session_state.pm.update_conversation_title(conversation_id, title)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Auto-titling failed: %s", e)

# ── Bootstrap ──────────────────────────────────────────────────────────────────
chunks       = _load_chunks()
corpus_ready = len(chunks) > 0

if corpus_ready and st.session_state.react_agent is None:
    with st.spinner("Loading BGE retriever…"):
        dense = _load_retriever(len(chunks))
    with st.spinner("Building BM25 index…"):
        bm25 = _load_bm25(len(chunks))
    with st.spinner("Initialising session index…"):
        emb_model   = _load_emb_model()
        session_idx = SessionIndex(emb_model)
        router      = SourceRouter(dense, session_idx)

    conv_mem = ConversationMemory()
    res_mem  = ResearchMemory()

    st.session_state.conv_memory  = conv_mem
    st.session_state.res_memory   = res_mem
    st.session_state.session_idx  = session_idx
    st.session_state.router       = router
    st.session_state.bm25         = bm25
    st.session_state.react_agent  = ReActAgent(router, bm25, session_idx, conv_mem, res_mem)
    st.session_state.lit_agent    = LiteratureAgent(router, bm25, session_idx)
    st.session_state.pm           = ProjectManager()
    
    # Resolve active research project and conversation thread dynamically
    researches = st.session_state.pm.list_researches()
    if researches:
        if st.session_state.active_research_id not in [r["research_id"] for r in researches]:
            st.session_state.active_research_id = researches[0]["research_id"]
            
        threads = st.session_state.pm.list_conversations(st.session_state.active_research_id)
        if threads:
            if st.session_state.active_conversation_id not in [t["conversation_id"] for t in threads]:
                st.session_state.active_conversation_id = threads[0]["conversation_id"]
        else:
            st.session_state.active_conversation_id = None
    else:
        st.session_state.active_research_id = None
        st.session_state.active_conversation_id = None
        
    st.session_state.corpus_ready = True

if corpus_ready and st.session_state.pm is None:
    st.session_state.pm = ProjectManager()
    researches = st.session_state.pm.list_researches()
    if researches:
        if st.session_state.active_research_id not in [r["research_id"] for r in researches]:
            st.session_state.active_research_id = researches[0]["research_id"]
        threads = st.session_state.pm.list_conversations(st.session_state.active_research_id)
        if threads:
            if st.session_state.active_conversation_id not in [t["conversation_id"] for t in threads]:
                st.session_state.active_conversation_id = threads[0]["conversation_id"]
        else:
            st.session_state.active_conversation_id = None
    else:
        st.session_state.active_research_id = None
        st.session_state.active_conversation_id = None

pm: ProjectManager = st.session_state.pm
session_idx: SessionIndex = st.session_state.session_idx

# ── Sidebar Overhaul ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:1.25rem;font-weight:700;color:#e8e4ff;margin-bottom:2px;letter-spacing:-0.01em;">🤖 arXiv Agent</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860;text-transform:uppercase;letter-spacing:.12em;margin-bottom:1.5rem;">BGE · ReAct · Gemini 2.5</p>', unsafe_allow_html=True)

    if pm:
        # ── ACTION 1: NEW CHAT ─────────────────────────────────────────────────
        if st.session_state.active_research_id:
            if st.button("💬 New chat", use_container_width=True, key="new_chat_sidebar_btn"):
                new_c_id = pm.create_conversation(st.session_state.active_research_id, "New Thread")
                if new_c_id:
                    st.session_state.active_conversation_id = new_c_id
                    st.rerun()
        else:
            st.button("💬 New chat (Select/Create Project)", use_container_width=True, disabled=True, key="new_chat_sidebar_btn_disabled")

        # ── GLOBAL SIDEBAR SEARCH ──────────────────────────────────────────────
        st.markdown("<p style='font-size:10px;font-family:JetBrains Mono,monospace;color:#7c6fff;margin-bottom:4px;text-transform:uppercase;font-weight:600;'>🔍 Search Messages</p>", unsafe_allow_html=True)
        search_query = st.text_input("Search Messages", placeholder="e.g. QLoRA loss...", key="global_sidebar_search_input", label_visibility="collapsed")
        if search_query.strip():
            matches = pm.search_messages(search_query.strip(), st.session_state.active_research_id)
            if matches:
                st.markdown(f"<p style='font-size:10px;color:#42db7a;font-style:italic;margin-bottom:6px;'>Found {len(matches)} matches:</p>", unsafe_allow_html=True)
                for idx_m, match in enumerate(matches[:5]):
                    m_title = match["conversation_title"]
                    m_content = match["content"][:20] + "..."
                    if st.button(f"🔍 {m_title} | {m_content}", key=f"search_match_{match['message_id']}_{idx_m}", use_container_width=True):
                        st.session_state.active_conversation_id = match["conversation_id"]
                        st.rerun()
            else:
                st.markdown("<p style='font-size:10px;color:#ff5c5c;font-style:italic;margin-bottom:4px;'>No matches found.</p>", unsafe_allow_html=True)
        st.divider()

        researches = pm.list_researches()
        research_names = [r["name"] for r in researches]

        # ── COLLAPSIBLE: STARRED DIRECTORY ─────────────────────────────────────
        starred_projects = [r for r in researches if r["is_starred"]]
        all_threads = pm.list_conversations(st.session_state.active_research_id)
        starred_threads = [t for t in all_threads if t["is_starred"]]
        recent_threads = [t for t in all_threads if not t["is_starred"]]

        if starred_projects or starred_threads:
            with st.expander("⭐ Starred", expanded=True):
                if starred_projects:
                    st.markdown("<p style='font-size:10px;font-family:JetBrains Mono,monospace;color:#7c6fff;margin-bottom:4px;text-transform:uppercase;font-weight:600;'>Projects</p>", unsafe_allow_html=True)
                    for rp in starred_projects:
                        is_active_proj = rp["research_id"] == st.session_state.active_research_id
                        p_label = f"📁 {rp['name']}"
                        
                        col_btn, col_pop = st.columns([8, 2])
                        with col_btn:
                            if st.button(p_label, key=f"starred_proj_btn_{rp['research_id']}", use_container_width=True):
                                st.session_state.active_research_id = rp["research_id"]
                                sibling_threads = pm.list_conversations(rp["research_id"])
                                st.session_state.active_conversation_id = sibling_threads[0]["conversation_id"] if sibling_threads else None
                                st.rerun()
                        with col_pop:
                            with st.popover("⋮", key=f"starred_proj_pop_{rp['research_id']}"):
                                st.markdown("**Project Settings**")
                                # Star check
                                p_star = st.checkbox("Starred", value=rp["is_starred"], key=f"starred_proj_star_{rp['research_id']}")
                                if p_star != rp["is_starred"]:
                                    pm.update_research_starred(rp["research_id"], p_star)
                                    st.rerun()
                                # Rename
                                p_rename = st.text_input("Rename Project", value=rp["name"], key=f"starred_proj_rename_{rp['research_id']}")
                                if st.button("Save Name", key=f"starred_proj_save_{rp['research_id']}", use_container_width=True):
                                    if p_rename.strip():
                                        pm.update_research_name(rp["research_id"], p_rename.strip())
                                        st.rerun()
                                st.divider()
                                # Delete
                                if st.button("🗑️ Delete Project", key=f"starred_proj_del_{rp['research_id']}", use_container_width=True):
                                    pm.delete_research(rp["research_id"])
                                    remaining = pm.list_researches()
                                    if remaining:
                                        st.session_state.active_research_id = remaining[0]["research_id"]
                                        sibling_threads = pm.list_conversations(remaining[0]["research_id"])
                                        st.session_state.active_conversation_id = sibling_threads[0]["conversation_id"] if sibling_threads else None
                                    else:
                                        st.session_state.active_research_id = None
                                        st.session_state.active_conversation_id = None
                                    st.rerun()
                
                if starred_threads:
                    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:10px;font-family:JetBrains Mono,monospace;color:#7c6fff;margin-bottom:4px;text-transform:uppercase;font-weight:600;'>Chats</p>", unsafe_allow_html=True)
                    for t in starred_threads:
                        lbl = f"⭐ {t['title'][:20]}…" if len(t['title']) > 20 else f"⭐ {t['title']}"
                        
                        col_btn, col_pop = st.columns([8, 2])
                        with col_btn:
                            if st.button(lbl, key=f"starred_th_btn_{t['conversation_id']}", use_container_width=True):
                                st.session_state.active_conversation_id = t["conversation_id"]
                                st.rerun()
                        with col_pop:
                            with st.popover("⋮", key=f"starred_th_pop_{t['conversation_id']}"):
                                st.markdown("**Thread Settings**")
                                star_state = st.checkbox("Starred", value=t["is_starred"], key=f"star_th_check_{t['conversation_id']}")
                                if star_state != t["is_starred"]:
                                    pm.update_conversation_starred(t["conversation_id"], star_state)
                                    st.rerun()
                                rename_title = st.text_input("Rename Thread", value=t["title"], key=f"starred_rename_{t['conversation_id']}")
                                if st.button("Save Title", key=f"starred_save_btn_{t['conversation_id']}", use_container_width=True):
                                    if rename_title.strip():
                                        pm.update_conversation_title(t["conversation_id"], rename_title.strip())
                                        st.rerun()
                                move_proj_name = st.selectbox("Move to Project", options=research_names, index=research_names.index(pm.get_research(st.session_state.active_research_id)["name"]), key=f"starred_move_select_{t['conversation_id']}")
                                if st.button("Move Thread", key=f"starred_move_btn_{t['conversation_id']}", use_container_width=True):
                                    target_p = researches[research_names.index(move_proj_name)]
                                    pm.move_conversation_to_project(t["conversation_id"], target_p["research_id"])
                                    st.rerun()
                                st.divider()
                                if st.button("🗑️ Delete Thread", key=f"starred_del_btn_{t['conversation_id']}", use_container_width=True):
                                    pm.delete_conversation(t["conversation_id"])
                                    sibling_threads = pm.list_conversations(st.session_state.active_research_id)
                                    st.session_state.active_conversation_id = sibling_threads[0]["conversation_id"] if sibling_threads else None
                                    st.rerun()
            st.divider()

        # ── COLLAPSIBLE: WORKSPACE PROJECTS DIRECTORY ──────────────────────────
        with st.expander("📁 Projects", expanded=True):
            for r in researches:
                # Highlight active
                is_act_proj = r["research_id"] == st.session_state.active_research_id
                bullet = "● " if is_act_proj else "○ "
                proj_label = f"{bullet}{r['name']}"
                
                col_btn, col_pop = st.columns([8, 2])
                with col_btn:
                    if st.button(proj_label, key=f"proj_side_btn_{r['research_id']}", use_container_width=True):
                        st.session_state.active_research_id = r["research_id"]
                        sibling_threads = pm.list_conversations(r["research_id"])
                        st.session_state.active_conversation_id = sibling_threads[0]["conversation_id"] if sibling_threads else None
                        st.rerun()
                with col_pop:
                    with st.popover("⋮", key=f"proj_side_pop_{r['research_id']}"):
                        st.markdown("**Project Settings**")
                        # Star
                        proj_starred = st.checkbox("Starred", value=r["is_starred"], key=f"proj_side_star_{r['research_id']}")
                        if proj_starred != r["is_starred"]:
                            pm.update_research_starred(r["research_id"], proj_starred)
                            st.rerun()
                        # Rename
                        rename_pname = st.text_input("Rename Project", value=r["name"], key=f"proj_side_rename_{r['research_id']}")
                        if st.button("Save Name", key=f"proj_side_save_{r['research_id']}", use_container_width=True):
                            if rename_pname.strip():
                                pm.update_research_name(r["research_id"], rename_pname.strip())
                                st.rerun()
                        st.divider()
                        # Delete
                        if st.button("🗑️ Delete Project", key=f"proj_side_del_{r['research_id']}", use_container_width=True):
                            pm.delete_research(r["research_id"])
                            remaining = pm.list_researches()
                            if remaining:
                                st.session_state.active_research_id = remaining[0]["research_id"]
                                sibling_threads = pm.list_conversations(remaining[0]["research_id"])
                                st.session_state.active_conversation_id = sibling_threads[0]["conversation_id"] if sibling_threads else None
                            else:
                                st.session_state.active_research_id = None
                                st.session_state.active_conversation_id = None
                            st.rerun()

            # Create project workspace button
            with st.popover("➕ Create Project", use_container_width=True):
                with st.form("create_proj_side_form", clear_on_submit=True):
                    new_pname = st.text_input("New Project Name", placeholder="e.g. LLM Efficiency", key="new_proj_name_side_input")
                    new_pscope = st.radio("Data Scope", options=["User papers only", "User papers + Corpus + Live arXiv"], index=1, key="new_proj_scope_side")
                    create_side_clicked = st.form_submit_button("Create Project", use_container_width=True)
                    if create_side_clicked:
                        if new_pname.strip():
                            scope_type = "user_only" if new_pscope == "User papers only" else "all_sources"
                            created_id = pm.create_research(new_pname.strip(), scope_type)
                            if created_id:
                                pm.create_conversation(created_id, "General Discussion")
                                st.session_state.active_research_id = created_id
                                sibling_threads = pm.list_conversations(created_id)
                                st.session_state.active_conversation_id = sibling_threads[0]["conversation_id"] if sibling_threads else None
                                st.rerun()
                        else:
                            st.warning("Enter a name.")

        st.divider()

        # ── COLLAPSIBLE: RECENTS CHATS DIRECTORY ───────────────────────────────
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:9px;color:#4e4b78;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Recents</p>', unsafe_allow_html=True)
        if recent_threads:
            for t in recent_threads:
                is_active = t["conversation_id"] == st.session_state.active_conversation_id
                label = f"● {t['title'][:20]}" if is_active else f"○ {t['title'][:20]}"
                if len(t['title']) > 20:
                    label += "…"
                
                col_btn, col_pop = st.columns([8, 2])
                with col_btn:
                    if st.button(label, key=f"thread_{t['conversation_id']}", use_container_width=True):
                        st.session_state.active_conversation_id = t["conversation_id"]
                        st.rerun()
                with col_pop:
                    with st.popover("⋮", key=f"th_pop_{t['conversation_id']}"):
                        st.markdown("**Thread Settings**")
                        # Star/Unstar
                        star_state = st.checkbox("Starred", value=t["is_starred"], key=f"star_th_check_{t['conversation_id']}")
                        if star_state != t["is_starred"]:
                            pm.update_conversation_starred(t["conversation_id"], star_state)
                            st.rerun()
                        # Rename
                        rename_title = st.text_input("Rename Thread", value=t["title"], key=f"rename_{t['conversation_id']}")
                        if st.button("Save Title", key=f"save_btn_{t['conversation_id']}", use_container_width=True):
                            if rename_title.strip():
                                pm.update_conversation_title(t["conversation_id"], rename_title.strip())
                                st.rerun()
                        # Move Project
                        move_proj_name = st.selectbox("Move to Project", options=research_names, index=research_names.index(pm.get_research(st.session_state.active_research_id)["name"]), key=f"move_select_{t['conversation_id']}")
                        if st.button("Move Thread", key=f"move_btn_{t['conversation_id']}", use_container_width=True):
                            target_p = researches[research_names.index(move_proj_name)]
                            pm.move_conversation_to_project(t["conversation_id"], target_p["research_id"])
                            st.rerun()
                        st.divider()
                        # Delete
                        if st.button("🗑️ Delete Thread", key=f"del_btn_{t['conversation_id']}", use_container_width=True):
                            pm.delete_conversation(t["conversation_id"])
                            sibling_threads = pm.list_conversations(st.session_state.active_research_id)
                            st.session_state.active_conversation_id = sibling_threads[0]["conversation_id"] if sibling_threads else None
                            st.rerun()
        else:
            st.markdown('<p style="color:#4a4870;font-size:12px;font-style:italic;text-align:center;">No recent chats in this project.</p>', unsafe_allow_html=True)

        st.divider()

    # Reset System Button
    if st.button("🔄 Full Reset System", use_container_width=True):
        # Drop SQLite and indices
        db_file = config.DATA_DIR / "session_papers.db"
        faiss_file = config.RESULTS_DIR / "indices" / "session_index.faiss"
        faiss_meta = config.RESULTS_DIR / "indices" / "session_index_meta.pkl"
        if db_file.exists():
            try:
                db_file.unlink()
            except:
                pass
        for f in [faiss_file, faiss_meta]:
            if f.exists():
                try:
                    f.unlink()
                except:
                    pass

        for k in ["react_agent", "lit_agent", "router", "session_idx", "bm25",
                  "messages", "lit_review", "lit_progress", "added_papers", "pm",
                  "active_research_id", "active_conversation_id"]:
            if k in st.session_state:
                if k == "active_research_id":
                    st.session_state[k] = None
                else:
                    st.session_state[k] = None if k not in ["messages", "lit_progress", "added_papers"] else []
        st.rerun()

# ── MAIN LAYOUT ────────────────────────────────────────────────────────────────
active_r = pm.get_research(st.session_state.active_research_id) if (pm and st.session_state.active_research_id) else None
active_cid = st.session_state.active_conversation_id

if active_r and active_cid and pm:
    # 70% Chat pane, 30% Right-hand Library Workspace sidebar
    col_chat, col_workspace = st.columns([7, 3])

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN: THE ACTIVE CHAT TIMELINE
    # ══════════════════════════════════════════════════════════════════════════
    with col_chat:
        # Load active conversation and messages first so they are available for header config
        active_conv = pm.get_conversation(active_cid)
        messages = pm.get_messages(active_cid)
        # Project Title Header with Config popover merged (icon-only buttons)
        c_title, c_star, c_config, c_settings = st.columns([7, 1, 1, 1])
        with c_title:
            st.markdown(f"<h2 style='margin:0;font-family:Syne,sans-serif;color:#e8e4ff;font-size:20px;line-height:1.2;'>📁 {active_r['name']}</h2>", unsafe_allow_html=True)
        with c_star:
            star_icon = "⭐" if active_r["is_starred"] else "☆"
            if st.button(star_icon, key=f"star_proj_main_{active_r['research_id']}", use_container_width=True):
                pm.update_research_starred(active_r["research_id"], not active_r["is_starred"])
                st.rerun()
        with c_config:
            with st.popover("⚙️", use_container_width=True, key=f"config_proj_main_{active_r['research_id']}"):
                st.markdown("**Conversation Settings**")
                if active_conv:
                    sel_mode = st.selectbox(
                        "Mode Selector",
                        options=["💬 Q&A Mode", "📖 Literature Review Mode"],
                        index=0 if active_conv["mode"] == "qa" else 1,
                        key="chat_mode_selector_dropdown",
                        label_visibility="collapsed"
                    )
                    new_mode = "qa" if sel_mode == "💬 Q&A Mode" else "lit_review"
                    if new_mode != active_conv["mode"]:
                        pm.update_conversation_mode(active_cid, new_mode)
                        st.rerun()
                    
                    st.divider()
                    st.markdown("**Active Data Sources**")
                    cb_corpus = st.checkbox("📚 Corpus", value=bool(active_conv.get("use_corpus", 1)), key="cb_corpus_active")
                    cb_arxiv = st.checkbox("🌐 arXiv", value=bool(active_conv.get("use_arxiv", 1)), key="cb_arxiv_active")
                    cb_session = st.checkbox("📄 Uploads", value=bool(active_conv.get("use_session", 1)), key="cb_session_active")
                    
                    if (cb_corpus != bool(active_conv.get("use_corpus", 1)) or
                        cb_arxiv != bool(active_conv.get("use_arxiv", 1)) or
                        cb_session != bool(active_conv.get("use_session", 1))):
                        pm.update_conversation_toggles(active_cid, int(cb_corpus), int(cb_arxiv), int(cb_session))
                        st.rerun()
        with c_settings:
            with st.popover("⋮", use_container_width=True, key=f"settings_proj_main_{active_r['research_id']}"):
                st.markdown("**Workspace Options**")
                # Rename Project
                rename_proj_name = st.text_input("Rename Project", value=active_r["name"], key="rename_proj_main_input")
                if st.button("Save Workspace Name", key="save_proj_name_main_btn", use_container_width=True):
                    if rename_proj_name.strip():
                        pm.update_research_name(active_r["research_id"], rename_proj_name.strip())
                        st.rerun()
                st.divider()
                # Delete Project
                if st.button("🗑️ Delete Project", key="delete_proj_main_btn", use_container_width=True):
                    pm.delete_research(active_r["research_id"])
                    remaining = pm.list_researches()
                    if remaining:
                        st.session_state.active_research_id = remaining[0]["research_id"]
                        sibling_threads = pm.list_conversations(remaining[0]["research_id"])
                        st.session_state.active_conversation_id = sibling_threads[0]["conversation_id"] if sibling_threads else None
                    else:
                        st.session_state.active_research_id = None
                        st.session_state.active_conversation_id = None
                    st.rerun()

        st.markdown("<hr style='border:none;border-top:1px solid #16162a;margin:12px 0 16px;'>", unsafe_allow_html=True)

        # ── DUAL STATE LANDING VIEW ───────────────────────────────────────────
        if not messages:
            # Show a beautiful centered Q&A or Literature Review prompt card
            with st.container(border=True):
                st.markdown(f"""
                <h3 style="font-family:Syne,sans-serif;color:#e8e4ff;margin-top:0;margin-bottom:12px;font-size:22px;text-align:center;">What are we researching today?</h3>
                """, unsafe_allow_html=True)
                
                # Premium drag-and-drop PDF dropzone
                st.markdown("""
                <div style="border: 2px dashed rgba(124, 111, 255, 0.35); border-radius: 12px; padding: 24px; text-align: center; background: rgba(124, 111, 255, 0.02); margin-bottom: 20px;">
                    <p style="font-size: 28px; margin: 0 0 8px;">📤</p>
                    <h5 style="font-family:'Syne',sans-serif; color:#e8e4ff; margin:0 0 4px; font-size:14px;">Drag & Drop Research PDFs</h5>
                    <p style="color:#8e8aa8; font-size:11px; margin:0 0 12px;">Add custom PDFs to parse & query instantly inside this project workspace.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if "uploader_reset_ctr" not in st.session_state:
                    st.session_state.uploader_reset_ctr = 0
                
                uploader_key = f"landing_dropzone_uploader_{st.session_state.uploader_reset_ctr}"
                uploaded_landing_file = st.file_uploader("Upload PDF here", type=["pdf"], key=uploader_key, label_visibility="collapsed")
                if uploaded_landing_file:
                    with st.spinner("Parsing and embedding PDF..."):
                        chunks_up, meta_up = process_upload(uploaded_landing_file.read(), uploaded_landing_file.name, "", "")
                        if chunks_up:
                            session_idx.add_chunks(chunks_up, meta_up)
                            pm.add_paper_to_research(active_r["research_id"], meta_up["paper_id"])
                            st.session_state.uploader_reset_ctr += 1
                            st.toast(f"📄 Added paper: {meta_up['title']}!")
                            st.rerun()
                
                st.markdown("<hr style='border:none;border-top:1px solid #16162a;margin:16px 0;'>", unsafe_allow_html=True)
                
                with st.form("centered_prompt_box_form", clear_on_submit=True):
                    # Mode Selector inside prompt card
                    sel_mode = st.radio("Choose Mode", ["💬 Q&A Mode", "📖 Literature Review Mode"], index=0 if active_conv["mode"] == "qa" else 1, horizontal=True)
                    
                    st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7c6fff;text-transform:uppercase;margin-top:8px;margin-bottom:4px;'>Active Data Sources</div>", unsafe_allow_html=True)
                    col_t1, col_t2, col_t3 = st.columns(3)
                    with col_t1:
                        cb_corpus = st.checkbox("📚 Pre-indexed Corpus (150 papers)", value=bool(active_conv.get("use_corpus", 1)) if active_conv else True, key="cb_corpus_landing")
                    with col_t2:
                        cb_arxiv = st.checkbox("🌐 Live arXiv Fetch", value=bool(active_conv.get("use_arxiv", 1)) if active_conv else True, key="cb_arxiv_landing")
                    with col_t3:
                        cb_session = st.checkbox("📄 Uploaded Workspace Papers", value=bool(active_conv.get("use_session", 1)) if active_conv else True, key="cb_session_landing")
                    
                    # Dynamic inputs inside form
                    if sel_mode == "💬 Q&A Mode":
                        st.markdown("<p style='font-size:12px;color:#8e8aa8;'>Ask questions to reason step-by-step over the papers associated with this workspace project.</p>", unsafe_allow_html=True)
                        user_query = st.text_input("Question", placeholder="e.g. Compare LoRA and prefix tuning on parameter efficiency...", key="centered_qa_input")
                        submit_clicked = st.form_submit_button("➤ Ask Agent", use_container_width=True)
                        
                        if submit_clicked and user_query.strip():
                            submit_query(pm, active_cid, user_query,
                                         cb_corpus, cb_arxiv, cb_session)
                            st.rerun()
                    else:
                        st.markdown("<p style='font-size:12px;color:#8e8aa8;'>Generate a full academic literature review with citation formatting (APA, Chicago, IEEE, etc.).</p>", unsafe_allow_html=True)
                        lit_topic = st.text_input("Research Topic", placeholder="e.g. Catastrophic Forgetting in Continual Learning", key="centered_lit_topic")
                        c1, c2 = st.columns(2)
                        with c1:
                            citation_fmt = st.selectbox("Citation Format", config.CITATION_FORMATS, key="centered_lit_format")
                        with c2:
                            n_themes = st.slider("Number of Themes", 3, 6, 4, key="centered_lit_themes")
                            
                        submit_clicked = st.form_submit_button("🚀 Generate Review", use_container_width=True)
                        
                        if submit_clicked and lit_topic.strip() and corpus_ready and st.session_state.lit_agent:
                            # Save toggles
                            pm.update_conversation_toggles(active_cid, int(cb_corpus), int(cb_arxiv), int(cb_session))
                            # Set mode
                            pm.update_conversation_mode(active_cid, "lit_review")
                            progress_placeholder = st.empty()
                            progress_log: list[str] = []
                            def _progress(msg: str):
                                progress_log.append(msg)
                                with progress_placeholder.container():
                                    for m in progress_log[-8:]:
                                        st.write(m)
                                        
                            with st.spinner("Analyzing topic & researching..."):
                                # Resolve scopes
                                allowed_pids = None
                                if active_r and active_r["scope_type"] == "user_only":
                                    allowed_pids = pm.get_research_paper_ids(active_r["research_id"])
                                    if not allowed_pids:
                                        allowed_pids = {"dummy_no_papers_associated"}
                                        
                                source_cfg = SourceConfig(
                                    use_corpus=bool(cb_corpus),
                                    use_session=bool(cb_session)
                                )
                                
                                review_obj = st.session_state.lit_agent.run(
                                    topic=lit_topic.strip(),
                                    citation_format=citation_fmt,
                                    n_themes=n_themes,
                                    use_live_arxiv=bool(cb_arxiv),
                                    allowed_paper_ids=allowed_pids,
                                    source_cfg=source_cfg,
                                    progress_cb=_progress
                                )
                                # Save to message logs as JSON
                                serialized_review = serialize_review(review_obj)
                                pm.add_message(active_cid, "user", f"Generate a literature review on: {lit_topic.strip()}")
                                pm.add_message(active_cid, "assistant", serialized_review, message_type="lit_review")
                                
                                # Auto-title thread
                                _auto_title_thread(active_cid, f"Review: {lit_topic.strip()}")
                                
                                # Trigger memory consolidator daemon
                                try:
                                    from rag.agent.memory_consolidator import MemoryConsolidator
                                    mc = MemoryConsolidator(pm)
                                    mc.consolidate(active_r["research_id"], f"Literature review topic: {lit_topic.strip()}", serialized_review)
                                except Exception as mc_err:
                                    import logging
                                    logging.getLogger(__name__).warning("Memory consolidator failed in review: %s", mc_err)
                                    
                                st.rerun()

            # Render styled list of recent thread items inside this project
            sibling_chats = [t for t in all_threads if t["conversation_id"] != active_cid]
            if sibling_chats:
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:12px;font-family:Syne,sans-serif;color:#a8a5c4;font-weight:600;margin-bottom:8px;'>Recent conversations in this project:</p>", unsafe_allow_html=True)
                for sc in sibling_chats[:3]:
                    sc_lbl = f"⭐ {sc['title']}" if sc["is_starred"] else f"💬 {sc['title']}"
                    if st.button(sc_lbl, key=f"centered_siblings_{sc['conversation_id']}", use_container_width=True):
                        st.session_state.active_conversation_id = sc["conversation_id"]
                        st.rerun()
        # ── STANDARD ACTIVE TIMELINE VIEW ─────────────────────────────────────
        else:
            # Render message logs
            chat_container = st.container()
            
            # Build global papers catalog for citation detail expansion
            papers_by_id = {}
            for p in _load_metadata():
                papers_by_id[p["paper_id"]] = p
            if session_idx:
                for p in session_idx.list_papers():
                    papers_by_id[p["paper_id"]] = p
                    
            with chat_container:
                for idx_m, msg in enumerate(messages):
                    if msg["role"] == "user":
                        with st.chat_message("user", avatar="🧑"):
                            st.markdown(msg["content"])
                    else:
                        # Dispatch on the explicit message_type (legacy reviews were
                        # backfilled during the DB migration) — no more JSON-sniffing.
                        lit_review_obj = (
                            deserialize_review(msg["content"])
                            if msg.get("message_type") == "lit_review" else None
                        )
                        if lit_review_obj:
                            # Render inline literature review tabbed board
                            st.markdown(f"""
                            <div class="msg-assistant">
                                <div class="msg-avatar">📖</div>
                                <div style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;color:#e8e4ff;margin-top:6px;margin-bottom:10px;">📋 Literature Review: {lit_review_obj.topic}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.container(border=True):
                                col_la, col_lb = st.columns([1, 1])
                                with col_la:
                                    docx_buffer = BytesIO()
                                    export_to_docx(lit_review_obj, docx_buffer)
                                    st.download_button(
                                        label="📥 Word (.docx)",
                                        data=docx_buffer.getvalue(),
                                        file_name=f"literature_review_{lit_review_obj.citation_format}.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        key=f"dl_docx_{msg['message_id']}",
                                        use_container_width=True
                                    )
                                with col_lb:
                                    st.download_button(
                                        label="📥 LaTeX (.tex)",
                                        data=render_latex(lit_review_obj),
                                        file_name=f"literature_review_{lit_review_obj.citation_format}.tex",
                                        mime="application/x-tex",
                                        key=f"dl_tex_{msg['message_id']}",
                                        use_container_width=True
                                    )
                                    
                                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                                
                                theme_tabs = st.tabs(
                                    ["1. Introduction"] +
                                    [f"{idx}. {sec.theme.title[:20]}…" for idx, sec in enumerate(lit_review_obj.sections, 2)] +
                                    ["Gaps", "Future Work", "Conclusion", "References"]
                                )
                                
                                with theme_tabs[0]:
                                    st.markdown(lit_review_obj.introduction)
                                    
                                for idx, sec in enumerate(lit_review_obj.sections):
                                    offset_idx = idx + 1
                                    with theme_tabs[offset_idx]:
                                        st.markdown(sec.content)
                                        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                                        
                                        col_sa, col_sb = st.columns([7, 3])
                                        with col_sb:
                                            if st.button("🔄 Regenerate Section", key=f"regen_{msg['message_id']}_{idx}", use_container_width=True):
                                                with st.spinner("Regenerating section content…"):
                                                    # Resolve scopes
                                                    allowed_pids = None
                                                    if active_r and active_r["scope_type"] == "user_only":
                                                        allowed_pids = pm.get_research_paper_ids(active_r["research_id"])
                                                        if not allowed_pids:
                                                            allowed_pids = {"dummy_no_papers_associated"}
                                                            
                                                    source_cfg = SourceConfig(
                                                        use_corpus=bool(active_conv.get("use_corpus", 1)) if active_conv else True,
                                                        use_session=bool(active_conv.get("use_session", 1)) if active_conv else True
                                                    )
                                                    
                                                    updated_review = st.session_state.lit_agent.regenerate_section(
                                                        review=lit_review_obj,
                                                        section_idx=idx,
                                                        use_live_arxiv=bool(active_conv.get("use_arxiv", 1)),
                                                        allowed_paper_ids=allowed_pids,
                                                        source_cfg=source_cfg
                                                    )
                                                    new_json_str = serialize_review(updated_review)
                                                    pm.update_message_content(msg["message_id"], new_json_str)
                                                    st.success("Section updated successfully!")
                                                    st.rerun()
                                        if sec.sources:
                                            st.markdown("**Sources used in this section:**")
                                            chips = "".join(
                                                f'<span class="source-chip">{s["title"][:50]}<span class="source-score">{s["score"]:.3f}</span></span>'
                                                for s in sec.sources[:6]
                                            )
                                            st.markdown(chips, unsafe_allow_html=True)
                                            
                                offset = len(lit_review_obj.sections) + 1
                                with theme_tabs[offset]:
                                    st.markdown(lit_review_obj.gaps)
                                with theme_tabs[offset + 1]:
                                    st.markdown(lit_review_obj.future_work)
                                with theme_tabs[offset + 2]:
                                    st.markdown(lit_review_obj.conclusion)
                                with theme_tabs[offset + 3]:
                                    if lit_review_obj.references:
                                        fmt_ref = CitationFormatter(lit_review_obj.citation_format)
                                        ref_text = fmt_ref.format_list(lit_review_obj.references)
                                        st.markdown(ref_text.replace(chr(10), "<br>"), unsafe_allow_html=True)
                                    else:
                                        st.info("No references tracked.")
                        else:
                            hydrated_content = hydrate_citations(msg["content"], papers_by_id)
                            with st.chat_message("assistant", avatar="🤖"):
                                # Render the answer as genuine markdown. Content is NOT wrapped
                                # in a block-level <div> anymore, so headings/lists/code/tables
                                # from the model parse correctly; inline citation spans still pass through.
                                st.markdown(hydrated_content, unsafe_allow_html=True)

                            scratchpad_obj = msg.get("scratchpad")
                            scratchpad_list = []
                            sub_questions = []
                            critique_verdict = "pass"
                            total_steps = 0
                            latency_ms = 0
                            out_of_scope = False
                            sources = []
                            
                            if isinstance(scratchpad_obj, dict):
                                scratchpad_list = scratchpad_obj.get("scratchpad", [])
                                sub_questions = scratchpad_obj.get("sub_questions", [])
                                critique_verdict = scratchpad_obj.get("critique_verdict", "pass")
                                total_steps = scratchpad_obj.get("total_steps", 0)
                                latency_ms = scratchpad_obj.get("latency_ms", 0)
                                out_of_scope = scratchpad_obj.get("out_of_scope", False)
                                sources = scratchpad_obj.get("sources", [])
                            elif isinstance(scratchpad_obj, list):
                                scratchpad_list = scratchpad_obj
                                
                            if not out_of_scope:
                                if sub_questions:
                                    tags = "".join(f'<span style="display:inline-block;background:#12122a;border:1px solid #2a2a50;border-radius:6px;padding:3px 9px;font-size:11px;color:#7c6fff;margin:2px">{q}</span>' for q in sub_questions)
                                    st.markdown(f'<div style="margin-left:48px;margin-top:-6px;margin-bottom:6px"><span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860">DECOMPOSED: </span>{tags}</div>', unsafe_allow_html=True)

                                if critique_verdict:
                                    cls  = "badge-pass" if critique_verdict == "pass" else "badge-retry"
                                    txt  = "✅ Self-critique passed" if critique_verdict == "pass" else "🔄 Refined after critique"
                                    st.markdown(f'<div style="margin-left:48px;margin-bottom:6px"><span class="{cls}">{txt}</span> <span style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860">{total_steps} steps · {latency_ms:.0f}ms</span></div>', unsafe_allow_html=True)

                                if scratchpad_list:
                                    with st.expander(f"🔍 Agent Reasoning ({len(scratchpad_list)} steps)"):
                                        for i, step in enumerate(scratchpad_list, 1):
                                            action_color = "#4ade80" if step["action"] == "finish" else "#7c6fff"
                                            obs_html = f'<div class="step-obs">📋 {step["observation"][:300]}{"…" if len(step["observation"])>300 else ""}</div>' if step.get("observation") else ""
                                            st.markdown(f"""
                                            <div class="scratchpad-step">
                                                <div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860">Step {i}</div>
                                                <div class="step-thought">💭 {step["thought"]}</div>
                                                <div style="margin-top:4px;color:{action_color}">⚡ {step["action"]}({step["action_input"][:80]}{"…" if len(step["action_input"])>80 else ""})</div>
                                                {obs_html}
                                            </div>""", unsafe_allow_html=True)

                                if sources:
                                    chips = "".join(
                                        f'<span class="source-chip">[{i+1}] {s["title"][:45]}{"…" if len(s["title"])>45 else ""}<span class="source-score">{s["score"]:.3f}</span></span>'
                                        for i, s in enumerate(sources[:8])
                                    )
                                    st.markdown(f'<div style="margin-left:48px;margin-top:4px"><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#3a3860;margin-bottom:5px">SOURCES</div>{chips}</div>', unsafe_allow_html=True)

                                # Render Adversarial Peer Review scorecard
                                if isinstance(scratchpad_obj, dict) and scratchpad_obj.get("peer_review"):
                                    pr = scratchpad_obj["peer_review"]
                                    score = pr.get("overall_score", 5)
                                    conf = pr.get("confidence_score", 3)
                                    strengths = pr.get("strengths", [])
                                    weaknesses = pr.get("weaknesses", [])
                                    feedback = pr.get("constructive_feedback", "")
                                    
                                    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                                    with st.expander(f"🏅 Double-Blind Peer Review Audit (Score: {score}/10 | Conf: {conf}/5)"):
                                        st.markdown(f"**Overall Score:** `{score}/10` (NeurIPS / ICML Scale) | **Reviewer Confidence:** `{conf}/5`")
                                        st.markdown("##### **Strengths Identified:**")
                                        for str_item in strengths:
                                            st.markdown(f"- {str_item}")
                                        st.markdown("##### **Areas for Improvement / Weaknesses:**")
                                        for wk_item in weaknesses:
                                            st.markdown(f"- {wk_item}")
                                        if feedback:
                                            st.markdown("##### **Constructive Feedback & Rigor Check:**")
                                            st.markdown(feedback)

                                # Render floating micro-actions row
                                st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
                                col_act1, col_act2, col_act3, col_act4, col_act5, col_spacer = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 12.5])
                                with col_act1:
                                    if st.button("📌", key=f"pin_note_{msg['message_id']}", help="Pin message to Workspace Notes", use_container_width=True):
                                        pm.add_pinned_note(active_r["research_id"], msg["content"])
                                        st.toast("📌 Message pinned to Workspace Notes!")
                                        st.rerun()
                                with col_act2:
                                    if st.button("🔀", key=f"branch_{msg['message_id']}", help="Branch conversation from this point", use_container_width=True):
                                        new_cid = pm.branch_conversation(active_cid, msg["message_id"])
                                        if new_cid:
                                            st.session_state.active_conversation_id = new_cid
                                            st.toast("🔀 Thread branched successfully!")
                                            st.rerun()
                                with col_act3:
                                    st.download_button(
                                        label="📥",
                                        data=msg["content"],
                                        file_name=f"arxiv_agent_response_{msg['message_id'][:8]}.md",
                                        mime="text/markdown",
                                        key=f"dl_md_{msg['message_id']}",
                                        help="Download as Markdown",
                                        use_container_width=True
                                    )
                                with col_act4:
                                    if st.button("📋", key=f"copy_{msg['message_id']}", help="Copy message to active session", use_container_width=True):
                                        st.toast("📋 Content stored in active session. Drag-select to copy directly.")
                                with col_act5:
                                    if st.button("🔄", key=f"regen_point_{msg['message_id']}", help="Regenerate from this message onward", use_container_width=True):
                                        try:
                                            pm.delete_messages_from(active_cid, msg["created_at"])
                                            st.session_state["trigger_agent_run"] = True
                                            st.toast("🔄 Regenerating response...")
                                            st.rerun()
                                        except Exception as regen_err:
                                            st.error(f"Regeneration failed: {regen_err}")
                                if isinstance(scratchpad_obj, dict) and "follow_ups" in scratchpad_obj:
                                    fu_list = scratchpad_obj["follow_ups"]
                                    if fu_list:
                                        st.markdown("<div style='margin-left:4px; margin-top:8px; margin-bottom:4px;'><span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7c6fff;font-weight:600;letter-spacing:.08em;'>💡 Suggested Follow-ups:</span></div>", unsafe_allow_html=True)
                                        # Display follow-up pills as inline buttons
                                        for idx_fu, q_fu in enumerate(fu_list):
                                            if st.button(f"➔ {q_fu}", key=f"fu_{msg['message_id']}_{idx_fu}"):
                                                pm.add_message(active_cid, "user", q_fu)
                                                st.session_state["trigger_agent_run"] = True
                                                st.rerun()

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            # Active chat bottom composer — native Enter interface.
            # st.chat_input sends on Enter, supports Shift+Enter newlines, auto-clears,
            # and disables itself while the agent is running.
            if active_conv and active_conv["mode"] == "lit_review":
                # If review has already been generated, offer chat follow-ups or reset
                st.markdown("<p style='font-size:11px;color:#6b68a0;font-style:italic;'>Literature Review Generated! Toggle to Q&A Mode to converse further or delete this conversation to generate a new topic review.</p>", unsafe_allow_html=True)
            else:
                user_input = st.chat_input(
                    "Ask the agent anything about AI research…",
                    disabled=st.session_state.get("trigger_agent_run", False),
                    key="chat_main_composer",
                )
                if user_input and user_input.strip() and pm:
                    submit_query(
                        pm, active_cid, user_input,
                        bool(active_conv.get("use_corpus", 1)) if active_conv else True,
                        bool(active_conv.get("use_arxiv", 1)) if active_conv else True,
                        bool(active_conv.get("use_session", 1)) if active_conv else True,
                    )
                    st.rerun()

        # Agent processing
        if (corpus_ready and st.session_state.react_agent and active_cid and pm and
                st.session_state.get("trigger_agent_run")):
            
            st.session_state["trigger_agent_run"] = False
            messages = pm.get_messages(active_cid)
            if messages and messages[-1]["role"] == "user":
                query = messages[-1]["content"]
                agent: ReActAgent = st.session_state.react_agent
                
                # Resolve scopes
                allowed_pids = None
                if active_r and active_r["scope_type"] == "user_only":
                    allowed_pids = pm.get_research_paper_ids(active_r["research_id"])
                    if not allowed_pids:
                        allowed_pids = {"dummy_no_papers_associated"}
                        
                source_cfg = SourceConfig(
                    use_corpus=bool(active_conv.get("use_corpus", 1)) if active_conv else True,
                    use_session=bool(active_conv.get("use_session", 1)) if active_conv else True
                )
                use_arxiv = bool(active_conv.get("use_arxiv", 1)) if active_conv else True
                instr_val = active_r.get("instructions") if active_r else None

                # Phase 1: gather evidence — reasoning steps stream into the status box.
                with st.status("🤖 Agent thinking…", expanded=True) as status:
                    st.write("🔎 Evaluating scope & reasoning over papers…")

                    step_placeholder = st.empty()
                    def on_agent_step(step_num, thought, action, action_input):
                        with step_placeholder.container():
                            st.write(f"🧠 **Step {step_num + 1}:** {thought}")
                            st.markdown(f"└ ⚡ *Calling Tool:* `{action}` with input `{action_input[:80]}...`")

                    gathered = agent.gather(
                        query,
                        allowed_paper_ids=allowed_pids,
                        source_cfg=source_cfg,
                        use_arxiv=use_arxiv,
                        custom_instructions=instr_val,
                        step_callback=on_agent_step,
                    )
                    status.update(
                        label="Out of scope" if gathered.out_of_scope else "✍️ Writing answer…",
                    )

                # Phase 2: stream the final answer token-by-token into the assistant bubble.
                with st.chat_message("assistant", avatar="🤖"):
                    try:
                        st.write_stream(agent.stream_synthesis(query, gathered))
                    except Exception as stream_err:
                        import logging
                        logging.getLogger(__name__).warning("Answer streaming failed: %s", stream_err)
                        st.markdown("_Answer generation was interrupted — please retry._")

                response = agent.last_response
                if response is None:
                    # Defensive: stream_synthesis always sets last_response, but never render nothing.
                    from rag.agent.react_agent import AgentResponse
                    response = AgentResponse(
                        answer="I could not complete this response — please try again.",
                        scratchpad=gathered.steps, sources=[],
                    )
                status.update(
                    label="Out of scope" if response.out_of_scope else "Done",
                    state="complete",
                )

                # Generate follow-up questions
                follow_ups = agent.generate_follow_ups(query, response.answer, response.sources)

                scratchpad_dicts = [
                    {"thought": s.thought, "action": s.action,
                     "action_input": s.action_input, "observation": s.observation,
                     "is_final": s.is_final}
                    for s in response.scratchpad
                ]
                
                pm.add_message(
                    conversation_id=active_cid,
                    role="assistant",
                    content=response.answer,
                    scratchpad_list={
                        "scratchpad": scratchpad_dicts,
                        "sources": response.sources,
                        "sub_questions": response.sub_questions,
                        "critique_verdict": response.critique.verdict if response.critique else "pass",
                        "total_steps": response.total_steps,
                        "latency_ms": response.latency_ms,
                        "out_of_scope": response.out_of_scope,
                        "peer_review": response.critique.peer_review if (response.critique and response.critique.peer_review) else None,
                        "follow_ups": follow_ups,
                    }
                )
                
                if len(messages) == 1:
                    _auto_title_thread(active_cid, query)
                    
                # Trigger memory consolidator daemon
                try:
                    from rag.agent.memory_consolidator import MemoryConsolidator
                    mc = MemoryConsolidator(pm)
                    mc.consolidate(active_r["research_id"], query, response.answer)
                except Exception as mc_err:
                    import logging
                    logging.getLogger(__name__).warning("Memory consolidator failed: %s", mc_err)
                    
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN: THE CLAUDE-STYLE WORKSPACE SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    with col_workspace:
        mems = pm.get_workspace_memories(active_r["research_id"]) if (pm and active_r) else []
        ents = pm.get_workspace_entities(active_r["research_id"]) if (pm and active_r) else []
        linked_pids = pm.get_research_paper_ids(active_r["research_id"])

        tab_workspace, tab_details, tab_analytics, tab_tree, tab_compare = st.tabs([
            "📋 Workspace", "📄 Paper Details", "📊 Analytics", "🔀 Citation Tree", "⚖️ Compare View"
        ])
        
        with tab_workspace:
            # 1. Interactive Cognitive Memory Dashboard
            st.markdown("""
            <div style="background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-family:Syne,sans-serif; font-weight:600; font-size:13px; color:#e8e4ff;">Active Semantic Memory</span>
                    <span style="background:rgba(124,111,255,0.08); color:#7c6fff; border-radius:12px; padding:2px 8px; font-size:9px; font-weight:600; font-family:JetBrains Mono,monospace;">🔒 Active Graph</span>
                </div>
            """, unsafe_allow_html=True)
            
            if mems or ents:
                facts = [m for m in mems if m["category"] == "fact"]
                prefs = [m for m in mems if m["category"] == "preference"]
                hyps = [m for m in mems if m["category"] == "hypothesis"]
                
                if facts:
                    with st.expander(f"💡 Scientific Facts ({len(facts)})", expanded=True):
                        for f in facts[:8]:
                            col_lbl, col_del = st.columns([8, 2])
                            with col_lbl:
                                st.markdown(f"<p style='font-size:11px;color:#c4c1dc;margin:0;'>• {f['content']}</p>", unsafe_allow_html=True)
                            with col_del:
                                if st.button("✕", key=f"del_mem_{f['memory_id']}", help="Delete Fact"):
                                    pm.delete_workspace_memory(f["memory_id"])
                                    st.rerun()
                if prefs:
                    with st.expander(f"🎯 Objectives & Preferences ({len(prefs)})", expanded=False):
                        for p in prefs[:5]:
                            col_lbl, col_del = st.columns([8, 2])
                            with col_lbl:
                                st.markdown(f"<p style='font-size:11px;color:#c4c1dc;margin:0;'>• {p['content']}</p>", unsafe_allow_html=True)
                            with col_del:
                                if st.button("✕", key=f"del_mem_{p['memory_id']}", help="Delete Objective"):
                                    pm.delete_workspace_memory(p["memory_id"])
                                    st.rerun()
                if hyps:
                    with st.expander(f"❓ Hypotheses & Open Questions ({len(hyps)})", expanded=False):
                        for h in hyps[:5]:
                            col_lbl, col_del = st.columns([8, 2])
                            with col_lbl:
                                st.markdown(f"<p style='font-size:11px;color:#c4c1dc;margin:0;'>• {h['content']}</p>", unsafe_allow_html=True)
                            with col_del:
                                if st.button("✕", key=f"del_mem_{h['memory_id']}", help="Delete Hypothesis"):
                                    pm.delete_workspace_memory(h["memory_id"])
                                    st.rerun()
                if ents:
                    with st.expander(f"🔑 Key Entities ({len(ents)})", expanded=True):
                        chips_html = "".join(
                            f'<span class="source-chip" style="border-color: rgba(124, 111, 255, 0.35); color:#e8e4ff; margin:2px;" title="{e["description"]}">{e["name"]}</span>'
                            for e in ents
                        )
                        st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#8e8aa8; font-size:12px; margin:0; line-height:1.5; font-style:italic;'>Workspace cognitive memory will populate automatically as you ask questions and upload papers.</p>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

            # 2. Instructions Card
            st.markdown("""
            <div style="background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <div style="font-family:Syne,sans-serif; font-weight:600; font-size:13px; color:#e8e4ff; margin-bottom:8px;">Instructions</div>
            """, unsafe_allow_html=True)
            instr_text = active_r.get("instructions", "")
            if instr_text:
                st.markdown(f"<div style='font-size:12px;color:#a8a5c4;background:rgba(10,10,20,0.4);border-radius:6px;padding:8px 12px;margin-bottom:12px;line-height:1.5;max-height:150px;overflow-y:auto;'>{instr_text}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#6b68a0; font-size:12px; margin-bottom:12px; line-height:1.5; font-style:italic;'>Add instructions to tailor responses in this project.</p>", unsafe_allow_html=True)
                
            with st.popover("✏️ Edit Instructions", use_container_width=True):
                new_instr = st.text_area("Instructions", value=active_r.get("instructions", ""), placeholder="e.g. Focus on parameter efficiency metrics and prefix tuning details...", height=120)
                if st.button("Save Guidelines", key="save_instr_workspace_btn", use_container_width=True):
                    pm.update_research_instructions(active_r["research_id"], new_instr)
                    st.success("Guidelines updated!")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            # 3. Files/Papers Card
            st.markdown("""
            <div style="background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <div style="font-family:Syne,sans-serif; font-weight:600; font-size:13px; color:#e8e4ff; margin-bottom:8px;">Files</div>
            """, unsafe_allow_html=True)
            
            if linked_pids:
                meta_corpus = _load_metadata()
                meta_session = session_idx.list_papers() if session_idx else []
                lookup = {p["paper_id"]: p for p in meta_corpus}
                for p in meta_session:
                    lookup[p["paper_id"]] = p
                    
                for idx_f, pid in enumerate(linked_pids):
                    p_item = lookup.get(pid)
                    if p_item:
                        col_file_lbl, col_unlink = st.columns([8, 2])
                        with col_file_lbl:
                            if st.button(f"📄 {p_item['title'][:40]}...", key=f"view_paper_btn_{pid}_{idx_f}", help="View Details", use_container_width=True):
                                st.session_state["selected_paper_id"] = pid
                                st.toast("📄 Paper details loaded! Switch to 'Paper Details' tab to view.")
                                st.rerun()
                        with col_unlink:
                            if st.button("✕", key=f"unlink_{pid}_{idx_f}"):
                                pm.remove_paper_from_research(active_r["research_id"], pid)
                                st.rerun()
            else:
                st.markdown("<p style='color:#6b68a0; font-size:12px; margin-bottom:12px; line-height:1.5; font-style:italic;'>Add PDFs, documents, or other text to reference in this project.</p>", unsafe_allow_html=True)

            # Popover to add files/papers
            with st.popover("➕ Add Papers", use_container_width=True):
                opt_mode = st.radio("Add Method", ["Upload PDF", "Search arXiv", "Link from Corpus"])
                if opt_mode == "Upload PDF":
                    up_side_file = st.file_uploader("Upload PDF", type=["pdf"], key="side_uploader_widget")
                    if up_side_file:
                        u_side_title = st.text_input("Title (optional)", key="side_up_title")
                        u_side_auths = st.text_input("Authors (optional)", key="side_up_authors")
                        if st.button("Add PDF to Project", key="side_up_btn", use_container_width=True):
                            with st.spinner("Processing PDF…"):
                                chunks_up, meta_up = process_upload(up_side_file.read(), up_side_file.name, u_side_title, u_side_auths)
                                if chunks_up:
                                    session_idx.add_chunks(chunks_up, meta_up)
                                    pm.add_paper_to_research(active_r["research_id"], meta_up["paper_id"])
                                    st.success("Added successfully!")
                                    st.rerun()
                elif opt_mode == "Search arXiv":
                    ax_side_query = st.text_input("Search arXiv ID or query", placeholder="e.g. 2106.09685", key="side_ax_input")
                    if st.button("Search arXiv", key="side_ax_btn", use_container_width=True):
                        if ax_side_query.strip():
                            st.session_state["side_ax_results"] = search_arxiv(ax_side_query.strip(), max_results=3)
                    
                    if st.session_state.get("side_ax_results"):
                        for idx_ax, r in enumerate(st.session_state["side_ax_results"]):
                            with st.expander(f"📄 {r['title'][:40]}…", expanded=True):
                                st.markdown(f"**Authors:** {', '.join(r['authors'][:2])}")
                                if st.button("Add Paper", key=f"side_add_ax_{r['paper_id']}_{idx_ax}", use_container_width=True):
                                    if session_idx.has_paper(r["paper_id"]):
                                        pm.add_paper_to_research(active_r["research_id"], r["paper_id"])
                                        st.success("Linked paper!")
                                        st.rerun()
                                    else:
                                        with st.spinner("Downloading paper…"):
                                            new_ax_chunks = fetch_paper_chunks(r)
                                            if new_ax_chunks:
                                                session_idx.add_chunks(new_ax_chunks, r)
                                                pm.add_paper_to_research(active_r["research_id"], r["paper_id"])
                                                st.success("Added and Linked!")
                                                st.rerun()
                elif opt_mode == "Link from Corpus":
                    options_corpus = []
                    for p in _load_metadata():
                        options_corpus.append({"id": p["paper_id"], "label": f"📚 [CORPUS] {p['title'][:60]}..."})
                    if session_idx:
                        for p in session_idx.list_papers():
                            src_badge = p.get("source", "upload").upper()
                            options_corpus.append({"id": p["paper_id"], "label": f"📄 [{src_badge}] {p['title'][:60]}..."})
                    
                    labels_corpus = [o["label"] for o in options_corpus]
                    ids_corpus = [o["id"] for o in options_corpus]
                    
                    default_sel = [o["label"] for o in options_corpus if o["id"] in linked_pids]
                    
                    selected_labels = st.multiselect("Link/Unlink Papers", options=labels_corpus, default=default_sel, key="linker_multiselect")
                    new_pids = [options_corpus[labels_corpus.index(l)]["id"] for l in selected_labels]
                    
                    if set(new_pids) != linked_pids:
                        pm.set_research_papers(active_r["research_id"], new_pids)
                        st.success("Papers mapped!")
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            # Workspace notes card
            notes = pm.get_pinned_notes(active_r["research_id"])
            st.markdown("""
            <div style="background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <div style="font-family:Syne,sans-serif; font-weight:600; font-size:13px; color:#e8e4ff; margin-bottom:8px;">📌 Workspace Notes</div>
            """, unsafe_allow_html=True)
            if notes:
                for n in notes:
                    col_note_txt, col_note_del = st.columns([8, 2])
                    with col_note_txt:
                        st.markdown(f"<div style='font-size:11px;color:#c4c1dc;margin-bottom:8px;'>• {n['content'][:80]}...</div>", unsafe_allow_html=True)
                    with col_note_del:
                        if st.button("✕", key=f"del_note_{n['note_id']}", help="Delete Note"):
                            pm.delete_pinned_note(n["note_id"])
                            st.rerun()
            else:
                st.markdown("<p style='color:#6b68a0; font-size:12px; line-height:1.5; font-style:italic;'>Pin timeline responses to build notes here.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_details:
            selected_pid = st.session_state.get("selected_paper_id")
            if selected_pid:
                meta_corpus = _load_metadata()
                meta_session = session_idx.list_papers() if session_idx else []
                lookup = {p["paper_id"]: p for p in meta_corpus}
                for p in meta_session:
                    lookup[p["paper_id"]] = p
                    
                paper = lookup.get(selected_pid)
                if paper:
                    st.markdown(f"""
                    <div style="background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.2); border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                        <h4 style="font-family:Syne,sans-serif; color:#e8e4ff; margin-top:0; margin-bottom:8px; font-size:15px;">📄 {paper['title']}</h4>
                        <div style="font-size:11px; color:#c084fc; font-weight:600; margin-bottom:12px; font-family:monospace;">ID: {paper['paper_id']}</div>
                    """, unsafe_allow_html=True)
                    
                    auths = paper.get("authors")
                    if isinstance(auths, list):
                        authors_str = ", ".join(auths)
                    else:
                        authors_str = str(auths)
                    st.markdown(f"**Authors:** `{authors_str}`")
                    
                    published = paper.get("published", "")
                    if published:
                        st.markdown(f"**Published:** `{published}`")
                    venue = paper.get("venue", "")
                    if venue:
                        st.markdown(f"**Venue:** `{venue}`")
                        
                    pdf_url = paper.get("pdf_url") or paper.get("url") or f"https://arxiv.org/abs/{paper['paper_id']}"
                    st.markdown(f"[🔗 View Document / PDF Link ↗]({pdf_url})")
                    
                    st.divider()
                    st.markdown("##### **Abstract**")
                    st.markdown(f"<div style='font-size:12px; color:#c4c1dc; line-height:1.6; text-align:justify; max-height:260px; overflow-y:auto; padding:10px; background:rgba(0,0,0,0.25); border-radius:6px;'>{paper.get('abstract', 'No abstract available.')}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    col_close, col_unlink_act = st.columns(2)
                    with col_close:
                        if st.button("⬅️ Back to Workspace", use_container_width=True):
                            st.toast("Switch to '📋 Workspace' tab to view files.")
                    with col_unlink_act:
                        if st.button("🗑️ Unlink Paper", use_container_width=True):
                            pm.remove_paper_from_research(active_r["research_id"], selected_pid)
                            st.toast("Paper unlinked from workspace.")
                            st.rerun()
                else:
                    st.info("Paper metadata not found.")
            else:
                st.info("Select a paper in the Files list to view its details here.")

        with tab_analytics:
            st.markdown("""
            <div style="background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <h4 style="font-family:Syne,sans-serif; color:#e8e4ff; margin-top:0; margin-bottom:12px; font-size:15px;">📊 Workspace Session Analytics</h4>
            """, unsafe_allow_html=True)
            
            total_papers = len(linked_pids)
            total_msg = len(messages)
            
            facts_count = len([m for m in mems if m["category"] == "fact"])
            prefs_count = len([m for m in mems if m["category"] == "preference"])
            hyps_count = len([m for m in mems if m["category"] == "hypothesis"])
            entities_count = len(ents)
            
            st.markdown(f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:15px;">
                <div style="background:rgba(124, 111, 255, 0.05); padding:12px; border-radius:8px; border:1px solid rgba(124, 111, 255, 0.1); text-align:center;">
                    <div style="font-size:20px; font-weight:700; color:#7c6fff;">{total_papers}</div>
                    <div style="font-size:10px; color:#8e8aa8; text-transform:uppercase; margin-top:2px;">Papers Linked</div>
                </div>
                <div style="background:rgba(192, 132, 252, 0.05); padding:12px; border-radius:8px; border:1px solid rgba(192, 132, 252, 0.1); text-align:center;">
                    <div style="font-size:20px; font-weight:700; color:#c084fc;">{total_msg}</div>
                    <div style="font-size:10px; color:#8e8aa8; text-transform:uppercase; margin-top:2px;">Chat Messages</div>
                </div>
                <div style="background:rgba(124, 111, 255, 0.05); padding:12px; border-radius:8px; border:1px solid rgba(124, 111, 255, 0.1); text-align:center;">
                    <div style="font-size:20px; font-weight:700; color:#7c6fff;">{facts_count + prefs_count + hyps_count}</div>
                    <div style="font-size:10px; color:#8e8aa8; text-transform:uppercase; margin-top:2px;">Semantic Memories</div>
                </div>
                <div style="background:rgba(192, 132, 252, 0.05); padding:12px; border-radius:8px; border:1px solid rgba(192, 132, 252, 0.1); text-align:center;">
                    <div style="font-size:20px; font-weight:700; color:#c084fc;">{entities_count}</div>
                    <div style="font-size:10px; color:#8e8aa8; text-transform:uppercase; margin-top:2px;">Key Entities</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### **Workspace Activity Log**")
            st.markdown(f"- **Thread mode:** `{active_conv['mode'].upper()}`")
            st.markdown(f"- **Active search settings:** Corpus: `{bool(active_conv.get('use_corpus', 1))}`, arXiv: `{bool(active_conv.get('use_arxiv', 1))}`, Uploads: `{bool(active_conv.get('use_session', 1))}`")
            st.markdown(f"- **Scope:** `{'User documents only' if active_r.get('scope_type') == 'user_only' else 'All papers + Live search'}`")
            
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_tree:
            st.markdown("""
            <div style="background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <h4 style="font-family:Syne,sans-serif; color:#e8e4ff; margin-top:0; margin-bottom:8px; font-size:15px;">🔀 Bibliography Citation Tree</h4>
                <p style="font-size:11px; color:#8e8aa8; margin-bottom:12px;">Tracing references and timeline connection networks between papers in this workspace.</p>
            """, unsafe_allow_html=True)
            
            if len(linked_pids) < 1:
                st.info("No papers linked to this project workspace yet. Add papers to generate a citation tree.")
            else:
                meta_corpus = _load_metadata()
                meta_session = session_idx.list_papers() if session_idx else []
                lookup = {p["paper_id"]: p for p in meta_corpus}
                for p in meta_session:
                    lookup[p["paper_id"]] = p
                    
                papers = [lookup[pid] for pid in linked_pids if pid in lookup]
                
                dot_lines = [
                    "digraph G {",
                    '  background="#07070f";',
                    '  node [style="filled", fillcolor="#16162e", color="#7c6fff", fontcolor="#e8e4ff", fontname="sans-serif", shape="box", style="rounded,filled"];',
                    '  edge [color="#c084fc", arrowhead="vee", arrowsize="0.6"];'
                ]
                
                for p in papers:
                    title = p["title"]
                    clean_t = title.replace('"', '\\"').replace("\n", " ")
                    if len(clean_t) > 30:
                        clean_t = clean_t[:28] + "..."
                    year = p.get("published", "")[:4]
                    label = f"{clean_t}\\n({year})" if year else clean_t
                    dot_lines.append(f'  "{p["paper_id"]}" [label="{label}"];')
                    
                edges_count = 0
                for i in range(len(papers)):
                    for j in range(i + 1, len(papers)):
                        p1 = papers[i]
                        p2 = papers[j]
                        
                        a1 = set(p1.get("authors") or [])
                        a2 = set(p2.get("authors") or [])
                        shared = a1.intersection(a2)
                        
                        w1 = set(re.sub(r"[^\w\s]", "", p1["title"].lower()).split())
                        w2 = set(re.sub(r"[^\w\s]", "", p2["title"].lower()).split())
                        shared_words = w1.intersection(w2) - {"and", "of", "the", "a", "in", "to", "for", "with", "on", "using", "an"}
                        
                        if shared or len(shared_words) >= 2:
                            y1 = p1.get("published", "")[:4]
                            y2 = p2.get("published", "")[:4]
                            if y1 and y2 and y1 < y2:
                                dot_lines.append(f'  "{p1["paper_id"]}" -> "{p2["paper_id"]}";')
                            else:
                                dot_lines.append(f'  "{p2["paper_id"]}" -> "{p1["paper_id"]}";')
                            edges_count += 1
                            if edges_count > 10:
                                break
                    if edges_count > 10:
                        break
                        
                dot_lines.append("}")
                dot_graph = "\n".join(dot_lines)
                
                try:
                    st.graphviz_chart(dot_graph, use_container_width=True)
                except Exception as g_err:
                    st.warning("Failed to render interactive graphviz tree.")
                    
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_compare:
            st.markdown("""
            <div style="background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <h4 style="font-family:Syne,sans-serif; color:#e8e4ff; margin-top:0; margin-bottom:8px; font-size:15px;">⚖️ Workspace Compare Matrix</h4>
                <p style="font-size:11px; color:#8e8aa8; margin-bottom:12px;">Side-by-side evaluation of methods, domains, and attributes of linked documents.</p>
            """, unsafe_allow_html=True)
            
            if len(linked_pids) < 2:
                st.info("Link at least 2 papers to perform a side-by-side comparison.")
            else:
                meta_corpus = _load_metadata()
                meta_session = session_idx.list_papers() if session_idx else []
                lookup = {p["paper_id"]: p for p in meta_corpus}
                for p in meta_session:
                    lookup[p["paper_id"]] = p
                    
                papers = [lookup[pid] for pid in linked_pids if pid in lookup]
                
                import pandas as pd
                compare_data = []
                for p in papers:
                    auths = p.get("authors") or []
                    author_str = ", ".join(auths[:2]) + ("..." if len(auths)>2 else "") if isinstance(auths, list) else str(auths)
                    year = p.get("published", "")[:4]
                    
                    compare_data.append({
                        "Attribute": "Paper Title",
                        p["paper_id"]: p["title"]
                    })
                    compare_data.append({
                        "Attribute": "Publication Year",
                        p["paper_id"]: year or "N/A"
                    })
                    compare_data.append({
                        "Attribute": "Lead Authors",
                        p["paper_id"]: author_str
                    })
                    words = [w for w in re.sub(r"[^\w\s]", "", p["title"].lower()).split() if w not in {"and", "of", "the", "a", "in", "to", "for", "with", "on", "using", "an"}]
                    domain = " / ".join(words[:2]).upper()
                    compare_data.append({
                        "Attribute": "Scientific Focus",
                        p["paper_id"]: domain
                    })
                    
                df_compare = pd.DataFrame(compare_data)
                df_pivoted = df_compare.groupby("Attribute").first()
                st.dataframe(df_pivoted, use_container_width=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="max-width:600px; margin: 80px auto; padding: 40px; background: rgba(13, 13, 24, 0.45); border: 1px solid rgba(124, 111, 255, 0.16); border-radius: 20px; text-align: center; backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
        <p style="font-size: 48px; margin: 0 0 16px;">📁</p>
        <h3 style="font-family: 'Syne', sans-serif; font-weight: 700; color: #e8e4ff; margin-top: 0; margin-bottom: 8px; font-size: 24px;">Welcome to arXiv Agent</h3>
        <p style="color: #a8a5c4; font-size: 14px; margin-bottom: 24px; line-height: 1.6;">
            Create a new project workspace in the sidebar or select an active thread to start researching with structural RAG, cognitive memory, and peer review checks.
        </p>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #7c6fff; text-transform: uppercase; letter-spacing: 0.1em;">
            SYSTEM STATUS: READY TO BOOTSTRAP
        </div>
    </div>
    """, unsafe_allow_html=True)
