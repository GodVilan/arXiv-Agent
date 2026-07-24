<div align="center">

<h1>arXiv Agent v2.1</h1>

<p>
  <strong>The 100x Semantic Research Companion — A cognitive, layout-aware ML research partner with persistent SQLite workspaces, active semantic memory consolidation, multi-agent adversarial peer review, structural RAG, and recursive citation tracing.</strong>
</p>

<p>
  <a href="#vision">Vision</a> ·
  <a href="#core-upgrades">v2.1 Core Upgrades</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#features">Features</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#project-structure">Directory Map</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/BGE-large--en-0078D4?style=flat-square" alt="BGE">
  <img src="https://img.shields.io/badge/Gemini-2.5_Flash_Lite-4285F4?style=flat-square&logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-0078D4?style=flat-square" alt="FAISS">
  <img src="https://img.shields.io/badge/SQLite-Workspaces-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Streamlit-Claude_UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" alt="MIT">
</p>

</div>

---

## 🌌 Vision

**arXiv-Agent v2.1** redefines the literature search workflow from a simple search tool to an active cognitive research partner. Built for ML researchers and computer scientists, it merges a curated local corpus, private uploads, and live arXiv fetches into a unified layout-aware RAG pipeline. It maintains active semantic memory graphs across persistent SQLite workspace threads, critiques its own claims through adversarial double-blind reviewers, and recursively trails bibliographic citation networks to surface overlooked references.

arXiv-Agent v2.1 provides a completely tabless, split-pane Claude-style workspace, presenting structured conversations on the left and interactive active semantic library states on the right.

---

## 🚀 v2.1 Core Upgrades

1. **Persistent SQLite Workspace Database**: Migrated workspace configurations and threads to SQLite (`session_papers.db`), creating isolated "Researches" and persistent conversation threads complete with starring, renaming, project switching, and structural settings toggles.
2. **Active Cognitive Memory Consolidation**: Implemented a background semantic consolidator daemon powered by Gemini. Following every chat interaction or literature review generation, the daemon extracts scientific facts, user preferences, technical hypotheses, and keyword entities into SQLite, displaying them inside an interactive workspace accordion.
3. **Adversarial Double-Blind Peer Reviewer**: Upgraded self-critique modules to a top-tier ML peer reviewer, auditing every response under NeurIPS criteria (scoring 1–10 on rigor and 1–5 on confidence, detailing strengths, weaknesses, and constructive improvements).
4. **Layout-Aware targeted RAG**: Modified the PDF parser to classify text chunks into logical scientific sections (`abstract`, `introduction`, `methodology`, `experiments`, `conclusion`). The query router filters semantic search candidates targeted by section, preventing chunk blindness.
5. **Recursive Bibliographic Trailing**: Equipped the agent with a recursive bibliography tracker tool (`trace_bibliography`) that extracts citations from target papers, fetches cited papers, and maps academic graphs.
6. **Unified Glassmorphic Omnibox Card**: Consolidated the home landing state into a single, DOM-safe native container card, ensuring 100% stable Streamlit rendering without unclosed HTML tag vulnerabilities.

---

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- [Google AI Studio API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
git clone https://github.com/GodVilan/arXiv-agent
cd arXiv-agent

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Set: GEMINI_API_KEY=AIzaSy...
```

### Bootstrap Database & Vector Indices

```bash
# 1. Download pre-indexed 150 ML papers (~10 min)
python -c "from rag.data.collector import download_papers; download_papers()"

# 2. Build BGE layout-aware structural index (~15 min CPU, ~3 min GPU/MPS)
python build_index.py

# 3. Boot Streamlit Dashboard
streamlit run app.py
```

### CLI Command Guides

```bash
# Start interactive CLI terminal chat
python main.py

# Ask specific structural queries
python main.py --query "Explain QLoRA methodology details"

# Build academic literature reviews with formal citation styles
python main.py --review "Continual Learning in Language Models" --format IEEE --themes 4

# Run literature review blending local uploads + live arXiv fetching
python main.py --review "LoRA Parameter Efficiency" --format APA --live
```

---

## 📐 System Architecture

```
                       ┌──────────────────────────────────────────────────────────┐
                       │                   Streamlit Workspace UI                 │
                       │           (Unified Omnibox | Tabless Split Pane)         │
                       └─────────────┬──────────────────────────────┬─────────────┘
                                     │                              │
                                     ▼                              ▼
                       ┌────────────────────────────┐ ┌───────────────────────────┐
                       │         Left Column        │ │        Right Column       │
                       │   (Chat Timeline & Boards) │ │    (Library Workspace)    │
                       └─────────────┬──────────────┘ └─────────────┬─────────────┘
                                     │                              │
                                     ▼                              ▼
  ┌─────────────┐      ┌────────────────────────────┐ ┌───────────────────────────┐
  │ SQLite DB   │◄────►│   ProjectManager Manager   │ │  Active Semantic Memory   │
  │ Workspace,  │      │  (Star, Rename, Threading) │ │ (Accordion Deletes & Chips)│
  │ Threads,    │      └─────────────┬──────────────┘ └─────────────┬─────────────┘
  │ Memories    │                    │                              │
  └─────────────┘                    ▼                              ▼
  ┌─────────────┐      ┌────────────────────────────┐ ┌───────────────────────────┐
  │   Gemini    │◄────►│         ReAct Agent        │ │    Memory Consolidator    │
  │  Flash Lite │      │      (Multi-Hop Plan)      │ │  (Passive-to-Active Graph)│
  └─────────────┘      └─────────────┬──────────────┘ └───────────────────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │     ReAct Tool Registry    │
                       │   (Keyword/Semantic search)│
                       │   (trace_bibliography tool)│
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │        Source Router       │
                       │  (Merged Dense/Sparse RAG) │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │ Layout-Aware Vector Store  │
                       │  (Abstract, Intro, Method) │
                       └────────────────────────────┘
```

---

## 💎 Features Deep Dive

### 1. Persistent Workspaces & Dynamic Switching (SQLite)
Researchers create isolated research projects ("Researches") in the sidebar. When creating a project, the user selects the persistent **Data Scope**:
- **Option A (User papers only)**: Queries search *exclusively* within PDFs uploaded to that project.
- **Option B (All sources)**: Blends uploaded papers, the pre-indexed 150-paper corpus, and live arXiv fetches.

Within active timeline chat boxes, researchers can dynamically check/uncheck toggles (`📚 Corpus`, `🌐 arXiv`, `📄 Uploads`) to adjust retrieval scopes on-the-fly.

### 2. Layout-Aware Retrieval & Scientific Chunk Heuristics
Rather than treating PDFs as plain text, the chunker tags chunks with their logical section names (`abstract`, `introduction`, `methodology`, `experiments`, `conclusion`, `general`). When users query specific sections (e.g. *"Explain their experiments"*), `SourceRouter.search` isolates methodology and experiment chunks, avoiding information loss.

### 3. Active Cognitive Memory Dashboard
During research chats, a background memory consolidator parses responses. It extracts:
- **Scientific Facts**: Extracted key results, hardware configurations, learning rates, or baseline scores.
- **User Preferences**: Context constraints, specific citation style preferences, or excluded models.
- **Hypotheses / Open Questions**: Identified gaps, unresolved problems, or suggested future explorations.
- **Key Entities**: Custom glowing visual tags representing unique keywords, methodologies, or algorithms.
Researchers manage these items in real-time, removing incorrect insights by clicking the delete (`✕`) triggers.

### 4. Double-Blind Peer Review Audit
To ensure absolute correctness, after the agent generates a chat answer, the critique module runs an academic peer review audit. It parses the answer under a NeurIPS/ICML reviewer persona, outputting:
- **Overall Score** (1-10 scale)
- **Confidence Rating** (1-5 scale)
- **Strengths and Weaknesses Checklist**
- **Rigor Feedback**
This scorecard is available inside the collapsible `🏅 Double-Blind Peer Review Audit` panel under assistant timeline bubbles.

### 5. Multi-Source Literature Review Boards
Generates comprehensive academic literature reviews based on planned themes.
* **Exporters**: One-click download generators for **Word (.docx)** and **LaTeX (.tex)** files.
* **Granular Section Regeneration**: Click the `🔄 Regenerate Section` trigger on any tabbed theme segment. The generator fetches fresh papers based on updated instructions, rewrites the segment, and links the updated sources transparently.

---

## 📈 Benchmarks

The embedding models and chunk sizes are selected based on empirical tests across 100 benchmark QA pairs:

### Dense Retrieval Performance (Chunk Size 512)

| Model | MRR@5 | Precision@5 | Recall@10 | Latency |
|---|---|---|---|---|
| **BAAI/bge-large-en** ⭐ | **0.990** | **0.950** | **0.341** | 31.38 ms |
| Okapi BM25 | 0.978 | 0.862 | 0.293 | 11.67 ms |
| all-MiniLM-L6-v2 | 0.975 | 0.872 | 0.302 | **6.84 ms** |
| all-mpnet-base-v2 | 0.917 | 0.864 | 0.302 | 19.57 ms |

### Generation Accuracy metrics (Chunk Size 512)

| Retriever Model | Answer Relevance | Faithfulness | Context Precision |
|---|---|---|---|
| **BGE** ⭐ | **0.912** | 0.989 | **1.000** |
| all-MiniLM-L6-v2 | 0.728 | **0.997** | 0.987 |
| all-mpnet-base-v2 | 0.719 | 0.995 | 0.987 |
| Okapi BM25 | 0.125 | 0.986 | **1.000** |

---

## 📁 Project Structure

```
arXiv-agent/
├── app.py                       # Premium Streamlit UI — split-pane Claude workspace
├── main.py                      # CLI research terminal entry point
├── build_index.py               # Document embedder & layout-aware index boot
├── requirements.txt
├── .env.example
│
└── rag/
    ├── config.py                # System settings (LLM targets, vector scopes, parameters)
    │
    ├── agent/
    │   ├── react_agent.py       # ReAct orchestrator, scope checks, and loops
    │   ├── literature_agent.py  # Literature review synthesiser and regenerator
    │   ├── tools.py             # ReAct tool registry + trace_bibliography tool
    │   ├── citation_formatter.py # APA, MLA, Chicago, IEEE, Vancouver engine
    │   ├── memory.py            # Chat session sliding memory state
    │   ├── memory_consolidator.py # [NEW] SQLite-backed background cognitive memory consolidator
    │   ├── critic.py            # [UPGRADED] Double-blind academic peer reviewer scorecard
    │   ├── exporter.py          # Word (.docx) & LaTeX (.tex) generators
    │   └── planner.py           # Multi-hop query plan decomposer
    │
    ├── sources/
    │   ├── project_manager.py   # [NEW] SQLite workspace schemas, tables, and CRUD APIs
    │   ├── session_index.py     # Dynamic local FAISS index managers
    │   ├── source_router.py     # [UPGRADED] Layout-aware targeted search router
    │   ├── arxiv_fetcher.py     # Live papers downloader
    │   └── pdf_uploader.py      # PDF chunker uploader
    │
    ├── retrieval/
    │   ├── dense.py             # FAISS vectors indexes
    │   ├── bm25.py              # Okapi BM25 keyword matching
    │   ├── embeddings.py        # Embedding cache managers
    │   └── vector_store.py      # FAISS cosine wrappers
    │
    └── processing/
        └── chunker.py           # [UPGRADED] Recursive scientific section chunk tagging
```

---

## 🔮 Roadmap

- [ ] **Interactive Citation Graphs**: Visual, browser-based nodes network mapping bibliographic tracing results natively in Streamlit.
- [ ] **Semantic Scholar API Integration**: Augment local database papers with citation counts, H-indices, and journal venue metadata dynamically.
- [ ] **Multi-Corpus Workspace Switcher**: Quick-load domain-specific curated databases (e.g. BioRxiv, PubMed, NIPS) from isolated workspace drawers.
- [ ] **Automated Paper Recommendations**: Asynchronous background cron scanner recommending new preprints matching active workspace memory objectives.
- [ ] **Local LLM Offline support**: Deploy local indexer pipeline and generation clusters via Ollama/Llama.cpp.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.