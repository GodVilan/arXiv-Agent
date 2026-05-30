# arXiv Agent

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![BGE](https://img.shields.io/badge/BGE-large--en-0078D4?style=flat-square)](https://huggingface.co/BAAI/bge-large-en)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**A ReAct agent that reasons step-by-step over 150 arXiv ML papers.**  
**Multi-hop planning · Self-critique · Conversation memory · Live arXiv fetch**

</div>

---

## What makes this an agent (not just RAG)

| Feature | RAG pipeline | arXiv Agent |
|---|---|---|
| Retrieval | Single fixed query | Iterative — agent decides what to search |
| Query handling | One-shot | Multi-hop decomposition for complex questions |
| Answer quality | Static | Self-critique loop refines weak answers |
| Memory | None | Conversation window + persistent research notes |
| Tools | Search only | Search · BM25 · Live arXiv fetch · Summarize · Compare |
| Transparency | Black box | Full scratchpad shown in UI |

---

## Architecture

```
User query
    │
    ▼
QueryPlanner
    │  Is this complex? Decompose into sub-questions.
    │
    ▼
ReAct Loop  (per sub-question)
    │
    ├─ Thought: what tool do I need?
    ├─ Action:  call tool
    │           ├── search_corpus   (BGE/FAISS)
    │           ├── keyword_search  (BM25 fallback)
    │           ├── fetch_arxiv     (live arXiv API)
    │           ├── summarize_paper
    │           └── compare_papers
    ├─ Observation: tool output
    └─ Repeat until finish
    │
    ▼
Synthesiser  (for multi-hop: merge sub-answers)
    │
    ▼
AnswerCritic
    │  grounded? complete? specific?
    │  If not → corrective search → re-synthesise
    │
    ▼
Final answer + scratchpad + sources
```

---

## Quick Start

```bash
git clone https://github.com/you/arXiv-agent
cd arXiv-agent

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add: GEMINI_API_KEY=AIza...

# 1. Download 150 papers (~10 min)
python -c "from rag.data.collector import download_papers; download_papers()"

# 2. Launch the UI
streamlit run app.py

# 3. Or use the CLI
python main.py
python main.py --query "Compare LoRA and prefix tuning"
python main.py --verbose   # show full scratchpad
```

---

## Agent in action

```
❓ Compare how LoRA and prefix tuning handle parameter efficiency

🗺 Planner: complex → 3 sub-questions
   • What is LoRA and how does it achieve parameter efficiency?
   • What is prefix tuning and how does it work?
   • How do LoRA and prefix tuning compare empirically?

Step 1  💭 I should search for LoRA papers first
        ⚡ search_corpus(LoRA low-rank adaptation parameter efficient fine-tuning)
        📋 [Source 1] LoRA: Low-Rank Adaptation ... (score: 0.941)

Step 2  💭 Now search for prefix tuning papers
        ⚡ search_corpus(prefix tuning parameter efficient NLP)
        📋 [Source 1] Prefix-Tuning: Optimizing ... (score: 0.923)

Step 3  💭 I have enough to compare — answering now
        ⚡ finish(Based on [Source: LoRA], ... [Source: Prefix-Tuning], ...)

🔍 Self-critique: ✅ passed
💬 Answer: [well-cited comparison of both approaches]
```

---

## Project Structure

```
arXiv-agent/
├── app.py                     # Streamlit UI with scratchpad display
├── main.py                    # CLI entry point
├── requirements.txt
│
└── rag/
    ├── config.py              # BGE-only config + agent settings
    ├── agent/
    │   ├── react_agent.py     # ReAct loop orchestrator
    │   ├── tools.py           # Tool registry (5 tools)
    │   ├── memory.py          # Conversation + research memory
    │   ├── planner.py         # Multi-hop query decomposition
    │   └── critic.py          # Self-critique and reflection
    ├── retrieval/
    │   ├── dense.py           # BGE/FAISS retriever
    │   ├── bm25.py            # BM25 keyword fallback
    │   ├── embeddings.py      # SentenceTransformer wrapper
    │   └── vector_store.py    # FAISS index
    ├── generation/
    │   └── generator.py       # Gemini backend
    ├── processing/
    │   └── chunker.py         # PDF extraction + chunker
    └── data/
        └── collector.py       # arXiv downloader
```

---

## Design decisions

**BGE only** — benchmarked MiniLM / MPNet / BGE on 100-question QA set.
BGE won on every metric that matters for an agent: MRR@5=0.990, Answer Relevance=0.912,
Context Precision=1.000. The model switcher is gone from the UI — that's a benchmark
artefact, not a user feature.

**BM25 as silent fallback** — kept as `keyword_search` tool. The agent picks it
automatically for queries with specific method names or paper titles where sparse
retrieval wins.

**Critic before answer** — one extra Gemini call catches incomplete or ungrounded
answers before they reach the user, improving faithfulness without changing the
retriever.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | BGE-large-en (1024-dim, BAAI) |
| Vector index | FAISS IndexFlatIP |
| Sparse retrieval | rank-bm25 (Okapi BM25) |
| Live paper fetch | arxiv Python library |
| Agent backbone | Google Gemini 2.5 Flash Lite |
| UI | Streamlit |
| Acceleration | Apple MPS · CUDA · CPU |

---

## License

MIT
