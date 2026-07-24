"""
run_evaluation.py – RAG Evaluation and Benchmarking Framework.

Uses Gemini as an LLM-as-a-judge to evaluate:
  1. Answer Relevance
  2. Groundedness
  3. Completeness
"""
import json
import logging
import re
import sys
import time
from pathlib import Path

# Add root folder to python path
sys.path.append(str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from rag import config
from rag.processing.chunker import load_chunks
from rag.retrieval.dense import Retriever
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.embeddings import EmbeddingModel
from rag.sources.session_index import SessionIndex
from rag.sources.source_router import SourceRouter
from rag.agent import ReActAgent, ConversationMemory, ResearchMemory

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)

_JUDGE_SYSTEM = """\
You are an expert RAG system evaluator. Analyze the user's question, the retrieved context, and the proposed answer.
Score the proposed answer on three metrics from 1 to 5 (1 = worst, 5 = best):

1. Relevance: Does the answer directly address the user's question?
2. Groundedness: Are all claims in the answer supported by the retrieved context? Deduct points if it brings in outside knowledge or makes unverified claims.
3. Completeness: Does the answer address all parts of the question, including side comparisons or details?

Provide the output strictly in the following JSON format:
{
  "relevance_score": 5,
  "relevance_reason": "reasoning...",
  "groundedness_score": 4,
  "groundedness_reason": "reasoning...",
  "completeness_score": 5,
  "completeness_reason": "reasoning..."
}
"""

def evaluate_rag(client, question: str, answer: str, context: str) -> dict:
    prompt = (
        f"Question: {question}\n\n"
        f"Proposed Answer:\n{answer}\n\n"
        f"Retrieved Context:\n{context[:10000]}"
    )
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_JUDGE_SYSTEM,
                temperature=0.0,
                max_output_tokens=500,
            ),
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
        return json.loads(raw)
    except Exception as e:
        print(f"Error calling judge: {e}")
        return {
            "relevance_score": 0, "relevance_reason": f"Error: {e}",
            "groundedness_score": 0, "groundedness_reason": "",
            "completeness_score": 0, "completeness_reason": ""
        }

def main():
    print("\n" + "═" * 70)
    print("  arXiv Agent — Automated Evaluation Framework")
    print("═" * 70)

    # ── Check metadata ─────────────────────────────────────────────
    cache = config.DATA_DIR / f"chunks_{config.DEFAULT_CHUNK}.json"
    if not cache.exists():
        print("❌ Error: prebuilt corpus chunks not found. Run build_index.py first.")
        sys.exit(1)

    print("Loading corpus chunks…")
    chunks = load_chunks(cache)
    print(f"Loaded {len(chunks):,} chunks.")

    print("Initializing active RAG system components…")
    dense = Retriever.build(
        chunks=chunks,
        chunk_size=config.DEFAULT_CHUNK,
        index_dir=config.RESULTS_DIR / "indices",
    )
    bm25 = BM25Retriever(chunks)
    emb_model = EmbeddingModel()
    session_idx = SessionIndex(emb_model)
    router = SourceRouter(dense, session_idx)
    
    # Initialize memories
    conv_mem = ConversationMemory()
    res_mem = ResearchMemory()
    agent = ReActAgent(router, bm25, session_idx, conv_mem, res_mem)

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # Representative evaluation questions
    questions = [
        "Compare LoRA and prefix tuning on parameter efficiency.",
        "What approaches are used to tackle catastrophic forgetting in continual learning?",
        "Who was the first president of the United States?" # Off-topic check
    ]

    results = []
    print("\nRunning RAG agent evaluation benchmark…\n")

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Question: '{q}'")
        t0 = time.time()
        
        # Run agent
        response = agent.run(q)
        latency = round((time.time() - t0) * 1000, 1)

        print(f"   -> Agent finished in {latency}ms (Steps: {response.total_steps})")
        
        if response.out_of_scope:
            print("   -> Guarded correctly: Out-of-Scope!")
            results.append({
                "question": q,
                "latency": latency,
                "steps": response.total_steps,
                "out_of_scope": True,
                "scores": {
                    "relevance_score": 5, "relevance_reason": "Successfully filtered as out-of-scope.",
                    "groundedness_score": 5, "groundedness_reason": "N/A",
                    "completeness_score": 5, "completeness_reason": "N/A"
                }
            })
            continue

        # Format context for judge
        flat_context = "\n\n".join([step.observation for step in response.scratchpad if step.observation])
        
        print("   -> Evaluating answer quality via LLM-as-a-judge…")
        scores = evaluate_rag(client, q, response.answer, flat_context)
        
        results.append({
            "question": q,
            "latency": latency,
            "steps": response.total_steps,
            "out_of_scope": False,
            "scores": scores
        })
        print(f"      Relevance Score    : {scores.get('relevance_score', 0)}/5")
        print(f"      Groundedness Score : {scores.get('groundedness_score', 0)}/5")
        print(f"      Completeness Score : {scores.get('completeness_score', 0)}/5")
        print()

    # ── Final Scoreboard ─────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  EVALUATION BENCHMARK SCOREBOARD")
    print("═" * 70)
    print(f"{'Question':<55} | {'Steps':<5} | {'Rel':<3} | {'Grd':<3} | {'Cmp':<3} | {'Time(ms)':<8}")
    print("-" * 90)
    
    total_rel, total_grd, total_cmp, valid_count = 0, 0, 0, 0
    for res in results:
        scores = res["scores"]
        q_trunc = res["question"][:52] + "..." if len(res["question"]) > 52 else res["question"]
        print(f"{q_trunc:<55} | {res['steps']:<5d} | {scores.get('relevance_score', 0):<3d} | {scores.get('groundedness_score', 0):<3d} | {scores.get('completeness_score', 0):<3d} | {res['latency']:<8.0f}")
        
        if not res["out_of_scope"]:
            total_rel += scores.get('relevance_score', 0)
            total_grd += scores.get('groundedness_score', 0)
            total_cmp += scores.get('completeness_score', 0)
            valid_count += 1

    print("-" * 90)
    if valid_count > 0:
        avg_rel = total_rel / valid_count
        avg_grd = total_grd / valid_count
        avg_cmp = total_cmp / valid_count
        print(f"{'AVERAGES (ML Queries)':<55} | {'-':<5} | {avg_rel:<3.1f} | {avg_grd:<3.1f} | {avg_cmp:<3.1f} |")
    print("═" * 70 + "\n")

if __name__ == "__main__":
    main()
