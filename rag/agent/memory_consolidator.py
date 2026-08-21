"""
memory_consolidator.py – Background insight extraction daemon for arXiv-Agent v2.
"""
import json
import logging
import re
from rag.llm import get_client
from google.genai import types
from rag import config
from rag.sources.project_manager import ProjectManager

log = logging.getLogger(__name__)

_CONSOLIDATE_SYSTEM = """\
You are a cognitive memory extraction daemon for an AI ML research workspace.
Your task is to analyze the latest chat turn between the user and the research agent and extract:
1. Scientific Facts / Insights (proven parameters, mathematical properties, benchmarks, methods discovered).
2. User Preferences / Objectives (e.g. user wants to focus on Parameter-Efficient Fine-Tuning, user prefers lightweight models).
3. Open Research Hypotheses / Questions (open problems to solve, future explorations).
4. Major Technical Entities / Keywords (e.g. "LoRA", "NF4 Quantization") with brief descriptions.

Format the output strictly as a JSON object inside a markdown code block:
{
  "memories": [
    {
      "category": "fact" | "preference" | "hypothesis",
      "content": "clear concise scientific statement",
      "importance": 0.1 to 1.0
    }
  ],
  "entities": [
    {
      "name": "Entity Name",
      "description": "brief technical definition"
    }
  ]
}

Guidelines:
- Extract ONLY clear scientific facts and actual technical entities. Avoid fluff.
- Keep content statements extremely dense and informative.
- If no new facts, preferences, or entities exist, return empty arrays.
"""


class MemoryConsolidator:
    def __init__(self, pm: ProjectManager) -> None:
        self.pm = pm
        self.client = get_client()

    def consolidate(self, research_id: str, last_query: str, last_response: str) -> None:
        """Runs in background or main timeline post-run to extract cognitive insights."""
        try:
            # 1. Prepare chat turn context
            turn_str = f"User Question: {last_query}\nAgent Response: {last_response[:4000]}"
            
            # 2. Call Gemini
            resp = self.client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=turn_str,
                config=types.GenerateContentConfig(
                    system_instruction=_CONSOLIDATE_SYSTEM,
                    temperature=0.2,
                    max_output_tokens=800,
                ),
            )
            raw = resp.text.strip()
            # Clean JSON markdown if any
            raw_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
            parsed = json.loads(raw_clean)
            
            # 3. Save memories
            existing_mems = self.pm.get_workspace_memories(research_id)
            existing_texts = {m["content"].lower().strip() for m in existing_mems}
            
            for mem in parsed.get("memories", []):
                cat = mem.get("category")
                content = mem.get("content", "").strip()
                importance = float(mem.get("importance", 0.5))
                
                if cat in ("fact", "preference", "hypothesis") and content:
                    # Deduplicate simple overlaps
                    if content.lower().strip() not in existing_texts:
                        self.pm.add_workspace_memory(research_id, cat, content, importance)
                        
            # 4. Sync entities
            entities = parsed.get("entities", [])
            if entities:
                existing_ents = self.pm.get_workspace_entities(research_id)
                ent_lookup = {e["name"].lower().strip(): e for e in existing_ents}
                
                # Merge entities list to avoid losing previous ones
                new_ent_list = []
                for ent in entities:
                    name = ent.get("name", "").strip()
                    desc = ent.get("description", "").strip()
                    if name:
                        ent_lookup[name.lower().strip()] = {"name": name, "description": desc}
                        
                sync_payload = [{"name": e["name"], "description": e["description"]} for e in ent_lookup.values()]
                self.pm.sync_extracted_entities(research_id, sync_payload[:15]) # cap at 15 key entities
                
            log.info("Cognitive memory consolidation complete for workspace %s", research_id)
        except Exception as e:
            log.error("Failed in memory consolidation: %s", e)
