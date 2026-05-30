"""
collector.py – Download ML papers from arXiv using the arXiv API.

Usage:
    python3 collector.py
"""

import time
import json
import logging
from pathlib import Path

import arxiv

from rag import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def download_papers(
    category: str = config.ARXIV_CATEGORY,
    num_papers: int = config.NUM_PAPERS,
    output_dir: Path = config.DATA_DIR,
) -> list[dict]:
    """
    Query arXiv for recent cs.LG papers, download their PDFs,
    and save metadata as a JSON file.

    Returns a list of metadata dicts for successfully downloaded papers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    # ── Resume from existing metadata ────────────────────────────────────────
    existing_meta: list[dict] = []
    existing_ids: set[str] = set()
    if metadata_path.exists():
        with open(metadata_path) as f:
            existing_meta = json.load(f)
        existing_ids = {p["paper_id"] for p in existing_meta}
        log.info("Found %d already-downloaded papers.", len(existing_meta))
        if len(existing_meta) >= num_papers:
            log.info("Target already reached – skipping download.")
            return existing_meta

    # ── Build arXiv query ────────────────────────────────────────────────────
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=num_papers * 2,          # request extra to account for failures
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    collected: list[dict] = list(existing_meta)
    need = num_papers - len(existing_meta)

    log.info("Querying arXiv for %s papers (need %d more)…", category, need)

    for result in client.results(search):
        if len(collected) >= num_papers:
            break

        paper_id = result.entry_id.split("/")[-1]
        if paper_id in existing_ids:
            continue

        pdf_path = output_dir / f"{paper_id}.pdf"

        # Skip if PDF already present on disk but not in metadata
        if not pdf_path.exists():
            try:
                log.info("[%d/%d] Downloading %s – %s",
                         len(collected) + 1, num_papers, paper_id, result.title[:60])
                result.download_pdf(dirpath=str(output_dir), filename=f"{paper_id}.pdf")
                time.sleep(1)          # be polite to arXiv
            except Exception as exc:
                log.warning("  Failed to download %s: %s", paper_id, exc)
                continue

        meta = {
            "paper_id":   paper_id,
            "title":      result.title,
            "authors":    [a.name for a in result.authors],
            "abstract":   result.summary,
            "published":  str(result.published),
            "categories": result.categories,
            "pdf_path":   str(pdf_path),
        }
        collected.append(meta)
        existing_ids.add(paper_id)

        # Save incrementally so progress is not lost on interruption
        with open(metadata_path, "w") as f:
            json.dump(collected, f, indent=2)

    log.info("Download complete: %d papers saved to %s", len(collected), output_dir)
    return collected


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    papers = download_papers()
    print(f"\n✅  Downloaded {len(papers)} papers to '{config.DATA_DIR}'")
