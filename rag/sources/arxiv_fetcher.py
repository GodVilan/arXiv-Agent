"""
arxiv_fetcher.py – Fetch full arXiv papers (PDF + metadata) into the session index.
"""
import logging
import re
import tempfile
import time
import urllib.request
from pathlib import Path

import arxiv

from rag import config
from rag.processing.chunker import chunks_from_bytes

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def _normalise_id(query: str) -> str | None:
    m = _ID_RE.search(query)
    return m.group(1) if m else None


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv and return metadata dicts (no PDF download).
    Used for showing search results before the user decides to fetch.
    """
    client = arxiv.Client(num_retries=2, delay_seconds=1.0)
    arxiv_id = _normalise_id(query)

    if arxiv_id:
        search = arxiv.Search(id_list=[arxiv_id])
    else:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

    results = []
    try:
        for paper in client.results(search):
            pid = paper.get_short_id().split("v")[0]
            results.append({
                "paper_id":  pid,
                "title":     paper.title,
                "authors":   [str(a) for a in paper.authors],
                "abstract":  paper.summary.replace("\n", " ")[:500],
                "published": paper.published.strftime("%Y-%m-%d"),
                "url":       paper.entry_id,
                "pdf_url":   paper.pdf_url,
                "source":    "arxiv",
            })
            time.sleep(0.3)
    except Exception as exc:
        log.error("arXiv search error: %s", exc)

    return results


def fetch_paper_chunks(meta: dict, chunk_size: int = config.DEFAULT_CHUNK) -> list:
    """
    Download the PDF for a paper and return Chunk objects.
    meta must contain 'pdf_url', 'paper_id', 'title', 'authors'.
    """
    pdf_url = meta.get("pdf_url", "")
    if not pdf_url:
        log.warning("No PDF URL for %s", meta.get("paper_id"))
        return []

    try:
        log.info("Downloading PDF: %s", pdf_url)
        req = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "arXiv-Agent/1.0 (research tool)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            pdf_bytes = resp.read()

        chunks = chunks_from_bytes(
            data      = pdf_bytes,
            paper_id  = meta["paper_id"],
            title     = meta["title"],
            authors   = meta.get("authors", []),
            source    = "arxiv",
            chunk_size= chunk_size,
        )
        log.info("Fetched %d chunks for '%s'", len(chunks), meta["title"][:60])
        return chunks

    except Exception as exc:
        log.error("PDF download failed for %s: %s", meta.get("paper_id"), exc)
        return []


def enrich_with_semantic_scholar(paper_id: str) -> dict:
    """
    Enriches arXiv paper metadata by querying the free Semantic Scholar API.
    Returns a dict with extra metadata fields: venue, doi, citation_count.
    """
    import urllib.request
    import json
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}?fields=venue,year,externalIds,citationCount"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "arXiv-Agent/1.0 (research tool)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "venue": data.get("venue", "") or "arXiv",
                "doi": data.get("externalIds", {}).get("DOI", ""),
                "citation_count": data.get("citationCount", 0),
                "year": str(data.get("year", "")) if data.get("year") else None
            }
    except Exception as e:
        log.debug("Semantic Scholar enrichment failed for %s: %s", paper_id, e)
    return {"venue": "arXiv", "doi": "", "citation_count": 0, "year": None}
