"""
collector.py – Download arXiv papers and save PDFs + metadata.

Fixes:
  - download_pdf() removed in arxiv>=2.x → use urllib directly from pdf_url
  - 429 rate limiting → smaller page size, longer delays, resume-safe
"""
import json
import logging
import re
import time
import urllib.request
from pathlib import Path

import arxiv

from rag import config

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def _extract_id(entry_id: str) -> str:
    """Extract bare paper ID from entry_id URL, e.g. 'http://arxiv.org/abs/2106.09685v1' → '2106.09685'"""
    m = _ID_RE.search(entry_id)
    return m.group(1) if m else entry_id.split("/")[-1].split("v")[0]


def _download_pdf(pdf_url: str, dest: Path, retries: int = 3) -> bool:
    """Download a PDF via urllib (works with all arxiv library versions)."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                pdf_url,
                headers={"User-Agent": "arXiv-Agent/1.0 (academic research tool)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as exc:
            log.warning("PDF download attempt %d/%d failed for %s: %s", attempt, retries, pdf_url, exc)
            if attempt < retries:
                time.sleep(5 * attempt)
    return False


def download_papers(
    category: str = config.ARXIV_CATEGORY,
    n: int = config.NUM_PAPERS,
) -> list[dict]:
    """
    Download up to n papers from arXiv for the given category.
    Resume-safe: skips papers whose PDFs already exist.
    Returns list of metadata dicts.
    """
    meta_path = config.DATA_DIR / "metadata.json"

    # Load existing metadata
    existing: dict[str, dict] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for p in json.load(f):
                existing[p["paper_id"]] = p
        log.info("Found %d existing papers.", len(existing))
        print(f"  Resuming — {len(existing)} papers already downloaded.")

    if len(existing) >= n:
        print(f"  Already have {len(existing)} papers (target: {n}). Nothing to do.")
        return list(existing.values())

    needed = n - len(existing)
    print(f"  Need to download {needed} more papers…")

    # Use smaller page size to avoid 429
    client = arxiv.Client(
        page_size=50,
        delay_seconds=5.0,   # polite — arXiv asks for 3s minimum
        num_retries=5,
    )
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=n * 2,   # fetch extra to account for skips
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    new_count = 0
    skipped   = 0

    try:
        for paper in client.results(search):
            if new_count >= needed:
                break

            pid = _extract_id(paper.entry_id)

            if pid in existing:
                skipped += 1
                continue

            pdf_path = config.DATA_DIR / f"{pid.replace('/', '_')}v1.pdf"

            # Download PDF using urllib (compatible with all arxiv lib versions)
            pdf_url = getattr(paper, "pdf_url", None) or f"https://arxiv.org/pdf/{pid}v1"
            success = _download_pdf(pdf_url, pdf_path)

            if not success:
                log.warning("Skipping %s — PDF download failed.", pid)
                continue

            authors = []
            for a in paper.authors:
                name = getattr(a, "name", str(a))
                authors.append(name)

            existing[pid] = {
                "paper_id":  pid,
                "title":     paper.title,
                "authors":   authors,
                "abstract":  paper.summary.replace("\n", " "),
                "published": paper.published.isoformat(),
                "pdf_path":  str(pdf_path),
                "url":       paper.entry_id,
            }
            new_count += 1

            # Save after every paper (resume-safe)
            _save_metadata(existing, meta_path)

            print(f"  [{new_count}/{needed}] {paper.title[:65]}…")

            # Polite delay between downloads
            time.sleep(3.0)

    except arxiv.HTTPError as exc:
        log.error("arXiv API error: %s", exc)
        print(f"\n  ⚠ arXiv rate-limited ({exc}). Saved {new_count} papers so far.")
        print("  Wait a few minutes then re-run — it will resume where it left off.")

    papers = list(existing.values())
    _save_metadata(existing, meta_path)
    print(f"\n  ✅ Done. {len(papers)} total papers saved to {meta_path}")
    return papers


def _save_metadata(existing: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(list(existing.values()), f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    papers = download_papers()
    print(f"Total: {len(papers)} papers.")