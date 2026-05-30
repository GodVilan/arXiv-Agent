"""
chunker.py – PDF text extraction and recursive chunking.

Pipeline:
    PDF file  →  raw text  →  cleaned text  →  list of Chunk objects

Usage:
    python3 chunker.py   (runs a quick smoke-test)
"""

import re
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF

from rag import config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:    str   # "{paper_id}_{chunk_index}"
    paper_id:    str
    title:       str
    text:        str
    token_count: int
    chunk_index: int


# ──────────────────────────────────────────────────────────────────────────────
# PDF → text
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract raw text from every page of a PDF using PyMuPDF.
    Returns an empty string on failure.
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        log.warning("Cannot read %s: %s", pdf_path, exc)
        return ""


def clean_text(raw: str) -> str:
    """
    Remove artefacts common in academic PDF extraction:
      - repeated whitespace / newlines
      - hyphen line-breaks
      - header/footer noise (lines that are just numbers)
    """
    text = raw
    # Re-join words broken across lines with a hyphen
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple blank lines into one paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Normalise whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

def _naive_token_count(text: str) -> int:
    """Approximate token count: split on whitespace (fast, no tokeniser needed)."""
    return len(text.split())


def recursive_chunk(
    text: str,
    chunk_size: int = config.DEFAULT_CHUNK,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping windows of approximately *chunk_size* tokens.

    Strategy (recursive):
      1. If text fits in one chunk → return it.
      2. Split on double newlines (paragraph boundary).
      3. If still too large, split on single newlines.
      4. If still too large, split on sentence boundaries.
      5. If still too large, split on whitespace (word level).
    """

    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_idx: int) -> list[str]:
        if sep_idx >= len(separators):
            # Hard fallback: character-level windows
            words = text.split()
            pieces, start = [], 0
            while start < len(words):
                pieces.append(" ".join(words[start: start + chunk_size]))
                start += chunk_size - overlap
            return pieces

        sep = separators[sep_idx]
        parts = text.split(sep)
        chunks, current, current_tokens = [], [], 0

        for part in parts:
            part_tokens = _naive_token_count(part)

            if current_tokens + part_tokens <= chunk_size:
                current.append(part)
                current_tokens += part_tokens
            else:
                if current:
                    chunks.append(sep.join(current))

                if part_tokens > chunk_size:
                    # Recurse into finer separators
                    chunks.extend(_split(part, sep_idx + 1))
                    current, current_tokens = [], 0
                else:
                    # Start a new chunk with overlap from previous
                    overlap_words = " ".join(sep.join(current).split()[-overlap:]) if current else ""
                    current = [overlap_words, part] if overlap_words else [part]
                    current_tokens = _naive_token_count(sep.join(current))

        if current:
            chunks.append(sep.join(current))

        return [c.strip() for c in chunks if c.strip()]

    if _naive_token_count(text) <= chunk_size:
        return [text.strip()]

    return _split(text, sep_idx=0)


# ──────────────────────────────────────────────────────────────────────────────
# High-level processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_papers(
    metadata: list[dict],
    chunk_size: int = config.DEFAULT_CHUNK,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    For every paper in *metadata*:
      1. Extract + clean text from its PDF.
      2. Chunk the text.
      3. Return a flat list of Chunk objects.
    """
    all_chunks: list[Chunk] = []

    for paper in metadata:
        paper_id = paper["paper_id"]
        pdf_path = paper.get("pdf_path", "")
        title    = paper.get("title", paper_id)

        if not pdf_path or not Path(pdf_path).exists():
            log.warning("PDF not found for %s, skipping.", paper_id)
            continue

        raw  = extract_text_from_pdf(pdf_path)
        text = clean_text(raw)

        if not text:
            log.warning("Empty text for %s, skipping.", paper_id)
            continue

        pieces = recursive_chunk(text, chunk_size=chunk_size, overlap=overlap)

        for idx, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id    = f"{paper_id}_{idx:04d}",
                    paper_id    = paper_id,
                    title       = title,
                    text        = piece,
                    token_count = _naive_token_count(piece),
                    chunk_index = idx,
                )
            )

    log.info("Processed %d papers → %d chunks (chunk_size=%d)",
             len(metadata), len(all_chunks), chunk_size)
    return all_chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)
    log.info("Saved %d chunks to %s", len(chunks), path)


def load_chunks(path: Path) -> list[Chunk]:
    with open(path) as f:
        data = json.load(f)
    return [Chunk(**d) for d in data]


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, logging
    logging.basicConfig(level=logging.INFO)

    meta_path = config.DATA_DIR / "metadata.json"
    if not meta_path.exists():
        print("Run data_collection.py first.")
    else:
        with open(meta_path) as f:
            meta = json.load(f)

        chunks = process_papers(meta[:5])   # quick test on first 5 papers
        print(f"\nGenerated {len(chunks)} chunks from 5 papers.")
        print("Sample chunk:\n", chunks[0].text[:300], "…")
