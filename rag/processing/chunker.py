"""
chunker.py – PDF text extraction and recursive chunking.
"""
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF

from rag import config

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id:    str
    paper_id:    str
    title:       str
    authors:     list
    text:        str
    token_count: int
    chunk_index: int
    source:      str = "corpus"   # corpus | arxiv | upload
    section_type: str = "general"


def classify_section(text: str) -> str:
    text_lower = text.lower()[:600]
    if any(k in text_lower for k in ("abstract", "summary", "tldr", "tldr:")):
        return "abstract"
    if any(k in text_lower for k in ("introduction", "background", "motivation")):
        return "introduction"
    if any(k in text_lower for k in ("methodology", "method", "approach", "proposed architecture", "mathematical model")):
        return "methodology"
    if any(k in text_lower for k in ("evaluation", "experiment", "result", "benchmarks", "ablation", "dataset")):
        return "experiments"
    if any(k in text_lower for k in ("conclusion", "future work", "discussion", "limitations")):
        return "conclusion"
    return "general"


def extract_text(pdf_path: str | Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        log.warning("Cannot read %s: %s", pdf_path, exc)
        return ""


def extract_text_from_bytes(data: bytes) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        log.warning("Cannot read PDF bytes: %s", exc)
        return ""


def clean_text(raw: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _token_count(text: str) -> int:
    return len(text.split())


def recursive_chunk(
    text: str,
    chunk_size: int = config.DEFAULT_CHUNK,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_idx: int) -> list[str]:
        if sep_idx >= len(separators):
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
            part_tokens = _token_count(part)
            if current_tokens + part_tokens <= chunk_size:
                current.append(part)
                current_tokens += part_tokens
            else:
                if current:
                    chunks.append(sep.join(current))
                if part_tokens > chunk_size:
                    chunks.extend(_split(part, sep_idx + 1))
                    current, current_tokens = [], 0
                else:
                    overlap_words = " ".join(sep.join(current).split()[-overlap:]) if current else ""
                    current = [overlap_words, part] if overlap_words else [part]
                    current_tokens = _token_count(sep.join(current))

        if current:
            chunks.append(sep.join(current))

        return [c.strip() for c in chunks if c.strip()]

    if _token_count(text) <= chunk_size:
        return [text.strip()]

    return _split(text, sep_idx=0)


def process_papers(
    metadata: list[dict],
    chunk_size: int = config.DEFAULT_CHUNK,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for paper in metadata:
        pid      = paper["paper_id"]
        pdf_path = paper.get("pdf_path", "")
        title    = paper.get("title", pid)
        authors  = paper.get("authors", [])

        if not pdf_path or not Path(pdf_path).exists():
            log.warning("PDF not found for %s", pid)
            continue

        raw  = extract_text(pdf_path)
        text = clean_text(raw)
        if not text:
            continue

        for idx, piece in enumerate(recursive_chunk(text, chunk_size, overlap)):
            all_chunks.append(Chunk(
                chunk_id    = f"{pid}_{idx:04d}",
                paper_id    = pid,
                title       = title,
                authors     = authors,
                text        = piece,
                token_count = _token_count(piece),
                chunk_index = idx,
                source      = "corpus",
                section_type = classify_section(piece),
            ))

    log.info("Processed %d papers → %d chunks", len(metadata), len(all_chunks))
    return all_chunks


def chunks_from_bytes(
    data: bytes,
    paper_id: str,
    title: str,
    authors: list[str],
    source: str = "upload",
    chunk_size: int = config.DEFAULT_CHUNK,
) -> list[Chunk]:
    """Process a PDF from bytes (user upload or live fetch)."""
    raw  = extract_text_from_bytes(data)
    text = clean_text(raw)
    if not text:
        return []

    chunks = []
    for idx, piece in enumerate(recursive_chunk(text, chunk_size)):
        chunks.append(Chunk(
            chunk_id    = f"{paper_id}_{idx:04d}",
            paper_id    = paper_id,
            title       = title,
            authors     = authors,
            text        = piece,
            token_count = _token_count(piece),
            chunk_index = idx,
            source      = source,
            section_type = classify_section(piece),
        ))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)


def load_chunks(path: Path) -> list[Chunk]:
    with open(path) as f:
        data = json.load(f)
    return [Chunk(**d) for d in data]
