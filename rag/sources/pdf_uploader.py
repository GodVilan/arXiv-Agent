"""
pdf_uploader.py – Process user-uploaded PDFs into the session index.
"""
import hashlib
import logging
import re

from rag import config
from rag.processing.chunker import chunks_from_bytes

log = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _infer_year(text: str) -> str:
    m = _YEAR_RE.search(text[:2000])
    return m.group() if m else "Unknown"


def process_upload(
    file_bytes: bytes,
    filename: str,
    user_title: str = "",
    user_authors: str = "",
    chunk_size: int = config.DEFAULT_CHUNK,
) -> tuple[list, dict]:
    """
    Process a user-uploaded PDF.

    Returns:
        (chunks, metadata_dict)
    """
    # Stable ID from content hash
    paper_id = "upload_" + hashlib.md5(file_bytes[:4096]).hexdigest()[:10]

    title_str = user_title or ""
    authors_str = user_authors or ""
    title   = title_str.strip() or filename.replace(".pdf", "").replace("_", " ").strip()
    authors = [a.strip() for a in authors_str.split(",") if a.strip()]

    chunks = chunks_from_bytes(
        data       = file_bytes,
        paper_id   = paper_id,
        title      = title,
        authors    = authors,
        source     = "upload",
        chunk_size = chunk_size,
    )

    # Try to infer year from first chunks
    full_text = " ".join(c.text for c in chunks[:3])
    year = _infer_year(full_text)

    metadata = {
        "paper_id":  paper_id,
        "title":     title,
        "authors":   authors,
        "published": year,
        "source":    "upload",
        "filename":  filename,
        "url":       "",
    }

    log.info(
        "Processed upload '%s' → %d chunks (paper_id=%s)",
        filename, len(chunks), paper_id,
    )
    return chunks, metadata
