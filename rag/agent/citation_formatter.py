"""
citation_formatter.py – Format paper metadata into APA / MLA / Chicago / IEEE / Vancouver.

All formats handle arXiv preprints correctly (no volume/page numbers).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class PaperMeta:
    paper_id:  str
    title:     str
    authors:   list[str]
    year:      str
    url:       str  = ""
    source:    str  = "corpus"   # corpus | arxiv | upload
    doi:       str  = ""
    venue:     str  = ""
    citation_count: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "PaperMeta":
        year = d.get("published", "")[:4] or d.get("year", "n.d.")
        return cls(
            paper_id = d.get("paper_id", ""),
            title    = d.get("title", "Untitled"),
            authors  = d.get("authors", []),
            year     = year or "n.d.",
            url      = d.get("url", ""),
            source   = d.get("source", "corpus"),
            doi      = d.get("doi", ""),
            venue    = d.get("venue", ""),
            citation_count = d.get("citation_count", 0),
        )


class CitationFormatter:
    """Format a list of PaperMeta objects into a reference list."""

    def __init__(self, fmt: str = "APA") -> None:
        fmt = fmt.upper().strip()
        if fmt not in ("APA", "MLA", "CHICAGO", "IEEE", "VANCOUVER"):
            raise ValueError(f"Unknown format: {fmt}")
        self.fmt = fmt

    # ── Public API ─────────────────────────────────────────────────────────────

    def format_list(self, papers: list[PaperMeta]) -> str:
        """Return a full numbered reference list."""
        lines = []
        for i, paper in enumerate(papers, 1):
            entry = self._format_one(paper, i)
            lines.append(entry)
        return "\n\n".join(lines)

    def inline_cite(self, paper: PaperMeta, number: int) -> str:
        """Return a short inline citation string."""
        if self.fmt == "IEEE":
            return f"[{number}]"
        elif self.fmt in ("APA", "CHICAGO"):
            author_part = self._last_name(paper.authors[0]) if paper.authors else "Unknown"
            if len(paper.authors) > 1:
                author_part += " et al."
            return f"({author_part}, {paper.year})"
        elif self.fmt == "MLA":
            author_part = self._last_name(paper.authors[0]) if paper.authors else "Unknown"
            return f"({author_part})"
        elif self.fmt == "VANCOUVER":
            return f"[{number}]"
        return f"[{number}]"

    # ── Format implementations ─────────────────────────────────────────────────

    def _format_one(self, p: PaperMeta, n: int) -> str:
        if self.fmt == "APA":
            return self._apa(p, n)
        elif self.fmt == "MLA":
            return self._mla(p, n)
        elif self.fmt == "CHICAGO":
            return self._chicago(p, n)
        elif self.fmt == "IEEE":
            return self._ieee(p, n)
        elif self.fmt == "VANCOUVER":
            return self._vancouver(p, n)
        return self._apa(p, n)

    def _apa(self, p: PaperMeta, n: int) -> str:
        # Author, A. A., & Author, B. B. (Year). Title. Venue. DOI/URL
        authors = self._apa_authors(p.authors)
        venue_str = p.venue or (f"arXiv preprint arXiv:{p.paper_id}" if p.source in ("corpus", "arxiv") else "Unpublished manuscript")
        url_str = f" https://doi.org/{p.doi}" if p.doi else (f" {p.url}" if p.url else "")
        citation_str = f" (Cited {p.citation_count} times)" if p.citation_count > 0 else ""
        return f"[{n}] {authors} ({p.year}). {p.title}. *{venue_str}*.{url_str}{citation_str}"

    def _mla(self, p: PaperMeta, n: int) -> str:
        # Author Last, First. "Title." Venue, Year. DOI/URL
        authors = self._mla_authors(p.authors)
        venue_str = p.venue or "arXiv"
        url_str = f" https://doi.org/{p.doi}." if p.doi else (f" {p.url}." if p.url else "")
        return f'[{n}] {authors} "{p.title}." *{venue_str}*, {p.year}.{url_str}'

    def _chicago(self, p: PaperMeta, n: int) -> str:
        # Author Last, First. Year. "Title." Venue. DOI/URL
        authors = self._chicago_authors(p.authors)
        venue_str = p.venue or "arXiv preprint"
        url_str = f" https://doi.org/{p.doi}." if p.doi else (f" {p.url}." if p.url else "")
        return f'[{n}] {authors} {p.year}. "{p.title}." *{venue_str}*.{url_str}'

    def _ieee(self, p: PaperMeta, n: int) -> str:
        # [1] A. Author, "Title," Venue, Year, doi.
        authors = self._ieee_authors(p.authors)
        venue_str = p.venue or f"arXiv:{p.paper_id}"
        doi_str = f", doi: {p.doi}" if p.doi else ""
        return f'[{n}] {authors}, "{p.title}," *{venue_str}*, {p.year}{doi_str}.'

    def _vancouver(self, p: PaperMeta, n: int) -> str:
        # 1. Author AA. Title. Venue; Year. doi.
        authors = self._vancouver_authors(p.authors)
        venue_str = p.venue or "arXiv"
        doi_str = f" doi: {p.doi}." if p.doi else ""
        return f"{n}. {authors}. {p.title}. {venue_str}; {p.year}.{doi_str}"

    # ── Author formatting helpers ──────────────────────────────────────────────

    @staticmethod
    def _last_name(full: str) -> str:
        parts = full.strip().split()
        return parts[-1] if parts else full

    @staticmethod
    def _initials(full: str) -> str:
        parts = full.strip().split()
        if not parts:
            return ""
        last    = parts[-1]
        initials = "".join(p[0].upper() + "." for p in parts[:-1])
        return f"{last}, {initials}" if initials else last

    def _apa_authors(self, authors: list[str]) -> str:
        if not authors:
            return "Unknown Author"
        fmt  = [self._initials(a) for a in authors[:6]]
        rest = len(authors) - 6
        if rest > 0:
            return ", ".join(fmt) + f", … & {self._initials(authors[-1])}"
        if len(fmt) > 1:
            return ", ".join(fmt[:-1]) + f", & {fmt[-1]}"
        return fmt[0]

    def _mla_authors(self, authors: list[str]) -> str:
        if not authors:
            return "Unknown Author."
        first = authors[0].strip().split()
        if len(first) >= 2:
            first_fmt = f"{first[-1]}, {' '.join(first[:-1])}"
        else:
            first_fmt = first[0] if first else "Unknown"
        if len(authors) == 1:
            return first_fmt + "."
        if len(authors) == 2:
            return f"{first_fmt}, and {authors[1]}."
        return f"{first_fmt}, et al."

    def _chicago_authors(self, authors: list[str]) -> str:
        if not authors:
            return "Unknown Author."
        first = authors[0].strip().split()
        if len(first) >= 2:
            first_fmt = f"{first[-1]}, {' '.join(first[:-1])}"
        else:
            first_fmt = first[0] if first else "Unknown"
        if len(authors) == 1:
            return first_fmt + "."
        others = ", ".join(authors[1:4])
        return f"{first_fmt}, {others}." if len(authors) <= 4 else f"{first_fmt} et al."

    def _ieee_authors(self, authors: list[str]) -> str:
        if not authors:
            return "Unknown Author"
        def fmt(name):
            parts = name.strip().split()
            if len(parts) >= 2:
                initials = ". ".join(p[0].upper() for p in parts[:-1]) + "."
                return f"{initials} {parts[-1]}"
            return name
        fmted = [fmt(a) for a in authors[:3]]
        if len(authors) > 3:
            fmted.append("et al.")
        return ", ".join(fmted)

    def _vancouver_authors(self, authors: list[str]) -> str:
        if not authors:
            return "Unknown Author"
        def fmt(name):
            parts = name.strip().split()
            if len(parts) >= 2:
                initials = "".join(p[0].upper() for p in parts[:-1])
                return f"{parts[-1]} {initials}"
            return name
        fmted = [fmt(a) for a in authors[:6]]
        if len(authors) > 6:
            fmted.append("et al")
        return ", ".join(fmted)
