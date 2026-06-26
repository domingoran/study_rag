"""
PDF → ParsedDocument  using Docling.

Extracts:
  • document-level metadata (title, year, paper_id, authors)
  • ordered list of RawElements (text, section headers, tables, figures, formulas)

Author extraction (Phase 2)
---------------------------
After locating the TITLE item we scan the TEXT items on the same or next
page until we hit a section header (e.g. "Abstract").  A block is treated
as author-like when it:
  • contains no sentence-ending punctuation  (no full stops / colons)
  • contains at least one comma  OR  is a short line (≤ 80 chars) with
    multiple capitalised words
  • does not look like an affiliation / university line

This is a best-effort heuristic — for reliably-formatted arXiv PDFs it works
well; for scanned or non-standard layouts it may miss or over-include text.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter
from docling_core.types.doc import (
    DocItemLabel,
    PictureItem,
    TableItem,
    TextItem,
)

from core.schemas import ParsedDocument, RawElement

logger = logging.getLogger(__name__)

# Docling loads ML models (layout detection, table structure) at construction
# time. One converter is shared across all parse_pdf() calls in a process.
_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        logger.info("Loading Docling DocumentConverter …")
        _converter = DocumentConverter()
    return _converter


# -------------------------------------------------------------------------
# Author-extraction helpers
# -------------------------------------------------------------------------

# Keywords that strongly suggest an affiliation / institution line
_AFFILIATION_SIGNALS = re.compile(
    r'\b(university|institute|department|dept|lab|laboratory|school|'
    r'college|center|centre|research|inc\.|ltd\.|corporation|email|'
    r'@|\{|\})\b',
    re.IGNORECASE,
)

# Words that look like proper names (start with capital, ≥ 2 chars)
_CAPITALISED_WORD = re.compile(r'\b[A-Z][a-z]{1,}\b')


def _looks_like_authors(text: str) -> bool:
    """
    Heuristically decide whether *text* is an author-name block.

    Returns True when the text is short, contains mostly capitalised words,
    has at least one comma (or very few words), and does not look like
    an affiliation or URL.
    """
    text = text.strip()
    if not text or len(text) > 300:
        return False
    # Skip if it looks like an affiliation / institution
    if _AFFILIATION_SIGNALS.search(text):
        return False
    # Skip if it ends with a sentence-ending punctuation (likely a sentence)
    if re.search(r'[.!?;]$', text):
        return False
    # Must have at least two capitalised words (names)
    cap_words = _CAPITALISED_WORD.findall(text)
    if len(cap_words) < 2:
        return False
    # Positive signal: contains a comma (multiple names separated)
    # or is short enough to plausibly be one or two names
    if ',' in text or len(text.split()) <= 4:
        return True
    return False


def _extract_authors(elements: List[RawElement], title_page: int) -> List[str]:
    """
    Scan elements on the title page (and the next) to find author names.

    Args:
        elements:   All RawElements from the parsed document.
        title_page: Page number where the title was found.

    Returns:
        List of individual author name strings, or [] if nothing is found.
    """
    candidates: List[str] = []
    found_title = False

    for el in elements:
        # Only look on pages 1 and 2 relative to the title
        if el.page > title_page + 1:
            break

        # Start collecting after the title element
        if el.label == DocItemLabel.TITLE.value:
            found_title = True
            continue

        if not found_title:
            continue

        # Stop at the first section header (usually "Abstract")
        if el.label == DocItemLabel.SECTION_HEADER.value:
            break

        if el.text and _looks_like_authors(el.text):
            # A single text block may contain multiple names separated by commas
            # or newlines — split on commas and strip each token
            parts = [p.strip() for p in re.split(r',|\n', el.text) if p.strip()]
            candidates.extend(parts)

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for name in candidates:
        if name not in seen:
            unique.append(name)
            seen.add(name)

    return unique

# Labels we treat as plain text content
_TEXT_LABELS = {
    DocItemLabel.TEXT.value,
    DocItemLabel.LIST_ITEM.value,
    DocItemLabel.PARAGRAPH.value,
    DocItemLabel.FOOTNOTE.value,
    DocItemLabel.CAPTION.value,
    DocItemLabel.CODE.value,
}


def _extract_year(path: Path, doc_text: str) -> int:
    """
    Try to infer the publication year from (in order of preference):
      1. The file name  (e.g. "vaswani2017_attention.pdf")
      2. The first 2 000 characters of extracted text
    Returns 0 when nothing is found.
    """
    year_re = re.compile(r'\b(19[89]\d|20[0-2]\d)\b')

    # 1. filename
    m = year_re.search(path.stem)
    if m:
        return int(m.group())

    # 2. document text (header area)
    m = year_re.search(doc_text[:2000])
    if m:
        return int(m.group())

    return 0


def parse_pdf(pdf_path: Path) -> ParsedDocument:
    """
    Convert a single PDF file into a ParsedDocument.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        ParsedDocument with extracted metadata and ordered elements.

    Raises:
        ValueError: if Docling fails to convert the document.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Parsing: %s", pdf_path.name)
    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    paper_id = pdf_path.stem  # filename without extension

    # ------------------------------------------------------------------ #
    # Extract title (first TITLE item; fallback to paper_id)             #
    # ------------------------------------------------------------------ #
    title = paper_id
    title_page = 1
    for item, _ in doc.iterate_items():
        if item.label == DocItemLabel.TITLE and hasattr(item, "text") and item.text:
            title = item.text.strip()
            title_page = item.prov[0].page_no if item.prov else 1
            break

    # ------------------------------------------------------------------ #
    # Collect all structural elements                                     #
    # ------------------------------------------------------------------ #
    elements: list[RawElement] = []
    full_text_parts: list[str] = []

    for item, level in doc.iterate_items():
        label_value = item.label.value if hasattr(item.label, "value") else str(item.label)
        page = item.prov[0].page_no if item.prov else 0

        # --- Tables -------------------------------------------------------
        if isinstance(item, TableItem):
            md = item.export_to_markdown(doc) or ""
            caption = item.caption_text(doc) or ""
            elements.append(RawElement(
                label=DocItemLabel.TABLE.value,
                markdown=md,
                caption=caption or None,
                page=page,
                level=level,
            ))
            if caption:
                full_text_parts.append(caption)
            continue

        # --- Pictures / Figures -------------------------------------------
        if isinstance(item, PictureItem):
            caption = item.caption_text(doc) or ""
            if caption:
                elements.append(RawElement(
                    label=DocItemLabel.PICTURE.value,
                    caption=caption,
                    page=page,
                    level=level,
                ))
                full_text_parts.append(caption)
            continue

        # --- Text-bearing items (TextItem and subclasses) -----------------
        if isinstance(item, TextItem) and item.text:
            elements.append(RawElement(
                label=label_value,
                text=item.text.strip(),
                page=page,
                level=level,
            ))
            full_text_parts.append(item.text)

    # ------------------------------------------------------------------ #
    # Year + author extraction                                            #
    # ------------------------------------------------------------------ #
    year = _extract_year(pdf_path, " ".join(full_text_parts))
    authors = _extract_authors(elements, title_page)

    logger.info(
        "Parsed '%s' → %d elements, year=%s, authors=%s",
        title,
        len(elements),
        year or "unknown",
        authors or "unknown",
    )

    return ParsedDocument(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        elements=elements,
    )
