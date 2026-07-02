"""
Structure-aware + token-budget chunking.

Rules:
  • Section headers are primary split points.
  • Max token budget per chunk: CHUNK_MAX_TOKENS (≈ 512 tokens).
  • Tables     → one chunk each; surrounding prose stitched in as context.
  • Figures    → one chunk (with or without caption); context stitched in.
  • Equations  → standalone chunk; surrounding prose stitched in as context.
  • Text within a section is accumulated; split with overlap if over budget.

Phase 3 improvements
--------------------
  1. Context stitching: tables, figures, and equations include _CONTEXT_WORDS
     words of immediately preceding and following prose.  This gives the
     embedding model the semantic signal that normally lives around the element
     rather than just the raw content (which is often too sparse to embed well).
  2. Figures without captions are no longer silently dropped; a chunk is still
     created when surrounding context is available.
  3. Sequential IDs (tbl-N, fig-N, eq-N) are stored in ChunkMetadata so
     individual elements can be targeted by metadata filters.
  4. Large table markdown is truncated intelligently: the header row is kept
     and a row-count note is appended rather than slicing mid-cell.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import config
from core.schemas import Chunk, ChunkMetadata, ParsedDocument, RawElement
from docling_core.types.doc import DocItemLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------

# Labels that mark the start of a new section
_SECTION_LABELS = {
    DocItemLabel.SECTION_HEADER.value,
}

# Labels whose text should accumulate into a running text buffer
_TEXT_LABELS = {
    DocItemLabel.TEXT.value,
    DocItemLabel.LIST_ITEM.value,
    DocItemLabel.PARAGRAPH.value,
    DocItemLabel.FOOTNOTE.value,
    DocItemLabel.CODE.value,
    # CAPTION is handled inside the table / figure branches
}

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Milvus VARCHAR(8192) limit
_MAX_CONTENT_CHARS = 8_000

# _CONTEXT_WORDS and _MAX_TABLE_MARKDOWN_CHARS live in config so they can be
# tuned centrally.  Import them via the config module (see usages below).

# Regex that matches the separator row of a Markdown table: |---|---|
_TABLE_SEP_RE = re.compile(r'\|[-| :]+\|\s*\n')

# Sentence boundary splitter for boundary-aware chunking (#1). Splits after
# ., !, or ? followed by whitespace, then merges back any piece that ended on a
# known abbreviation (config.SENTENCE_ABBREVIATIONS) so "et al. Smith" stays whole.
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    """
    Split *text* into sentences, keeping abbreviations intact.

    The regex over-splits on abbreviation periods ("et al.", "e.g."); we repair
    that with a single pass that checks only the *last token* of each candidate
    sentence against config.SENTENCE_ABBREVIATIONS (one set lookup per sentence —
    negligible cost) and merges the fragment back into the previous sentence.
    """
    pieces = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    merged: List[str] = []
    for piece in pieces:
        if merged:
            # Strip leading brackets/quotes so "(cf." matches "cf.".
            last_token = merged[-1].rsplit(None, 1)[-1].lower().lstrip("([{<\"'")
            if last_token in config.SENTENCE_ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {piece}"
                continue
        merged.append(piece)
    return merged

# Section headers whose subtree is dropped as retrieval noise (#15).
_EXCLUDE_SECTION_RE = re.compile(config.EXCLUDE_SECTION_RE, re.IGNORECASE)


# ---------------------------------------------------------------------------
# [BREADCRUMB] Embedding-only breadcrumb context (improvement #4)
# ---------------------------------------------------------------------------
# build_embedding_text() returns the string that should be *embedded* for a
# chunk: a short "Paper / Section" header prepended to the chunk content. The
# header is NOT written back into chunk.content, so it never reaches Milvus,
# the BM25 index, or the LLM citation context — it only biases the dense vector.
#
# Called from core/pipeline.py at embed time. To disable, set
# config.EMBED_BREADCRUMB = False (or grep "[BREADCRUMB]" to find every touch
# point if you want to iterate on the format).
def build_embedding_text(chunk: "Chunk") -> str:
    """Compose the embed-only text for *chunk* (breadcrumb header + content)."""
    if not config.EMBED_BREADCRUMB:
        return chunk.content

    lines: List[str] = []
    if chunk.title:
        lines.append(f"Paper: {chunk.title}")
    if chunk.section:
        lines.append(f"Section: {chunk.section}")
    if not lines:
        return chunk.content
    return "\n".join(lines) + "\n---\n" + chunk.content


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 0.75 words for English."""
    return int(len(text.split()) / 0.75)


def _preceding_context(text_buffer: List[tuple[str, int]]) -> str:
    """
    Return the last config.CONTEXT_WORDS words from the accumulated text
    buffer without modifying it.
    """
    all_text = " ".join(t for t, _ in text_buffer)
    words    = all_text.split()
    if not words:
        return ""
    return " ".join(words[-config.CONTEXT_WORDS:])


def _following_context(elements: List[RawElement], start_idx: int) -> str:
    """
    Look ahead from *start_idx* in *elements* and collect up to
    config.CONTEXT_WORDS words from TEXT-like elements.  Stops at a section
    header.  No fixed element-count cap — iterates until the word budget is
    met, a section header is encountered, or elements end.
    """
    collected: List[str] = []
    for j in range(start_idx + 1, len(elements)):
        el = elements[j]
        if el.label in _SECTION_LABELS:
            break
        if el.label in _TEXT_LABELS and el.text:
            collected.extend(el.text.split())
            if len(collected) >= config.CONTEXT_WORDS:
                break
    return " ".join(collected[:config.CONTEXT_WORDS])


def _with_context(core: str, preceding: str, following: str) -> str:
    """
    Compose the final chunk content by surrounding *core* with prose snippets.

    Example output::

        [...attention is computed as a dot product of the query...]

        Table: Comparison of BLEU scores across architectures

        | Model   | EN-DE | EN-FR |
        |---------|-------|-------|
        ...

        [...bold entries indicate statistical significance (p < 0.05)...]
    """
    parts: List[str] = []
    if preceding:
        parts.append(f"[...{preceding}...]")
    parts.append(core)
    if following:
        parts.append(f"[...{following}...]")
    return "\n\n".join(parts)


def _truncate_table_markdown(markdown: str) -> str:
    """
    If *markdown* exceeds config.MAX_TABLE_MARKDOWN_CHARS, keep the header
    rows and as many data rows as fit, then append a row-count truncation note.

    Falls back to plain character truncation if no Markdown header is found.
    """
    limit = config.MAX_TABLE_MARKDOWN_CHARS
    if len(markdown) <= limit:
        return markdown

    match = _TABLE_SEP_RE.search(markdown)
    if match:
        header_end = match.end()
        header     = markdown[:header_end]
        rows       = [r for r in markdown[header_end:].splitlines() if r.strip()]

        budget   = limit - len(header) - 40   # room for truncation note
        included: List[str] = []
        used = 0
        for row in rows:
            row_len = len(row) + 1   # +1 for the newline
            if used + row_len > budget:
                break
            included.append(row)
            used += row_len

        dropped = len(rows) - len(included)
        result  = header + "\n".join(included)
        if dropped > 0:
            result += f"\n[... {dropped} rows truncated]"
        return result

    # No recognisable Markdown header — plain truncate
    return markdown[:limit] + "\n[... truncated]"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chunk_document(parsed_doc: ParsedDocument) -> List[Chunk]:
    """
    Convert a ParsedDocument into a flat list of Chunks.

    Args:
        parsed_doc: output of the docling parser.

    Returns:
        List of Chunk objects (without embeddings).
    """
    chunks: List[Chunk] = []
    current_section = "Abstract"   # reasonable default before first header
    # Buffer stores (word, page) pairs so each split sub-chunk gets the page
    # of its first word rather than the page of the flush trigger.
    text_buffer: List[tuple[str, int]] = []
    last_page: int = 0

    # Per-document sequential counters for element IDs
    table_counter    = 0
    figure_counter   = 0
    equation_counter = 0

    # (#15) True while iterating under a references/bibliography header; all
    # content in such a section is dropped until the next non-excluded header.
    in_excluded_section = False

    elements   = parsed_doc.elements
    n_elements = len(elements)

    # ---------------------------------------------------------------------- #
    # Inner helpers (closures over parsed_doc / chunks / text_buffer)        #
    # ---------------------------------------------------------------------- #

    def _make_chunk(
        chunk_type: str,
        content: str,
        section: str,
        page: int,
        meta: Optional[ChunkMetadata] = None,
    ) -> Chunk:
        return Chunk(
            paper_id   = parsed_doc.paper_id,
            title      = parsed_doc.title,
            authors    = parsed_doc.authors,
            year       = parsed_doc.year,
            section    = section,
            chunk_type = chunk_type,
            content    = content[:_MAX_CONTENT_CHARS],
            metadata   = meta if meta is not None else ChunkMetadata(page=page),
        )

    def _flush_text_buffer(section: str, fallback_page: int) -> None:
        """
        Emit the accumulated text buffer as one or more chunks using
        boundary-aware splitting (improvement #1):

          • Text is split into sentences; the final sentence of each source
            element marks a paragraph boundary.
          • A chunk grows until it reaches CHUNK_IDEAL_TOKENS, then closes at the
            next paragraph boundary. If CHUNK_MAX_TOKENS is hit first it closes
            at the current sentence boundary instead — never mid-sentence.
          • CHUNK_OVERLAP_TOKENS worth of trailing whole sentences are carried
            into the next chunk as semantic overlap.
        """
        nonlocal text_buffer
        if not text_buffer:
            return

        # Build sentence units: (sentence, page, is_paragraph_end).
        units: List[tuple[str, int, bool]] = []
        for text, pg in text_buffer:
            sentences = _split_sentences(text)
            for k, sent in enumerate(sentences):
                units.append((sent, pg, k == len(sentences) - 1))
        text_buffer = []
        if not units:
            return

        # Word budgets (1 token ≈ 0.75 words).
        ideal_words   = max(1, int(config.CHUNK_IDEAL_TOKENS * 0.75))
        max_words     = max(ideal_words, int(config.CHUNK_MAX_TOKENS * 0.75))
        overlap_words = max(0, int(config.CHUNK_OVERLAP_TOKENS * 0.75))
        # Overlap must stay strictly below max so packing always makes progress.
        overlap_words = min(overlap_words, max_words - 1)

        def _emit(buf: List[tuple[str, int]]) -> None:
            sub = " ".join(s for s, _ in buf).strip()
            if not sub:
                return
            pg = buf[0][1] or fallback_page
            chunks.append(_make_chunk("text", sub, section, pg))

        current: List[tuple[str, int]] = []   # (sentence, page)
        current_words   = 0
        new_since_flush = False               # guards against overlap-only chunks
        for sent, pg, para_end in units:
            current.append((sent, pg))
            current_words += len(sent.split())
            new_since_flush = True

            if current_words >= max_words or (current_words >= ideal_words and para_end):
                _emit(current)
                # Carry trailing whole sentences forward as semantic overlap.
                overlap: List[tuple[str, int]] = []
                ow = 0
                for s, p in reversed(current):
                    sw = len(s.split())
                    if ow + sw > overlap_words:
                        break
                    overlap.insert(0, (s, p))
                    ow += sw
                current         = overlap
                current_words   = ow
                new_since_flush = False

        # Flush remainder — unless it is purely carried-over overlap.
        if current and new_since_flush:
            _emit(current)

    # ---------------------------------------------------------------------- #
    # Main loop                                                               #
    # ---------------------------------------------------------------------- #

    for i, elem in enumerate(elements):
        page      = elem.page or last_page
        last_page = page

        # ---------------------------------------------------------------- #
        # Section header → flush buffer, update running section            #
        # A references/bibliography header (#15) toggles exclusion: its     #
        # content is dropped until the next non-matching header.            #
        # ---------------------------------------------------------------- #
        if elem.label in _SECTION_LABELS:
            _flush_text_buffer(current_section, page)
            header = elem.text.strip() if elem.text else ""
            if header and _EXCLUDE_SECTION_RE.search(header):
                in_excluded_section = True
                current_section = header
                continue
            in_excluded_section = False
            if header:
                current_section = header
            continue

        # Inside an excluded (references) section — skip all non-header items.
        if in_excluded_section:
            continue

        # ---------------------------------------------------------------- #
        # Title — already captured as parsed_doc.title; skip              #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.TITLE.value:
            continue

        # ---------------------------------------------------------------- #
        # Plain text → accumulate                                          #
        # ---------------------------------------------------------------- #
        if elem.label in _TEXT_LABELS:
            if elem.text:
                text_buffer.append((elem.text, page))
            continue

        # ---------------------------------------------------------------- #
        # Table                                                             #
        # → flush buffer                                                   #
        # → context-stitched standalone chunk with smart truncation        #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.TABLE.value:
            pre = _preceding_context(text_buffer)
            _flush_text_buffer(current_section, page)
            fol = _following_context(elements, i)

            core_parts: List[str] = []
            if elem.caption:
                core_parts.append(f"Table: {elem.caption}")
            if elem.markdown:
                core_parts.append(_truncate_table_markdown(elem.markdown))
            core = "\n\n".join(core_parts).strip()

            if core:
                table_counter += 1
                content = _with_context(core, pre, fol)
                meta    = ChunkMetadata(page=page, table_id=f"tbl-{table_counter}")
                chunks.append(_make_chunk("table", content, current_section, page, meta))
            continue

        # ---------------------------------------------------------------- #
        # Figure / Picture                                                  #
        # → flush buffer                                                   #
        # → context-stitched chunk; no-caption figures kept if context     #
        #   is available                                                    #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.PICTURE.value:
            pre = _preceding_context(text_buffer)
            _flush_text_buffer(current_section, page)
            fol = _following_context(elements, i)

            core = (
                f"Figure: {elem.caption}"
                if elem.caption
                else "Figure (no caption)"
            )

            # Only discard when there is truly nothing useful to embed
            if elem.caption or pre or fol:
                figure_counter += 1
                content = _with_context(core, pre, fol)
                meta    = ChunkMetadata(page=page, figure_id=f"fig-{figure_counter}")
                chunks.append(_make_chunk("figure", content, current_section, page, meta))
            continue

        # ---------------------------------------------------------------- #
        # Formula / Equation                                                #
        # → context-stitched standalone chunk                              #
        # NOTE: buffer is NOT flushed — equations are inline elements and  #
        #       the surrounding text flow must continue uninterrupted.     #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.FORMULA.value:
            pre = _preceding_context(text_buffer)
            fol = _following_context(elements, i)

            core = f"Equation: {elem.text}" if elem.text else "Equation"

            if elem.text or pre or fol:
                equation_counter += 1
                content = _with_context(core, pre, fol)
                meta    = ChunkMetadata(page=page, equation_id=f"eq-{equation_counter}")
                chunks.append(_make_chunk("equation", content, current_section, page, meta))

            # Also feed the formula text back into the running text buffer so
            # that inline formulas don't break the surrounding prose flow.
            # Example: "…we define attention as <formula> where x is…" should
            # remain a coherent sentence in the enclosing text chunk.
            if elem.text:
                text_buffer.append((elem.text, page))
            continue

        # ---------------------------------------------------------------- #
        # Anything else with text → treat as generic text                 #
        # ---------------------------------------------------------------- #
        if elem.text:
            text_buffer.append((elem.text, page))

    # Flush any remaining accumulated text
    _flush_text_buffer(current_section, last_page)

    logger.info(
        "Chunked '%s' → %d chunks  "
        "(tables=%d, figures=%d, equations=%d, paper_id=%s)",
        parsed_doc.title,
        len(chunks),
        table_counter,
        figure_counter,
        equation_counter,
        parsed_doc.paper_id,
    )
    return chunks
