"""Structure-aware chunking.

The rule this module exists to enforce: recover WHAT a piece of content is and
WHO it belongs to before deciding where to cut. Fixed-length splitting cuts
through the things that make a chunk answerable — it separates a clause number
from its clause text, splits one device's row away from its column headers, and
merges two devices' figures into one blob. After that no retriever can recover
the association, because the association is no longer in the text.

Four chunk kinds, each with a different unit of meaning:

  section       heading trail + prose, split at sentence boundaries
  clause        regulation code + clause number + title + body, kept together
  drawing_field label + value + provenance, one field per chunk
  table         a parent for the table, a child per business row

Overlap works differently per kind. Prose can afford a token overlap. A table
row or a clause does not need its neighbour's characters — it needs its own
context repeated: the heading trail, the clause number, the column header path,
the drawing number, the device id. Repeating context is what keeps a fragment
interpretable; repeating the previous 100 characters just adds noise.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence

from src.rag.config import ChunkingConfig, load_rag_config
from src.vision.schemas import (
    DocumentChunk, DrawingParseResult, RegulationMetadata, TableStructure,
)

_SENTENCE_END = re.compile(r"(?<=[。；！？!?;])")


def stable_chunk_id(document_id: str, kind: str, locator: str, content: str) -> str:
    """Deterministic id: same document + same structural position + same content
    produces the same id on every rebuild.

    A random UUID would break every stored reference each time a document is
    re-ingested, and would make two runs of the same pipeline undiffable.
    Content is part of the key so an edited passage gets a new id rather than
    silently changing under the old one.
    """
    digest = hashlib.sha256(
        f"{document_id}|{kind}|{locator}|{content}".encode("utf-8")).hexdigest()
    return f"{kind}:{document_id}:{digest[:16]}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def split_sentences(text: str) -> List[str]:
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


def _token_len(text: str) -> int:
    """Character count stands in for tokens. CJK is roughly one token per
    character for most tokenizers, and using a real tokenizer here would make
    chunk boundaries depend on a model that may not be installed."""
    return len(text)


# ---------------------------------------------------------------------------
# 1. prose sections
# ---------------------------------------------------------------------------

def build_section_chunks(
    document_id: str,
    sections: Sequence[Dict[str, Any]],
    cfg: ChunkingConfig,
) -> List[DocumentChunk]:
    """sections: [{"section_path": [...], "text": str, "page": int, "bbox": [...]}].

    A section is only split when it exceeds the configured budget, and then at
    sentence boundaries. Every piece keeps the full heading trail, so a
    fragment retrieved on its own still says which chapter it came from.
    """
    chunks: List[DocumentChunk] = []
    for index, section in enumerate(sections):
        path = list(section.get("section_path") or [])
        text = (section.get("text") or "").strip()
        if not text:
            continue
        page = int(section.get("page", 1))
        bbox = section.get("bbox") or []
        parent_id = f"{document_id}:section:{index}"

        pieces = _split_section(text, cfg)
        for piece_index, piece in enumerate(pieces):
            # The heading trail is prepended to the retrievable text, not just
            # stored as metadata: a lexical retriever can only match what is in
            # the text, and "4.2 风机" is often the only place the topic word
            # appears.
            prefixed = (" > ".join(path) + "\n" + piece) if path else piece
            chunks.append(DocumentChunk(
                text=prefixed,
                parent_id=parent_id,
                page_number=page,
                document_id=document_id,
                content_type="section",
                chunk_id=stable_chunk_id(document_id, "section",
                                         f"{index}.{piece_index}", piece),
                source_hash=content_hash(piece),
                page_start=page, page_end=page,
                section_path=path,
                source_bboxes=[bbox] if bbox else [],
                parser_source=section.get("parser_source", "native_text"),
                confidence=float(section.get("confidence", 1.0)),
                metadata={"piece_index": piece_index, "piece_count": len(pieces)},
            ))
    return chunks


def _split_section(text: str, cfg: ChunkingConfig) -> List[str]:
    if _token_len(text) <= cfg.section_max_tokens:
        return [text]
    sentences = split_sentences(text) or [text]
    pieces: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        candidate = "".join(current + [sentence])
        if current and _token_len(candidate) > cfg.section_max_tokens:
            pieces.append("".join(current))
            # Prose overlap repeats trailing sentences so a split mid-argument
            # still reads coherently on either side.
            current = _overlap_tail(current, cfg.section_overlap_tokens)
        current.append(sentence)
    if current:
        pieces.append("".join(current))
    return pieces


def _overlap_tail(sentences: List[str], budget: int) -> List[str]:
    tail: List[str] = []
    for sentence in reversed(sentences):
        if _token_len("".join([sentence] + tail)) > budget:
            break
        tail.insert(0, sentence)
    return tail


# ---------------------------------------------------------------------------
# 2. regulation clauses
# ---------------------------------------------------------------------------

def build_clause_chunks(
    document_id: str,
    metadata: RegulationMetadata,
    regulation_code: Optional[str] = None,
    page: int = 1,
) -> List[DocumentChunk]:
    """One chunk per clause, never splitting the number away from the text.

    A clause number with no body cannot answer anything, and a body with no
    number cannot be cited. They are one unit of meaning, so they are one chunk
    regardless of length.
    """
    chunks: List[DocumentChunk] = []
    for index, clause in enumerate(metadata.clauses):
        path = [p for p in (clause.chapter, clause.section) if p]
        locator = clause.article or clause.clause or f"idx{index}"
        # Code and clause number are repeated INTO the text: this is the
        # context repetition that replaces character overlap for clauses.
        header_bits = [b for b in (regulation_code or metadata.document_name,
                                   *path, locator) if b]
        text = " ".join(header_bits) + "\n" + clause.text
        chunks.append(DocumentChunk(
            text=text,
            parent_id=f"{document_id}:regulation",
            page_number=page,
            document_id=document_id,
            content_type="clause",
            chunk_id=stable_chunk_id(document_id, "clause", locator, clause.text),
            source_hash=content_hash(clause.text),
            page_start=page, page_end=page,
            section_path=path,
            field_name=locator,
            parser_source="regulation_parser",
            confidence=1.0,
            metadata={
                "regulation_code": regulation_code or metadata.document_name,
                "article": clause.article, "clause": clause.clause,
                "line_number": clause.line_number,
            },
        ))
    return chunks


# ---------------------------------------------------------------------------
# 3. drawing fields
# ---------------------------------------------------------------------------

def build_drawing_field_chunks(
    result: DrawingParseResult,
    parser_source: str = "vlm",
) -> List[DocumentChunk]:
    """One chunk per device parameter, each carrying its own provenance.

    Field-level rather than device-level because a query asks for one field
    ("A16 的功率是多少"), and a device-level blob forces the retriever to score
    the whole device on evidence about a different parameter.
    """
    chunks: List[DocumentChunk] = []
    meta = result.drawing_metadata
    drawing_id = meta.drawing_id or result.document_id
    parent_id = f"{result.document_id}:p{result.page_number}:drawing"

    bbox_by_field = {
        (e.field, str(e.value)): e.bbox for e in result.evidence if e.bbox}

    for device in result.devices:
        for parameter in device.parameters:
            value = parameter.raw_text or (
                f"{parameter.value}{parameter.unit or ''}" if parameter.value is not None else "")
            if not value:
                continue
            # Drawing number and device id repeated into the text: an isolated
            # "45kW" is unretrievable and uninterpretable without them.
            text = (f"图号 {drawing_id}；设备 {device.device_id}；"
                    f"{parameter.raw_name or parameter.name} {value}")
            chunks.append(DocumentChunk(
                text=text,
                parent_id=parent_id,
                page_number=result.page_number,
                document_id=result.document_id,
                content_type="drawing_field",
                chunk_id=stable_chunk_id(
                    result.document_id, "drawing_field",
                    f"{device.device_id}.{parameter.name}", value),
                source_hash=content_hash(text),
                page_start=result.page_number, page_end=result.page_number,
                entity_id=device.device_id,
                field_name=parameter.raw_name or parameter.name,
                field_value=value,
                source_bboxes=[bbox_by_field.get((parameter.name, value), [])] if
                bbox_by_field.get((parameter.name, value)) else [],
                # Never labelled "ocr" when a vision model produced it.
                parser_source=parser_source,
                confidence=1.0 if result.validation.status == "confirmed" else 0.6,
                structure_uncertain=result.validation.status in {"conflict", "incomplete"},
                metadata={"drawing_id": drawing_id,
                          "validation_status": result.validation.status},
            ))
    return chunks


# ---------------------------------------------------------------------------
# 4. tables
# ---------------------------------------------------------------------------

def _expand_header_paths(header_rows: Sequence[Sequence[str]],
                         column_count: int) -> List[List[str]]:
    """Flatten multi-level headers into one path per column.

    A blank cell inherits the value to its left on the same header row, which
    is how a merged header cell spanning several columns appears once the grid
    has been flattened. Without the inheritance, every column under a merged
    header except the first loses its parent.
    """
    paths: List[List[str]] = [[] for _ in range(column_count)]
    for level, row in enumerate(header_rows):
        carried = ""
        carried_ancestors: List[str] = []
        for column in range(column_count):
            cell = (row[column] if column < len(row) else "").strip()
            ancestors = paths[column][:level]
            if cell:
                carried, carried_ancestors = cell, ancestors
            elif carried and ancestors == carried_ancestors:
                # Blank continues the merged cell to its left ONLY while the
                # columns still share a parent. Without that guard the last
                # sub-header leaks sideways into the next top-level column: in
                # ["设备","参数","","备注"] / ["","功率","风量",""] the 备注
                # column would come out as 备注.风量, and a query for 风量 would
                # then match a remarks cell.
                pass
            else:
                carried, carried_ancestors = "", ancestors
            if carried:
                paths[column].append(carried)
    return [[part for part in path if part] for path in paths]


def build_table_chunks(
    document_id: str,
    table: TableStructure,
    *,
    table_id: str,
    page: int,
    header_row_count: int = 1,
    entity_column: int = 0,
    section_path: Optional[Sequence[str]] = None,
    table_title: str = "",
    parser_source: str = "table_parser",
    continuation_uncertain: bool = False,
) -> List[DocumentChunk]:
    """A parent chunk for the table plus one child per business row.

    The child text is written as `header.path=value` rather than a bare list of
    cells. "A16 45kW 28000m3/h" cannot answer "A16 的功率" — nothing in it says
    which number is the power. "参数.功率=45kW" can.
    """
    section_path = list(section_path or [])
    rows = [row for row in table.rows if any(cell.strip() for cell in row)]
    if not rows:
        return []

    header_rows = rows[:header_row_count] if table.headers == [] else [table.headers]
    body_rows = rows[header_row_count:] if table.headers == [] else rows
    column_count = max(len(row) for row in rows)
    header_paths = _expand_header_paths(header_rows, column_count)

    parent_text_bits = [b for b in (table_title, " > ".join(section_path)) if b]
    parent_text = ("；".join(parent_text_bits) + "；" if parent_text_bits else "") + \
        "列：" + "，".join(".".join(path) or f"第{i+1}列"
                         for i, path in enumerate(header_paths))
    parent_id = f"{document_id}:table:{table_id}"

    chunks = [DocumentChunk(
        text=parent_text,
        parent_id=parent_id,
        page_number=page,
        document_id=document_id,
        content_type="table_parent",
        chunk_id=stable_chunk_id(document_id, "table_parent", table_id, parent_text),
        source_hash=content_hash(parent_text),
        page_start=page, page_end=page,
        section_path=section_path,
        table_id=table_id,
        header_paths=header_paths,
        source_bboxes=[table.bbox] if table.bbox else [],
        parser_source=parser_source,
        confidence=table.confidence,
        structure_uncertain=continuation_uncertain,
        metadata={"table_title": table_title, "row_count": len(body_rows),
                  "continuation_uncertain": continuation_uncertain},
    )]

    carried_entity = ""
    for row_index, row in enumerate(body_rows):
        entity = (row[entity_column] if entity_column < len(row) else "").strip()
        if entity:
            carried_entity = entity
        else:
            # A blank entity cell continues the row above it — the flattened
            # form of a vertically merged entity column.
            entity = carried_entity

        pairs = []
        for column in range(column_count):
            if column == entity_column:
                continue
            value = (row[column] if column < len(row) else "").strip()
            if not value:
                continue
            path = ".".join(header_paths[column]) if column < len(header_paths) else f"第{column+1}列"
            pairs.append(f"{path}={value}")
        if not pairs:
            continue

        # Table title and entity repeated into every row: this is the context
        # repetition that replaces character overlap for tables.
        prefix_bits = [b for b in (table_title, f"设备{entity}" if entity else "") if b]
        text = "；".join(prefix_bits + pairs)
        chunks.append(DocumentChunk(
            text=text,
            parent_id=parent_id,
            page_number=page,
            document_id=document_id,
            content_type="table_row",
            chunk_id=stable_chunk_id(document_id, "table_row",
                                     f"{table_id}.{row_index}", text),
            source_hash=content_hash(text),
            page_start=page, page_end=page,
            section_path=section_path,
            table_id=table_id,
            entity_id=entity or None,
            header_paths=header_paths,
            source_bboxes=[table.bbox] if table.bbox else [],
            parser_source=parser_source,
            confidence=table.confidence,
            structure_uncertain=continuation_uncertain,
            metadata={"row_index": row_index, "table_title": table_title},
        ))
    return chunks


# ---------------------------------------------------------------------------
# cross-page continuation
# ---------------------------------------------------------------------------

def looks_like_repeated_header(row: Sequence[str], header_rows: Sequence[Sequence[str]]) -> bool:
    """A row identical to the header is a repeated header, not data."""
    normalized = [c.strip() for c in row]
    return any([c.strip() for c in header] == normalized for header in header_rows)


def assess_continuation(previous: Dict[str, Any], current: Dict[str, Any],
                        cfg: ChunkingConfig) -> Dict[str, Any]:
    """Decide whether two page-level tables are one table, and say why.

    Column count alone is never enough — two unrelated 3-column tables on
    facing pages are still two tables. Several independent signals have to
    agree, and when they do not the tables stay separate and the pair is
    flagged rather than merged on a guess. Wrongly merging invents rows that
    belong to a different table; wrongly splitting only loses continuity.
    """
    reasons: List[str] = []
    signals = {
        "adjacent_pages": current.get("page", 0) - previous.get("page", 0) == 1,
        "header_compatible": (
            [c.strip() for c in previous.get("headers", [])]
            == [c.strip() for c in current.get("headers", [])]),
        "column_count_compatible": (
            previous.get("column_count") == current.get("column_count")),
        "same_section": previous.get("section_path") == current.get("section_path"),
        "continuation_marker": bool(current.get("continuation_marker")),
        "previous_unterminated": bool(previous.get("unterminated")),
    }
    for name, value in signals.items():
        if not value:
            reasons.append(f"not_{name}")

    required = set(cfg.continuation_required_signals)
    satisfied = {name for name, value in signals.items() if value}
    hard_ok = required <= satisfied
    supporting = len(satisfied & set(cfg.continuation_supporting_signals))

    merged = hard_ok and supporting >= cfg.continuation_min_supporting
    return {
        "merged": merged,
        # Not merged but plausible: recorded so a human can look, never
        # silently joined.
        "continuation_uncertain": (not merged) and hard_ok,
        "signals": signals,
        "reasons": reasons,
    }


def build_chunks_for_document(payload: Dict[str, Any],
                              cfg: Optional[ChunkingConfig] = None) -> List[DocumentChunk]:
    """Convenience entry point over the four builders."""
    cfg = cfg or load_rag_config().chunking
    document_id = payload["document_id"]
    chunks: List[DocumentChunk] = []
    if payload.get("sections"):
        chunks += build_section_chunks(document_id, payload["sections"], cfg)
    if payload.get("regulation"):
        chunks += build_clause_chunks(
            document_id, payload["regulation"],
            regulation_code=payload.get("regulation_code"),
            page=payload.get("regulation_page", 1))
    for drawing in payload.get("drawings") or []:
        chunks += build_drawing_field_chunks(
            drawing, parser_source=payload.get("drawing_parser_source", "vlm"))
    for table in payload.get("tables") or []:
        chunks += build_table_chunks(document_id, table["structure"], **{
            k: v for k, v in table.items() if k != "structure"})
    return chunks
