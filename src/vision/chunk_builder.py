from __future__ import annotations

import json
from typing import List

from src.vision.schemas import DrawingParseResult, DocumentChunk, RegulationMetadata, TableStructure


def build_chunks(result: DrawingParseResult) -> List[DocumentChunk]:
    """Convert a validated DrawingParseResult into the DocumentChunk shape a
    real Chunk/BM25/Vector-DB pipeline would ingest (spec section 10-11).

    No such pipeline exists in this repo — this function is the boundary. A
    real ingestion step only needs to consume List[DocumentChunk]; nothing
    upstream of this function needs to know that.
    """
    chunks: List[DocumentChunk] = []
    meta = result.drawing_metadata
    parent_id = f"{result.document_id}:p{result.page_number}"

    base_metadata = {
        "document_id": result.document_id,
        "page": result.page_number,
        "drawing_id": meta.drawing_id,
        "revision": meta.revision,
        "station_id": meta.station_id,
    }

    meta_parts = []
    if meta.drawing_id:
        meta_parts.append(f"图纸 {meta.drawing_id}" + (f"，版本 {meta.revision}" if meta.revision else ""))
    if meta.station_id:
        meta_parts.append(f"所属站点 {meta.station_id}")
    if meta.device_type:
        meta_parts.append(f"主要设备类型为 {meta.device_type}")
    if meta_parts:
        chunks.append(DocumentChunk(
            text="。".join(meta_parts) + "。",
            metadata={**base_metadata, "content_type": "drawing_metadata"},
            parent_id=parent_id,
            page_number=result.page_number,
            document_id=result.document_id,
            content_type="drawing_metadata",
        ))

    for device in result.devices:
        lines = [f"{meta.station_id + '站 ' if meta.station_id else ''}{device.device_id} 设备"]
        if device.device_name:
            lines.append(f"为{device.device_name}")
        param_texts = []
        for p in device.parameters:
            unit = p.unit or ""
            value = p.value if p.value is not None else p.raw_text
            param_texts.append(f"{p.raw_name}{value}{unit}")
        if param_texts:
            lines.append("，".join(param_texts))
        text = "".join(lines[:1]) + ("，" + "，".join(lines[1:]) if len(lines) > 1 else "") + "。"
        chunks.append(DocumentChunk(
            text=text,
            metadata={
                **base_metadata,
                "device_id": device.device_id,
                "content_type": "drawing_parameter",
            },
            parent_id=parent_id,
            page_number=result.page_number,
            document_id=result.document_id,
            content_type="drawing_parameter",
        ))

    for relation in result.relations:
        chunks.append(DocumentChunk(
            text=f"{relation.source_id} {relation.relation} {relation.target_id}"
            f"（置信度 {relation.confidence:.2f}，来源：{relation.source}）。",
            metadata={
                **base_metadata,
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "confidence": relation.confidence,
                "content_type": "drawing_relation",
            },
            parent_id=parent_id,
            page_number=result.page_number,
            document_id=result.document_id,
            content_type="drawing_relation",
        ))

    for i, table in enumerate(result.tables):
        chunks.append(DocumentChunk(
            text=json.dumps(table, ensure_ascii=False),
            metadata={**base_metadata, "content_type": "drawing_table", "table_index": i},
            parent_id=parent_id,
            page_number=result.page_number,
            document_id=result.document_id,
            content_type="drawing_table",
        ))

    return chunks


def build_text_chunk(text: str, document_id: str, page_number: int) -> List[DocumentChunk]:
    """Plain native-text / OCR text page content -> one chunk. Trivial by
    design: a native-text page needs no restructuring, just traceability
    metadata consistent with every other chunk type."""
    if not text.strip():
        return []
    return [DocumentChunk(
        text=text.strip(),
        metadata={"document_id": document_id, "page": page_number, "content_type": "text"},
        parent_id=f"{document_id}:p{page_number}",
        page_number=page_number,
        document_id=document_id,
        content_type="text",
    )]


def build_regulation_chunks(meta: RegulationMetadata, document_id: str) -> List[DocumentChunk]:
    """One chunk per RegulationClause, in natural language with the
    structural location (chapter/section/article/clause) kept in metadata
    for exact filtering — mirrors the drawing_metadata/drawing_parameter
    split: structured fields for filtering, natural language for embedding.
    """
    chunks: List[DocumentChunk] = []
    for clause in meta.clauses:
        location_parts = [p for p in (clause.chapter, clause.section, clause.article, clause.clause) if p]
        location = "".join(location_parts)
        prefix = f"{meta.document_name}{location}：" if meta.document_name else (f"{location}：" if location else "")
        chunks.append(DocumentChunk(
            text=f"{prefix}{clause.text}",
            metadata={
                "document_id": document_id,
                "document_name": meta.document_name,
                "chapter": clause.chapter,
                "section": clause.section,
                "article": clause.article,
                "clause": clause.clause,
                "content_type": "article",
            },
            parent_id=f"{document_id}:{clause.article or clause.chapter or 'root'}",
            page_number=0,  # plain-text regulation input has no page concept in this repo
            document_id=document_id,
            content_type="article",
        ))
    return chunks


def build_table_row_chunks(table: TableStructure, document_id: str, page_number: int, table_id: str = "t0") -> List[DocumentChunk]:
    """One chunk per data row, natural language ('col1: val1, col2: val2'),
    keeping row_index/table_id in metadata so a real ingestion pipeline can
    still reconstruct the whole table if needed."""
    chunks: List[DocumentChunk] = []
    for i, row in enumerate(table.rows):
        if table.headers and len(table.headers) == len(row):
            text = "，".join(f"{h}：{v}" for h, v in zip(table.headers, row) if v)
        else:
            text = "，".join(v for v in row if v)
        if not text:
            continue
        chunks.append(DocumentChunk(
            text=text,
            metadata={
                "document_id": document_id, "page": page_number, "table_id": table_id,
                "row_index": i, "content_type": "table_row",
            },
            parent_id=f"{document_id}:p{page_number}:{table_id}",
            page_number=page_number,
            document_id=document_id,
            content_type="table_row",
        ))
    return chunks
