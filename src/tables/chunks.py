"""Table -> DocumentChunk.

Reuses the existing DocumentChunk rather than defining a parallel model: a
second chunk type would have to be indexed, retrieved and scored separately,
and the two would drift.

One parent chunk per table and one child per business row. A row child reads

    风机参数表；设备A16；参数.功率=45kW；参数.风量=28000m3/h；备注=常用

rather than "A16 45kW 28000m3/h", because the second cannot answer "A16 的功率"
— nothing in it says which number is the power. Naming the column is what makes
the value retrievable and interpretable on its own.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.tables.cells import detect_key_value_layout
from src.tables.headers import header_path_strings, looks_like_repeated_header
from src.tables.schemas import ImageRegionRef, ParsedTable, deterministic_id
from src.vision.schemas import DocumentChunk


def _cell_text(table: ParsedTable, row: int, col: int) -> str:
    cell = table.cell_at(row, col)
    return cell.text_normalized if cell else ""


def build_table_chunks(
    table: ParsedTable,
    header_paths: Dict[int, List[str]],
    *,
    parser_source: str = "table_parser",
    region_refs: Optional[Dict[str, ImageRegionRef]] = None,
) -> List[DocumentChunk]:
    """Parent + row children.

    Rows identical to a header are dropped: a header reprinted at the top of a
    continued page is not a device, and indexing it as one produces a row whose
    every value is a column name.
    """
    region_refs = region_refs or {}
    columns = header_path_strings(header_paths)
    parent_id = f"{table.document_id}:table:{table.table_id}"

    summary_bits = [b for b in (table.title, " > ".join(table.section_path)) if b]
    parent_text = ("；".join(summary_bits) + "；" if summary_bits else "") + \
        "列：" + "，".join(columns)

    def region_payload(key: str) -> Dict:
        ref = region_refs.get(key)
        return ref.model_dump(mode="json") if ref else None

    chunks = [DocumentChunk(
        text=parent_text,
        parent_id=parent_id,
        page_number=table.page_start,
        document_id=table.document_id,
        content_type="table_parent",
        chunk_id=f"table_parent:{table.document_id}:"
                 f"{deterministic_id(table.table_id, parent_text)}",
        source_hash=table.source_hash,
        page_start=table.page_start, page_end=table.page_end,
        section_path=list(table.section_path),
        table_id=table.table_id,
        header_paths=[list(header_paths.get(c, [])) for c in range(table.col_count)],
        source_bboxes=[list(table.bbox)] if table.bbox else [],
        parser_source=parser_source,
        confidence=table.structure_confidence,
        structure_uncertain=table.structure_uncertain,
        metadata={
            "table_title": table.title,
            "columns": columns,
            "row_count": len(table.data_rows()),
            "continuation_status": table.continuation_status,
            "continuation_reasons": table.continuation_reasons,
            "kind": table.kind,
            "image_region": region_payload("table"),
        },
    )]

    key_value = detect_key_value_layout(table)

    for row in table.data_rows():
        if looks_like_repeated_header(table, row, table.header_rows):
            continue

        if key_value:
            # Left cell is the field name, right cell its value. The row is a
            # FIELD, not a device, so it gets no entity id — minting one from
            # the field name is exactly the error this branch exists to avoid.
            name = _cell_text(table, row, 0)
            value = _cell_text(table, row, 1)
            if not name or not value:
                continue
            prefix = f"{table.title}；" if table.title else ""
            text = f"{prefix}{name} {value}"
            cell = table.cell_at(row, 1)
            chunks.append(DocumentChunk(
                text=text,
                parent_id=parent_id,
                page_number=table.page_start,
                document_id=table.document_id,
                content_type="table_row",
                chunk_id=f"table_row:{table.document_id}:"
                         f"{deterministic_id(table.table_id, row, text)}",
                source_hash=table.source_hash,
                page_start=table.page_start, page_end=table.page_end,
                section_path=list(table.section_path),
                table_id=table.table_id,
                entity_id=None,
                field_name=name,
                field_value=value,
                source_bboxes=[list(cell.bbox)] if cell and cell.bbox else [],
                parser_source=parser_source,
                confidence=table.structure_confidence,
                structure_uncertain=(table.structure_uncertain or
                                     bool(cell and cell.assignment_uncertain)),
                metadata={"row_index": row, "table_title": table.title,
                          "layout": "key_value",
                          "cell_refs": [c.cell_id for c in
                                        (table.cell_at(row, 0), cell) if c],
                          "continuation_status": table.continuation_status,
                          "image_region": region_payload(f"r{row}")},
            ))
            continue

        entity = (_cell_text(table, row, table.entity_column)
                  if table.entity_column is not None else "")
        pairs: List[str] = []
        cell_refs: List[str] = []
        boxes: List[List[float]] = []
        uncertain = table.structure_uncertain

        for col in range(table.col_count):
            if table.entity_column is not None and col == table.entity_column:
                continue
            cell = table.cell_at(row, col)
            if cell is None or not cell.text_normalized:
                continue
            path = ".".join(header_paths.get(col, [])) or f"第{col + 1}列"
            pairs.append(f"{path}={cell.text_normalized}")
            cell_refs.append(cell.cell_id)
            if cell.bbox:
                boxes.append(list(cell.bbox))
            uncertain = uncertain or cell.assignment_uncertain

        if not pairs:
            continue

        # Table title and entity repeated into every row. This is the context
        # repetition that replaces character overlap for tables: a row lifted
        # out of the index still says which table and which device it is from.
        prefix = [b for b in (table.title, f"设备{entity}" if entity else "") if b]
        text = "；".join(prefix + pairs)
        row_id = f"r{row}"
        chunks.append(DocumentChunk(
            text=text,
            parent_id=parent_id,
            page_number=table.page_start,
            document_id=table.document_id,
            content_type="table_row",
            chunk_id=f"table_row:{table.document_id}:"
                     f"{deterministic_id(table.table_id, row, text)}",
            source_hash=table.source_hash,
            page_start=table.page_start, page_end=table.page_end,
            section_path=list(table.section_path),
            table_id=table.table_id,
            # Only set when an entity column was actually identified. Minting
            # an entity id for a table that has none is worse than none at all.
            entity_id=entity or None,
            header_paths=[list(header_paths.get(c, []))
                          for c in range(table.col_count)],
            source_bboxes=boxes,
            parser_source=parser_source,
            confidence=table.structure_confidence,
            structure_uncertain=uncertain,
            metadata={
                "row_index": row,
                "table_title": table.title,
                "cell_refs": cell_refs,
                "continuation_status": table.continuation_status,
                "image_region": region_payload(row_id),
            },
        ))
    return chunks
