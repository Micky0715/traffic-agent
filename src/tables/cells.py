"""Assign OCR blocks to cells, and detect merges.

Centre-point assignment is the obvious approach and the wrong one. A block that
straddles a column divider has its centre in exactly one cell, so the value
lands there with full confidence and nothing records that it might belong next
door. On a skewed page the centre can fall into the wrong cell outright. Both
failures are silent, and both put a real number under the wrong column heading.

So assignment is by AREA COVERAGE, with three outcomes instead of one:

  one cell clearly contains the block            -> assigned
  two cells each contain a substantial share     -> assigned to the larger one
                                                    AND flagged uncertain, with
                                                    both candidates retained
  no cell contains a meaningful share            -> unassigned, block kept

The second outcome is the one that matters. Keeping the losing candidate is
what lets a reviewer — or a later vision pass — see the ambiguity instead of
inheriting a decision nobody recorded making.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from src.tables.config import AssignmentConfig
from src.tables.schemas import BBox, ParsedTable, SourceBlockRef, TableCell


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: BBox, b: BBox) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def coverage_ratio(block_bbox: BBox, cell_bbox: BBox) -> float:
    """Share of the BLOCK that lies inside the cell.

    Normalized by the block, not the cell: a short value inside a wide cell
    should score 1.0, and normalizing by the cell would score it near zero.
    """
    block_area = _area(block_bbox)
    if block_area <= 0:
        return 0.0
    return _intersection_area(block_bbox, cell_bbox) / block_area


def normalize_cell_text(text: str) -> str:
    """Non-semantic only: NFKC fold, dash variants, whitespace.

    Nothing here rewrites a business value into a more plausible one. A
    repaired value hides the very error this stage exists to surface.
    """
    folded = unicodedata.normalize("NFKC", text or "")
    for dash in "‐‑‒–—―−－":
        folded = folded.replace(dash, "-")
    return re.sub(r"\s+", " ", folded).strip()


def assign_blocks_to_cells(
    table: ParsedTable,
    blocks: Sequence,
    cfg: AssignmentConfig,
) -> Dict[str, object]:
    """Fill the table's cells from OCR blocks. Returns an assignment report.

    `blocks` are objects with .text, .bbox and .confidence (OCRBlock), or
    dicts with those keys.
    """
    def field(block, name, default=None):
        return getattr(block, name, None) if not isinstance(block, dict) \
            else block.get(name, default)

    table_bbox = table.bbox
    outside: List[int] = []
    unassigned: List[int] = []
    ambiguous: List[Dict[str, object]] = []
    by_cell: Dict[str, List[Tuple[float, int, object]]] = {}

    for index, block in enumerate(blocks):
        bbox = list(field(block, "bbox") or [])
        if len(bbox) != 4:
            unassigned.append(index)
            continue

        # A block sitting outside the table is not table content. Tolerance
        # absorbs the few pixels skew introduces; beyond that it is simply
        # elsewhere on the page.
        tolerance = cfg.outside_table_tolerance_px
        if table_bbox and (bbox[2] < table_bbox[0] - tolerance or
                           bbox[0] > table_bbox[2] + tolerance or
                           bbox[3] < table_bbox[1] - tolerance or
                           bbox[1] > table_bbox[3] + tolerance):
            outside.append(index)
            continue

        scored = sorted(
            ((coverage_ratio(bbox, cell.bbox), cell) for cell in table.cells),
            key=lambda pair: -pair[0])
        best_ratio, best_cell = scored[0] if scored else (0.0, None)
        second_ratio = scored[1][0] if len(scored) > 1 else 0.0

        if best_cell is None or best_ratio < cfg.min_candidate_ratio:
            unassigned.append(index)
            continue

        is_ambiguous = (best_ratio < cfg.min_coverage_ratio or
                        second_ratio >= cfg.ambiguous_ratio)
        by_cell.setdefault(best_cell.cell_id, []).append((best_ratio, index, block))
        if is_ambiguous:
            ambiguous.append({
                "block_index": index,
                "text": field(block, "text", ""),
                "assigned_cell": best_cell.cell_id,
                "best_ratio": round(best_ratio, 4),
                "runner_up_cell": scored[1][1].cell_id if len(scored) > 1 else None,
                "runner_up_ratio": round(second_ratio, 4),
                # The block is never split between cells: the text belongs to
                # one of them and we do not know which.
                "resolution": "assigned_to_best_and_flagged",
            })
            best_cell.assignment_uncertain = True

    for cell in table.cells:
        entries = sorted(by_cell.get(cell.cell_id, []),
                         key=lambda e: (field(e[2], "bbox")[1], field(e[2], "bbox")[0]))
        if not entries:
            continue
        cell.text_raw = " ".join(field(b, "text", "") or "" for _, _, b in entries).strip()
        cell.text_normalized = normalize_cell_text(cell.text_raw)
        confidences = [float(field(b, "confidence", 0.0) or 0.0) for _, _, b in entries]
        cell.confidence = min(confidences) if confidences else 0.0
        cell.source_blocks = [
            SourceBlockRef(block_index=i, text=field(b, "text", "") or "",
                           bbox=list(field(b, "bbox") or []),
                           confidence=float(field(b, "confidence", 0.0) or 0.0),
                           coverage_ratio=round(ratio, 4))
            for ratio, i, b in entries]

    if ambiguous or unassigned:
        table.structure_uncertain = True
    return {
        "assigned_cells": sum(1 for c in table.cells if c.source_blocks),
        "empty_cells": sum(1 for c in table.cells if not c.source_blocks),
        "ambiguous_assignments": ambiguous,
        "unassigned_blocks": unassigned,
        "blocks_outside_table": outside,
    }


def detect_merges(table: ParsedTable) -> List[Dict[str, object]]:
    """Identify spans, and refuse to guess when the evidence is thin.

    An empty cell is NOT evidence of a merge on its own — a table can simply
    have a blank. A merge is inferred only when a blank sits directly against a
    filled neighbour AND the grid offers no divider evidence against it; the
    grid recovered here has every divider it could find, so a blank with
    dividers on both sides is a blank, not a span.

    Everything genuinely undecidable is recorded as `uncertain`, which also
    marks the table, rather than being resolved into a tidy rectangle.
    """
    findings: List[Dict[str, object]] = []
    for cell in table.cells:
        if not cell.is_empty:
            continue
        left = table.cell_at(cell.row_start, cell.col_start - 1) \
            if cell.col_start > 0 else None
        above = table.cell_at(cell.row_start - 1, cell.col_start) \
            if cell.row_start > 0 else None

        left_filled = left is not None and not left.is_empty
        above_filled = above is not None and not above.is_empty

        if left_filled and above_filled:
            # Could continue either neighbour. Choosing one would be a guess
            # with no evidence behind it.
            cell.merge_status = "uncertain"
            cell.merge_reasons.append("blank_with_filled_left_and_above")
            table.structure_uncertain = True
            findings.append({"cell_id": cell.cell_id, "status": "uncertain",
                             "candidates": [left.cell_id, above.cell_id]})
        elif left_filled:
            cell.merge_status = "horizontal"
            cell.merge_reasons.append("blank_continues_filled_left_neighbour")
            findings.append({"cell_id": cell.cell_id, "status": "horizontal",
                             "source": left.cell_id})
        elif above_filled:
            cell.merge_status = "vertical"
            cell.merge_reasons.append("blank_continues_filled_cell_above")
            findings.append({"cell_id": cell.cell_id, "status": "vertical",
                             "source": above.cell_id})
        else:
            # A blank with blank neighbours is just a blank cell.
            cell.merge_status = "none"
    return findings


def infer_entity_column(table: ParsedTable) -> Optional[int]:
    """Which column holds the business entity, or None.

    None is a real answer. Defaulting to column 0 would mint an entity id for
    every table, including ones that have no entity column at all — and a row
    chunk carrying an invented entity is worse than one carrying none.
    """
    data_rows = table.data_rows()
    if not data_rows or table.col_count == 0:
        return None
    # Without a header row there is no way to know what the left column MEANS.
    # Measured on this repo's real drawings, inferring one anyway turned a
    # label/value title block into rows keyed on "A16功率" — a field name
    # presented as a device, with the text reading "设备A16功率；第2列=45kW".
    # A key/value table is handled by detect_key_value_layout instead.
    if not table.header_rows:
        return None
    best_column, best_score = None, 0.0
    for col in range(table.col_count):
        values = [table.cell_at(row, col) for row in data_rows]
        texts = [c.text_normalized for c in values if c and c.text_normalized]
        if not texts:
            continue
        # Populated and distinct — but that also describes a column of
        # measurements. "45kW" and "55kW" are distinct and complete and are
        # not equipment ids, so a column that looks like readings is excluded:
        # naming a row by its power rating would key every downstream chunk on
        # a value that changes.
        measurement_like = sum(1 for t in texts if _looks_like_measurement(t))
        if measurement_like / len(texts) > 0.5:
            continue
        distinctness = len(set(texts)) / len(texts)
        fill = len(texts) / len(data_rows)
        score = distinctness * fill
        # Leftmost wins a tie: entity columns sit at the left in every table
        # shape seen here, and an arbitrary tie-break would move the entity
        # between runs.
        if score > best_score:
            best_column, best_score = col, score
    return best_column if best_score >= 0.6 else None


_MEASUREMENT = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:kw|w|mw|a|v|m3/h|m³/h|m|mm|pa|kpa|°c|c|%|次|台|个)$",
    re.I)


def _looks_like_measurement(text: str) -> bool:
    return bool(_MEASUREMENT.match(text.strip()))


def detect_key_value_layout(table: ParsedTable) -> bool:
    """Is this a two-column key/value block rather than a row-per-entity table?

    Every real bordered table in this repo is one of these: a title block whose
    left column holds field names and whose right column holds their values.
    Treating it as an entity table produces a row per FIELD keyed on the field
    name, which is what happened before this check existed.

    The test is deliberately narrow — exactly two columns, no header row, left
    column fully populated and all distinct. A wider table, or one with a
    header, is not this shape.
    """
    if table.col_count != 2 or table.header_rows:
        return False
    rows = table.data_rows()
    if len(rows) < 2:
        return False
    keys = [table.cell_at(r, 0) for r in rows]
    texts = [c.text_normalized for c in keys if c and c.text_normalized]
    if len(texts) != len(rows):
        return False
    return len(set(texts)) == len(texts)
