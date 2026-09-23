"""Region-aware grid recovery.

Why this exists alongside the frozen src/vision/table_structure.py: that module
looks for lines spanning a fixed fraction of the WHOLE PAGE. A table occupying
the lower third of a drawing has internal dividers spanning only its own
height, so a page-relative threshold misses every one of them and reports the
page frame as the only grid. Measured on this repo's drawings: two vertical
lines found (the page borders), one internal column — for tables that visibly
have two columns and a dozen rows.

The fix is not a smaller threshold, which would start matching text strokes. It
is to locate the table band first and then look for lines relative to THAT
region. Nothing here modifies the frozen module; it is a separate parser with
its own freeze group.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.tables.config import GridConfig
from src.tables.schemas import BBox, ParsedTable, TableCell, deterministic_id


def image_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _line_positions(binary: np.ndarray, horizontal: bool,
                    min_fraction: float) -> List[int]:
    """Morphological line extraction along one axis.

    min_fraction is relative to the array handed in, which is what makes this
    region-aware: pass a page and it finds page-length lines, pass a table crop
    and it finds table-length ones.
    """
    height, width = binary.shape
    if horizontal:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(1, int(width * min_fraction)), 1))
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, int(height * min_fraction))))
    extracted = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    projection = extracted.sum(axis=1 if horizontal else 0)
    if projection.max() <= 0:
        return []
    threshold = projection.max() * 0.5
    positions: List[int] = []
    in_line = False
    start = 0
    for index, value in enumerate(projection):
        if value > threshold and not in_line:
            in_line, start = True, index
        elif value <= threshold and in_line:
            in_line = False
            positions.append((start + index) // 2)
    if in_line:
        positions.append((start + len(projection)) // 2)
    return positions


def _cluster(positions: Sequence[int], tolerance: int) -> List[int]:
    """Collapse near-duplicate line positions (a drawn line is several pixels
    wide and can be detected twice)."""
    if not positions:
        return []
    ordered = sorted(positions)
    clusters = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [int(sum(c) / len(c)) for c in clusters]


def find_table_regions(image: np.ndarray, cfg: GridConfig) -> List[BBox]:
    """Locate candidate table bands from long horizontal rules.

    A table shows up as a run of long horizontal lines with similar spacing.
    Bands are cut where the gap between consecutive rules jumps, which is what
    separates a title block from an unrelated table lower down the page.
    """
    binary = _binarize(image)
    height, width = binary.shape
    rules = _cluster(
        _line_positions(binary, horizontal=True, min_fraction=cfg.region_line_fraction),
        cfg.line_cluster_tolerance_px)
    # Drop rules that are the page frame itself.
    margin = int(height * cfg.page_frame_margin_fraction)
    rules = [y for y in rules if margin < y < height - margin]
    if len(rules) < cfg.min_rules_for_table:
        return []

    bands: List[List[int]] = [[rules[0]]]
    for previous, current in zip(rules, rules[1:]):
        gap = current - previous
        last_gap = (bands[-1][-1] - bands[-1][-2]) if len(bands[-1]) >= 2 else gap
        if last_gap and gap > last_gap * cfg.row_gap_break_ratio:
            bands.append([current])
        else:
            bands[-1].append(current)

    regions: List[BBox] = []
    for band in bands:
        if len(band) < cfg.min_rules_for_table:
            continue
        top, bottom = band[0], band[-1]
        strip = binary[top:bottom + 1, :]

        # Horizontal extent comes from the rules themselves, not from the page.
        # Using the page width pulls the page frame into the grid, and the
        # margins on either side then read as two extra empty columns —
        # measured here as 12x4 for a table that is plainly 12x2.
        rule_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(1, int(width * cfg.region_line_fraction)), 1))
        rules_only = cv2.morphologyEx(strip, cv2.MORPH_OPEN, rule_kernel)
        xs = cv2.findNonZero(rules_only)
        if xs is None:
            continue
        left = int(xs[:, 0, 0].min())
        right = int(xs[:, 0, 0].max())
        if right - left < cfg.line_cluster_tolerance_px * 2:
            continue
        regions.append([float(left), float(top), float(right), float(bottom)])
    return regions


def recover_grid(
    image_path: Path,
    cfg: GridConfig,
    *,
    document_id: str,
    page: int = 1,
    region: Optional[BBox] = None,
    table_index: int = 0,
    title: str = "",
) -> Optional[ParsedTable]:
    """Recover one table's cell grid from its own region.

    Returns None when no internal grid is present. Returning an empty table
    would be indistinguishable from a table that genuinely has no rows.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    if region is None:
        regions = find_table_regions(image, cfg)
        if not regions:
            return None
        region = regions[table_index] if table_index < len(regions) else regions[0]

    x0, y0, x1, y1 = (int(round(v)) for v in region)
    crop = _binarize(image[y0:y1 + 1, x0:x1 + 1])

    rows = _cluster(_line_positions(crop, True, cfg.cell_line_fraction),
                    cfg.line_cluster_tolerance_px)
    cols = _cluster(_line_positions(crop, False, cfg.cell_line_fraction),
                    cfg.line_cluster_tolerance_px)

    # A single bounding rectangle is the table's own border, not a grid. One
    # internal divider in some direction is the minimum for a real table.
    if len(rows) < 2 or len(cols) < 2 or (len(rows) - 1) * (len(cols) - 1) < 1:
        return None
    if len(rows) - 1 < cfg.min_rows or len(cols) - 1 < cfg.min_cols:
        return None

    table_id = f"T{table_index}:{deterministic_id(document_id, page, region)}"
    cells: List[TableCell] = []
    for row in range(len(rows) - 1):
        for col in range(len(cols) - 1):
            cell_bbox = [float(x0 + cols[col]), float(y0 + rows[row]),
                         float(x0 + cols[col + 1]), float(y0 + rows[row + 1])]
            cells.append(TableCell(
                cell_id=f"{table_id}:r{row}c{col}",
                row_start=row, row_end=row, col_start=col, col_end=col,
                bbox=cell_bbox))

    # Grid regularity, not text confidence — evenly spaced rules indicate a
    # well-formed table; how legible the text is belongs to OCR.
    regularity = 1.0
    for positions in (rows, cols):
        gaps = np.diff(positions)
        if len(gaps) > 1 and float(np.mean(gaps)):
            regularity *= max(0.0, 1.0 - float(np.std(gaps)) / float(np.mean(gaps)))

    return ParsedTable(
        document_id=document_id,
        table_id=table_id,
        kind="grid_table",
        page_start=page, page_end=page,
        bbox=[float(x0), float(y0), float(x1), float(y1)],
        title=title,
        cells=cells,
        row_count=len(rows) - 1,
        col_count=len(cols) - 1,
        structure_confidence=round(max(0.0, min(1.0, regularity)), 4),
        parser_source="table_parser",
        source_hash=image_sha256(image_path),
    )


def classify_region(image_path: Path, cfg: GridConfig,
                    region: Optional[BBox] = None) -> str:
    """grid_table | borderless_table_candidate | not_a_table.

    A borderless layout is reported as a CANDIDATE, never as a recovered table.
    Column positions inferred from text alignment are a hypothesis; presenting
    them as a grid would make a guess indistinguishable from a reading.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    if recover_grid(image_path, cfg, document_id="probe", region=region) is not None:
        return "grid_table"
    binary = _binarize(image)
    rules = _cluster(_line_positions(binary, True, cfg.region_line_fraction),
                     cfg.line_cluster_tolerance_px)
    return ("borderless_table_candidate"
            if len(rules) >= cfg.min_rules_for_table else "not_a_table")
