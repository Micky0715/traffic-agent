from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.vision.schemas import OCRResult, TableStructure


def _line_positions(binary: np.ndarray, horizontal: bool, min_line_frac: float = 0.4) -> List[int]:
    """Real morphological line extraction: erode+dilate along one axis with
    a kernel proportional to the image size, isolating long straight lines
    (a table grid) from everything else (text, drawing symbols).
    """
    h, w = binary.shape
    if horizontal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w * min_line_frac)), 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(h * min_line_frac))))
    extracted = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    axis = 1 if horizontal else 0
    projection = extracted.sum(axis=axis)
    threshold = projection.max() * 0.5 if projection.max() > 0 else 0
    positions: List[int] = []
    in_line = False
    for i, v in enumerate(projection):
        if v > threshold and not in_line:
            in_line = True
            start = i
        elif v <= threshold and in_line:
            in_line = False
            positions.append((start + i) // 2)
    if in_line:
        positions.append((start + len(projection)) // 2)
    return positions


def _blocks_in_cell(ocr: OCRResult, x0: int, y0: int, x1: int, y1: int) -> str:
    texts = []
    for block in ocr.blocks:
        if not block.bbox or len(block.bbox) < 4:
            continue
        bx0, by0, bx1, by1 = block.bbox[0], block.bbox[1], block.bbox[2], block.bbox[3]
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            texts.append(block.text)
    return " ".join(texts)


def detect_table_structure(
    image_path: Path,
    ocr_result: Optional[OCRResult] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    has_header_row: bool = False,
) -> TableStructure:
    """Real cv2 grid-line detection for BORDERED tables only.

    Returns confidence=0.0 (and empty rows) when no clean rectangular grid
    is found — that is the deliberate signal a borderless/complex/multi-
    header table needs the VLM fallback (extract_table in
    qwen_vl_adapter.py) instead of a claimed "traditional parser" result.
    Cell *text* is only populated when an OCRResult is supplied — this
    function recovers structure (row/column boundaries), not characters.

    has_header_row defaults to False: grid geometry alone cannot tell a
    two-column title-block (label, value) apart from a table whose first row
    is a real header — the first cv2-detected row is only treated as a
    header when the caller (who knows what kind of table this actually is)
    says so explicitly. Guessing this from geometry produced a real bug in
    an earlier version of this function (documented in
    interview/ocr_bad_cases.md): the first title-block row was silently
    dropped into "headers" and never returned in `rows`.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    if roi:
        x0, y0, x1, y1 = roi
        image = image[y0:y1, x0:x1]
    else:
        x0 = y0 = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_lines = _line_positions(binary, horizontal=True)
    col_lines = _line_positions(binary, horizontal=False)

    n_rows_found = max(0, len(row_lines) - 1)
    n_cols_found = max(0, len(col_lines) - 1)

    # A single row AND a single column (2 line positions each: the two
    # boundary edges, nothing internal) is just a bounding rectangle — the
    # page border itself will always look like this. A real table needs at
    # least one *internal* divider in some direction. Caught by a real test
    # against a genuinely borderless drawing this session: without this
    # check, the page's own outer border was mistaken for a 1x1 "table".
    no_internal_grid = n_rows_found <= 1 and n_cols_found <= 1
    if len(row_lines) < 2 or len(col_lines) < 2 or no_internal_grid:
        # No clean grid — a borderless or too-degraded table. Real, honest
        # "I can't recover this" signal, not a guess.
        return TableStructure(headers=[], rows=[], merged_cells=[], bbox=[], confidence=0.0)

    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1

    rows: List[List[str]] = []
    for r in range(n_rows):
        row_cells: List[str] = []
        for c in range(n_cols):
            cell_x0, cell_x1 = col_lines[c], col_lines[c + 1]
            cell_y0, cell_y1 = row_lines[r], row_lines[r + 1]
            text = _blocks_in_cell(ocr_result, cell_x0 + x0, cell_y0 + y0, cell_x1 + x0, cell_y1 + y0) if ocr_result else ""
            row_cells.append(text)
        rows.append(row_cells)

    headers = rows[0] if rows and ocr_result and has_header_row else []
    body_rows = rows[1:] if headers else rows

    # Confidence reflects grid regularity (a well-formed table has evenly
    # spaced lines), not text-recognition confidence — that's OCR's job.
    row_gaps = np.diff(row_lines)
    col_gaps = np.diff(col_lines)
    regularity = 1.0
    if len(row_gaps) > 1:
        regularity *= max(0.0, 1.0 - float(np.std(row_gaps)) / float(np.mean(row_gaps) or 1))
    if len(col_gaps) > 1:
        regularity *= max(0.0, 1.0 - float(np.std(col_gaps)) / float(np.mean(col_gaps) or 1))
    confidence = max(0.3, min(1.0, regularity)) if ocr_result is None else max(0.3, min(1.0, regularity)) * (0.5 + 0.5 * min(1.0, ocr_result.average_confidence))

    return TableStructure(
        headers=headers,
        rows=body_rows,
        merged_cells=[],  # merged-cell recovery is not implemented — see interview/ocr_bad_cases.md
        bbox=[float(col_lines[0] + x0), float(row_lines[0] + y0), float(col_lines[-1] + x0), float(row_lines[-1] + y0)],
        confidence=float(confidence),
    )
