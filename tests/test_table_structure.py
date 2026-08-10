from pathlib import Path

from src.vision.schemas import OCRBlock, OCRResult
from src.vision.table_structure import detect_table_structure

DRAWINGS = Path(__file__).resolve().parents[1] / "data" / "drawings"

# FAN-A13-02.png's title block is a real bordered grid: 6 label/value rows,
# 2 columns, drawn at (60, 480)-(940, 660) — see the PIL generator script
# used earlier this session. Real cv2 grid detection must recover exactly
# this shape from actual pixels, not a hand-set expectation.
TITLE_BLOCK_ROI = (60, 480, 940, 660)


def test_detects_correct_row_and_column_count_on_a_real_bordered_table():
    table = detect_table_structure(DRAWINGS / "FAN-A13-02.png", roi=TITLE_BLOCK_ROI)
    assert len(table.rows) == 6
    assert all(len(row) == 2 for row in table.rows)
    assert table.confidence > 0.0


def test_returns_zero_confidence_and_empty_rows_when_no_grid_present():
    """The drawing title text area (top strip, no table lines at all) must
    not be reported as a low-quality table — it must be reported as no
    table, honestly (confidence 0.0), which is the real trigger for VLM
    fallback rather than a traditional-parser result."""
    table = detect_table_structure(DRAWINGS / "FAN-A13-02.png", roi=(0, 0, 1000, 85))
    assert table.confidence == 0.0
    assert table.rows == []


def test_cell_text_is_populated_when_ocr_result_is_supplied():
    ocr = OCRResult(
        text="stub", engine="mock", engine_available=True,
        blocks=[
            OCRBlock(text="图号", bbox=[70, 485, 150, 505], confidence=0.9),
            OCRBlock(text="FAN-A13-02", bbox=[180, 485, 350, 505], confidence=0.9),
        ],
        average_confidence=0.9,
    )
    table = detect_table_structure(DRAWINGS / "FAN-A13-02.png", ocr_result=ocr, roi=TITLE_BLOCK_ROI)
    first_row_text = " ".join(table.rows[0]) if table.rows else ""
    assert "图号" in first_row_text
    assert "FAN-A13-02" in first_row_text


def test_header_row_is_kept_in_rows_by_default_not_silently_dropped():
    """Regression test for a real bug found this session: has_header_row
    defaults to False, so a title-block table (every row is data, there is
    no header) must return all 6 rows, not silently lose the first one into
    an unused `headers` field."""
    ocr = OCRResult(
        text="stub", engine="mock", engine_available=True,
        blocks=[
            OCRBlock(text="图号", bbox=[70, 485, 150, 505], confidence=0.9),
            OCRBlock(text="FAN-A13-02", bbox=[180, 485, 350, 505], confidence=0.9),
        ],
        average_confidence=0.9,
    )
    table = detect_table_structure(DRAWINGS / "FAN-A13-02.png", ocr_result=ocr, roi=TITLE_BLOCK_ROI)
    assert len(table.rows) == 6
    assert table.headers == []


def test_has_header_row_true_moves_first_row_into_headers():
    ocr = OCRResult(
        text="stub", engine="mock", engine_available=True,
        blocks=[
            OCRBlock(text="图号", bbox=[70, 485, 150, 505], confidence=0.9),
            OCRBlock(text="FAN-A13-02", bbox=[180, 485, 350, 505], confidence=0.9),
        ],
        average_confidence=0.9,
    )
    table = detect_table_structure(DRAWINGS / "FAN-A13-02.png", ocr_result=ocr, roi=TITLE_BLOCK_ROI, has_header_row=True)
    assert len(table.rows) == 5  # one row promoted to headers
    assert "图号" in table.headers or "FAN-A13-02" in table.headers


def test_no_ocr_result_means_cells_are_empty_but_structure_still_recovered():
    table = detect_table_structure(DRAWINGS / "FAN-A13-02.png", roi=TITLE_BLOCK_ROI)
    assert len(table.rows) == 6
    assert all(cell == "" for row in table.rows for cell in row)


def test_borderless_drawing_does_not_mistake_page_border_for_a_table():
    """Regression test for a real bug found this session: the outer page
    border rectangle (drawn on every synthetic drawing) has exactly 2
    horizontal and 2 vertical line positions — indistinguishable from a 1x1
    grid unless the code explicitly requires an internal divider. Before the
    fix, this returned confidence=1.0 with one fabricated empty row."""
    table = detect_table_structure(DRAWINGS / "FAN-A28-01-borderless.png")
    assert table.confidence == 0.0
    assert table.rows == []


def test_merged_cells_are_never_fabricated():
    """Merged-cell recovery is explicitly not implemented (see the module
    docstring / interview/ocr_bad_cases.md) — this must stay an empty list,
    not a guess."""
    table = detect_table_structure(DRAWINGS / "FAN-A13-02.png", roi=TITLE_BLOCK_ROI)
    assert table.merged_cells == []
