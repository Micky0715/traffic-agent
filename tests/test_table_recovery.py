"""Table structure recovery: grid, assignment, headers, merges, continuation.

Which inputs are which is stated per test, because it decides what the result
can support:

  REAL   data/drawings/*.png with data/ocr_fixtures/*.json — a replay of a
         recorded real PaddleOCR 3.7.0 run. Real recognition output, replayed;
         not live inference.
  SYNTH  ParsedTable objects built inline. The repo contains no real table with
         multi-level headers, merged cells or a cross-page continuation, so
         those behaviours can only be pinned synthetically and are labelled
         SYNTHETIC everywhere they are reported.

Nothing here branches on a file name or a case id.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tables.cells import (
    assign_blocks_to_cells, coverage_ratio, detect_merges, infer_entity_column,
    normalize_cell_text,
)
from src.tables.cells import detect_key_value_layout
from src.tables.chunks import build_table_chunks
from src.tables.config import load_tables_config
from src.tables.continuation import apply_continuation, assess_continuation
from src.tables.grid import classify_region, recover_grid
from src.tables.headers import (
    detect_header_rows, expand_header_paths, header_path_strings,
    looks_like_repeated_header,
)
from src.tables.schemas import ParsedTable, TableCell
from src.vision.ocr_engine import get_ocr_engine

ROOT = Path(__file__).resolve().parents[1]
DRAWINGS = ROOT / "data" / "drawings"
SYNTHETIC_TEST_FIXTURE = True     # for every synth_table() below


@pytest.fixture(scope="module")
def cfg():
    return load_tables_config()


# --------------------------------------------------------------------------
# REAL images
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stem,rows,cols", [
    ("FAN-MULTI-01", 12, 2),
    ("FAN-A13-02", 6, 2),
    ("PUMP-MULTI-02", 11, 2),
])
def test_real_bordered_table_grid_is_recovered(cfg, stem, rows, cols):
    """REAL. The frozen page-relative detector reports these as one column,
    because a table's internal divider spans its own height, not the page's."""
    table = recover_grid(DRAWINGS / f"{stem}.png", cfg.grid, document_id=stem)
    assert table is not None
    assert (table.row_count, table.col_count) == (rows, cols)
    assert len(table.cells) == rows * cols


@pytest.mark.parametrize("stem", ["FAN-A28-01-borderless", "FAN-A27-01-skewed"])
def test_real_borderless_or_skewed_page_yields_no_grid(cfg, stem):
    """REAL. Returning an empty table here would be indistinguishable from a
    table that genuinely has no rows."""
    assert recover_grid(DRAWINGS / f"{stem}.png", cfg.grid, document_id=stem) is None


def test_real_borderless_page_is_a_candidate_not_a_recovered_table(cfg):
    """REAL. Column positions guessed from text alignment are a hypothesis."""
    kind = classify_region(DRAWINGS / "FAN-A28-01-borderless.png", cfg.grid)
    assert kind in {"borderless_table_candidate", "not_a_table"}
    assert kind != "grid_table"


def test_real_ocr_blocks_land_in_the_right_cells(cfg):
    """REAL, end to end: image -> grid -> real OCR blocks -> cell text."""
    stem = "FAN-MULTI-01"
    image = DRAWINGS / f"{stem}.png"
    table = recover_grid(image, cfg.grid, document_id=stem)
    ocr = get_ocr_engine("fixture").recognize(image)
    report = assign_blocks_to_cells(table, ocr.blocks, cfg.assignment)

    assert report["empty_cells"] == 0
    assert report["unassigned_blocks"] == []
    texts = {(c.row_start, c.col_start): c.text_normalized for c in table.cells}
    assert texts[(0, 0)] == "图号" and texts[(0, 1)] == "FAN-MULTI-01"
    assert texts[(4, 0)] == "A16功率" and texts[(4, 1)] == "45kW"
    assert texts[(7, 1)] == "55kW"


def test_real_blocks_outside_the_table_are_excluded(cfg):
    """REAL. The drawing's title and device boxes sit above the table; pulling
    them in would put page furniture into a data row."""
    stem = "FAN-MULTI-01"
    image = DRAWINGS / f"{stem}.png"
    table = recover_grid(image, cfg.grid, document_id=stem)
    ocr = get_ocr_engine("fixture").recognize(image)
    report = assign_blocks_to_cells(table, ocr.blocks, cfg.assignment)
    excluded = {ocr.blocks[i].text for i in report["blocks_outside_table"]}
    assert "A16风机" in excluded and "A18风机" in excluded


def test_real_table_id_and_cell_ids_are_stable(cfg):
    stem = "FAN-A13-02"
    first = recover_grid(DRAWINGS / f"{stem}.png", cfg.grid, document_id=stem)
    second = recover_grid(DRAWINGS / f"{stem}.png", cfg.grid, document_id=stem)
    assert first.table_id == second.table_id
    assert [c.cell_id for c in first.cells] == [c.cell_id for c in second.cells]
    assert "uuid" not in first.table_id.lower()


# --------------------------------------------------------------------------
# block assignment
# --------------------------------------------------------------------------

def test_coverage_is_normalized_by_the_block_not_the_cell():
    """A short value inside a wide cell is fully contained; normalizing by the
    cell would score it near zero and lose it."""
    assert coverage_ratio([10, 10, 20, 20], [0, 0, 100, 100]) == pytest.approx(1.0)
    assert coverage_ratio([0, 0, 100, 100], [10, 10, 20, 20]) == pytest.approx(0.01)


def synth_table(rows: int, cols: int, width: int = 100, height: int = 20) -> ParsedTable:
    """SYNTHETIC. A clean grid with known cell geometry."""
    cells = [TableCell(cell_id=f"T:r{r}c{c}", row_start=r, row_end=r,
                       col_start=c, col_end=c,
                       bbox=[c * width, r * height, (c + 1) * width, (r + 1) * height])
             for r in range(rows) for c in range(cols)]
    return ParsedTable(document_id="synth", table_id="T", page_start=1, page_end=1,
                       bbox=[0, 0, cols * width, rows * height], cells=cells,
                       row_count=rows, col_count=cols)


class Block:
    def __init__(self, text, bbox, confidence=0.9):
        self.text, self.bbox, self.confidence = text, bbox, confidence


def test_a_block_straddling_two_cells_is_flagged_not_silently_placed(cfg):
    """SYNTHETIC. Centre-point assignment would put this in one cell with full
    confidence and record nothing about the other."""
    table = synth_table(2, 2)
    blocks = [Block("45kW", [80, 2, 120, 18])]     # half in c0, half in c1
    report = assign_blocks_to_cells(table, blocks, cfg.assignment)
    assert report["ambiguous_assignments"]
    ambiguous = report["ambiguous_assignments"][0]
    assert ambiguous["runner_up_cell"] is not None
    assert ambiguous["resolution"] == "assigned_to_best_and_flagged"
    assert table.structure_uncertain


def test_a_straddling_block_keeps_both_candidate_cells(cfg):
    table = synth_table(2, 2)
    report = assign_blocks_to_cells(table, [Block("x", [80, 2, 120, 18])],
                                    cfg.assignment)
    ambiguous = report["ambiguous_assignments"][0]
    assert ambiguous["assigned_cell"] != ambiguous["runner_up_cell"]
    assert ambiguous["best_ratio"] > 0 and ambiguous["runner_up_ratio"] > 0


def test_a_block_far_outside_the_table_is_not_forced_into_a_cell(cfg):
    table = synth_table(2, 2)
    report = assign_blocks_to_cells(table, [Block("页眉", [0, -400, 60, -380])],
                                    cfg.assignment)
    assert report["blocks_outside_table"] == [0]
    assert all(not c.source_blocks for c in table.cells)


def test_raw_and_normalized_cell_text_are_both_kept(cfg):
    table = synth_table(1, 1)
    assign_blocks_to_cells(table, [Block("ＦＡＮ－01", [5, 5, 95, 15])], cfg.assignment)
    cell = table.cells[0]
    assert cell.text_raw == "ＦＡＮ－01"        # evidence of what the page says
    assert cell.text_normalized == "FAN-01"   # what matching needs


def test_normalization_never_repairs_a_business_value():
    assert normalize_cell_text("审核专用毫AB9") == "审核专用毫AB9"
    assert normalize_cell_text("QF-I3") == "QF-I3"


# --------------------------------------------------------------------------
# multi-level headers — SYNTHETIC ONLY (no real multi-level table in the repo)
# --------------------------------------------------------------------------

def header_table(header_rows_text, data_rows_text) -> ParsedTable:
    """SYNTHETIC."""
    rows = list(header_rows_text) + list(data_rows_text)
    cols = max(len(r) for r in rows)
    table = synth_table(len(rows), cols)
    for r, row in enumerate(rows):
        for c in range(cols):
            cell = table.cell_at(r, c)
            cell.text_raw = row[c] if c < len(row) else ""
            cell.text_normalized = cell.text_raw
    return table


def test_two_level_header_expands_into_paths():
    """SYNTHETIC."""
    table = header_table(
        [["设备", "参数", "", "位置"], ["", "功率", "风量", ""]],
        [["A16", "45kW", "28000m3/h", "1号机房"]])
    paths = expand_header_paths(table, [0, 1])
    assert header_path_strings(paths) == ["设备", "参数.功率", "参数.风量", "位置"]


def test_three_level_header_expands_in_parent_to_child_order():
    """SYNTHETIC."""
    table = header_table(
        [["设备", "参数", "", ""], ["", "电气", "", "机械"], ["", "功率", "电流", "风量"]],
        [["A16", "45kW", "80A", "28000m3/h"]])
    paths = expand_header_paths(table, [0, 1, 2])
    assert paths[1] == ["参数", "电气", "功率"]
    assert paths[2] == ["参数", "电气", "电流"]
    assert paths[3] == ["参数", "机械", "风量"]


def test_remarks_column_does_not_inherit_the_neighbouring_header():
    """SYNTHETIC. Regression for a bug found by running the real pipeline: the
    备注 column came out as 备注.风量 because a blank sub-header inherited
    across a top-level boundary, after which a query for 风量 matches a
    remarks cell."""
    table = header_table(
        [["设备", "参数", "", "备注"], ["", "功率", "风量", ""]],
        [["A16", "45kW", "28000m3/h", "常用"]])
    paths = expand_header_paths(table, [0, 1])
    assert paths[3] == ["备注"]
    assert "风量" not in paths[3]


def test_a_blank_top_header_is_read_as_a_span_and_that_is_recorded():
    """SYNTHETIC, and a documented limitation rather than a triumph.

    In ["设备", "", "参数"] the blank at column 1 is indistinguishable, from the
    flattened grid alone, between "设备 spans two columns" and "column 1 simply
    has no top-level header". The rule chosen is to treat it as a span, because
    that is what a merged header cell looks like once the grid is flattened.
    The consequence is asserted here so the choice is visible: column 1 becomes
    设备.功率, not 功率.
    """
    table = header_table(
        [["设备", "", "参数"], ["", "功率", "风量"]],
        [["A16", "45kW", "28000m3/h"]])
    paths = expand_header_paths(table, [0, 1])
    assert paths[1] == ["设备", "功率"]
    assert paths[2] == ["参数", "风量"]   # the boundary still stops inheritance


def test_repeated_header_text_is_not_treated_as_one_merged_span():
    """SYNTHETIC. Two adjacent columns both headed 功率 may be two different
    measurements; collapsing them merges two quantities into one."""
    table = header_table(
        [["设备", "功率", "功率"], ["", "额定", "峰值"]],
        [["A16", "45kW", "52kW"]])
    paths = expand_header_paths(table, [0, 1])
    assert paths[1] == ["功率", "额定"] and paths[2] == ["功率", "峰值"]


def test_each_data_cell_can_name_the_header_cells_its_path_came_from():
    """SYNTHETIC."""
    table = header_table(
        [["设备", "参数", ""], ["", "功率", "风量"]],
        [["A16", "45kW", "28000m3/h"]])
    expand_header_paths(table, [0, 1])
    data_cell = table.cell_at(2, 1)
    assert data_cell.header_path == ["参数", "功率"]
    assert data_cell.header_source_cell_ids     # points at real header cells


def test_declared_header_rows_win_over_the_heuristic(cfg):
    """SYNTHETIC. Guessing header depth from geometry silently swallowed a
    title block's first row once already."""
    table = header_table([["图号", "FAN-01"]], [["名称", "A16风机"]])
    assert detect_header_rows(table, cfg.header, declared=0) == []


def test_a_repeated_header_row_is_recognised_as_a_header():
    """SYNTHETIC. On a continued table the reprinted header is not a device."""
    table = header_table([["设备", "功率"]], [["设备", "功率"], ["A16", "45kW"]])
    expand_header_paths(table, [0])
    assert looks_like_repeated_header(table, 1, [0])
    assert not looks_like_repeated_header(table, 2, [0])


# --------------------------------------------------------------------------
# merges — SYNTHETIC ONLY
# --------------------------------------------------------------------------

def test_a_blank_continuing_the_cell_to_its_left_is_a_horizontal_merge():
    """SYNTHETIC. Only the left neighbour is filled, so there is one candidate
    and no ambiguity."""
    table = header_table([["设备", "参数", "备注"]],
                         [["A16", "45kW", ""], ["A17", "55kW", ""]])
    table.header_rows = [0]
    findings = detect_merges(table)
    # Row 2's blank has a filled left neighbour and a BLANK cell above it, so
    # there is exactly one candidate. Row 1's blank sits under the 备注 header
    # and beside a filled value, which is the ambiguous case covered separately.
    horizontal = [f for f in findings
                  if f["status"] == "horizontal" and f["cell_id"].endswith("r2c2")]
    assert horizontal


def test_a_blank_continuing_the_cell_above_is_a_vertical_merge():
    """SYNTHETIC. The flattened form of a vertically merged entity column."""
    table = header_table([["设备", "参数"]], [["A16", "45kW"], ["", "28000m3/h"]])
    table.header_rows = [0]
    findings = detect_merges(table)
    assert any(f["status"] == "vertical" and f["cell_id"].endswith("r2c0")
               for f in findings)


def test_a_blank_with_filled_neighbours_both_ways_is_uncertain():
    """SYNTHETIC. It could continue either; picking one is a guess with no
    evidence behind it."""
    table = header_table([["A", "B"]], [["x", "y"], ["z", ""]])
    table.header_rows = [0]
    findings = detect_merges(table)
    uncertain = [f for f in findings if f["status"] == "uncertain"]
    assert uncertain and len(uncertain[0]["candidates"]) == 2
    assert table.structure_uncertain


def test_a_blank_with_blank_neighbours_is_just_a_blank():
    """SYNTHETIC. Emptiness alone is not evidence of a span."""
    table = header_table([["A", "B"]], [["", ""], ["", ""]])
    table.header_rows = [0]
    for finding in detect_merges(table):
        assert finding["status"] != "horizontal"


def test_a_column_of_measurements_is_not_mistaken_for_an_entity_column():
    """SYNTHETIC. 45kW and 55kW are distinct and complete, which is exactly
    what a naive "populated and distinct" test rewards. Keying rows on a power
    rating would name every row by a value that changes."""
    table = header_table([["参数", "值"]], [["功率", "45kW"], ["功率", "55kW"]])
    table.header_rows = [0]
    assert infer_entity_column(table) is None


# --------------------------------------------------------------------------
# key/value layout — REAL. Every bordered table in this repo is this shape.
# --------------------------------------------------------------------------

def test_a_headerless_two_column_table_has_no_entity_column(cfg):
    """REAL IMAGE + REAL OCR FIXTURE REPLAY.

    Regression for a measured defect: the label column of FAN-MULTI-01's title
    block was inferred as an entity column, so rows came out keyed on "A16功率"
    — a field name presented as a device — and the structure audit reported
    cross_device_contamination 12/48 = 25.0%.
    """
    table = recover_grid(DRAWINGS / "FAN-MULTI-01.png", cfg.grid,
                         document_id="FAN-MULTI-01")
    assign_blocks_to_cells(table, get_ocr_engine("fixture").recognize(
        DRAWINGS / "FAN-MULTI-01.png").blocks, cfg.assignment)
    assert table.header_rows == []
    assert infer_entity_column(table) is None
    assert detect_key_value_layout(table) is True


def test_a_key_value_row_becomes_a_field_not_a_device(cfg):
    """REAL IMAGE + REAL OCR FIXTURE REPLAY."""
    image = DRAWINGS / "FAN-MULTI-01.png"
    table = recover_grid(image, cfg.grid, document_id="FAN-MULTI-01")
    assign_blocks_to_cells(table, get_ocr_engine("fixture").recognize(image).blocks,
                           cfg.assignment)
    table.entity_column = infer_entity_column(table)
    rows = [c for c in build_table_chunks(table, expand_header_paths(table, []))
            if c.content_type == "table_row"]

    assert rows, "expected row chunks"
    # No row invents an entity id, and none reads as "设备<field name>".
    assert all(c.entity_id is None for c in rows)
    assert not any(c.text.startswith("设备") for c in rows)

    power = [c for c in rows if c.field_name == "A16功率"]
    assert len(power) == 1
    assert power[0].field_value == "45kW"
    assert power[0].metadata["layout"] == "key_value"
    # The field name still carries the device, so the value stays retrievable.
    assert "A16" in power[0].text and "45kW" in power[0].text


def test_a_table_with_a_header_still_gets_its_entity_column():
    """SYNTHETIC. The key/value fix must not disable entity tables: with a
    header row the left column's meaning is declared, so it can be used."""
    table = header_table([["设备编号", "功率"]],
                         [["A16", "45kW"], ["A17", "55kW"]])
    table.header_rows = [0]
    assert detect_key_value_layout(table) is False
    assert infer_entity_column(table) == 0


def test_a_three_column_headerless_table_is_not_key_value():
    """SYNTHETIC. Key/value is a two-column shape; a third column means the
    left cell is not simply the name of the one value beside it."""
    table = header_table([], [["A16", "45kW", "常用"], ["A17", "55kW", "备用"]])
    assert detect_key_value_layout(table) is False


def test_a_two_column_table_with_a_repeated_left_value_is_not_key_value():
    """SYNTHETIC. Repeating left values mean grouped rows, not field names."""
    table = header_table([], [["功率", "45kW"], ["功率", "55kW"]])
    assert detect_key_value_layout(table) is False


# --------------------------------------------------------------------------
# cross-page continuation — SYNTHETIC ONLY (no real cross-page table exists)
# --------------------------------------------------------------------------

def page_table(page: int, cols: int = 2, bbox=None) -> ParsedTable:
    """SYNTHETIC."""
    table = synth_table(3, cols)
    table.page_start = table.page_end = page
    table.bbox = bbox or [60.0, 200.0, 940.0, 500.0]
    table.document_id = "doc"
    return table


def test_matching_column_count_alone_never_merges(cfg):
    """SYNTHETIC. Two unrelated 2-column tables on facing pages would otherwise
    merge and invent rows belonging to a different table."""
    result = assess_continuation(
        page_table(1), page_table(2), cfg.continuation,
        previous_header_paths=["设备", "功率"],
        current_header_paths=["站点", "编号"])
    assert result["status"] == "separate_table"
    assert "not_header_path_compatible" in result["reasons"]


def test_non_adjacent_pages_never_merge(cfg):
    result = assess_continuation(page_table(1), page_table(5), cfg.continuation,
                                 previous_header_paths=["a"], current_header_paths=["a"])
    assert result["status"] == "separate_table"
    assert "not_adjacent_pages" in result["reasons"]


def test_a_different_horizontal_extent_blocks_the_merge(cfg):
    """SYNTHETIC. A table starting at a different x is a different table."""
    result = assess_continuation(
        page_table(1), page_table(2, bbox=[300.0, 200.0, 700.0, 500.0]),
        cfg.continuation, previous_header_paths=["a"], current_header_paths=["a"])
    assert result["status"] == "separate_table"
    assert "not_horizontal_extent_compatible" in result["reasons"]


def test_shape_matches_but_nothing_says_it_continued(cfg):
    """SYNTHETIC. Left separate and flagged, never merged on a guess."""
    previous, current = page_table(1), page_table(2)
    result = assess_continuation(previous, current, cfg.continuation,
                                 previous_header_paths=["设备", "功率"],
                                 current_header_paths=["设备", "功率"])
    assert result["status"] == "continuation_uncertain"
    apply_continuation(previous, current, result)
    assert current.structure_uncertain


def test_a_repeated_header_on_the_next_page_confirms_continuation(cfg):
    """SYNTHETIC."""
    previous, current = page_table(1), page_table(2)
    result = assess_continuation(previous, current, cfg.continuation,
                                 previous_header_paths=["设备", "功率"],
                                 current_header_paths=["设备", "功率"],
                                 repeated_header_on_next_page=True,
                                 previous_unterminated=True)
    assert result["status"] == "continued_confirmed"
    apply_continuation(previous, current, result)
    assert current.continuation_status == "continuation_candidate"


# --------------------------------------------------------------------------
# chunks
# --------------------------------------------------------------------------

def built_chunks(table, header_rows):
    paths = expand_header_paths(table, header_rows)
    table.entity_column = infer_entity_column(table)
    return build_table_chunks(table, paths), paths


def test_table_produces_one_parent_and_one_child_per_row():
    """SYNTHETIC."""
    table = header_table([["设备", "参数", ""], ["", "功率", "风量"]],
                         [["A16", "45kW", "28000m3/h"],
                          ["A17", "55kW", "32000m3/h"]])
    chunks, _ = built_chunks(table, [0, 1])
    assert sum(1 for c in chunks if c.content_type == "table_parent") == 1
    rows = [c for c in chunks if c.content_type == "table_row"]
    assert {c.entity_id for c in rows} == {"A16", "A17"}


def test_one_device_values_never_appear_in_another_devices_chunk():
    """SYNTHETIC."""
    table = header_table([["设备", "参数", ""], ["", "功率", "风量"]],
                         [["A16", "45kW", "28000m3/h"],
                          ["A17", "55kW", "32000m3/h"]])
    chunks, _ = built_chunks(table, [0, 1])
    rows = {c.entity_id: c.text for c in chunks if c.content_type == "table_row"}
    assert "45kW" in rows["A16"] and "55kW" not in rows["A16"]
    assert "55kW" in rows["A17"] and "45kW" not in rows["A17"]


def test_row_text_names_the_column_not_just_the_value():
    """SYNTHETIC. "A16 45kW" cannot answer "A16 的功率"."""
    table = header_table([["设备", "参数", ""], ["", "功率", "风量"]],
                         [["A16", "45kW", "28000m3/h"]])
    chunks, _ = built_chunks(table, [0, 1])
    row = next(c for c in chunks if c.content_type == "table_row")
    assert "参数.功率=45kW" in row.text


def test_a_repeated_header_row_does_not_become_a_data_chunk():
    """SYNTHETIC."""
    table = header_table([["设备", "功率"]],
                         [["设备", "功率"], ["A16", "45kW"]])
    chunks, _ = built_chunks(table, [0])
    rows = [c for c in chunks if c.content_type == "table_row"]
    assert all(c.entity_id != "设备" for c in rows)


def test_chunk_ids_are_stable_for_identical_input():
    """SYNTHETIC."""
    def build():
        table = header_table([["设备", "功率"]], [["A16", "45kW"]])
        return [c.chunk_id for c in built_chunks(table, [0])[0]]
    assert build() == build()


def test_real_table_chunks_carry_page_and_bbox(cfg):
    """REAL."""
    stem = "FAN-MULTI-01"
    image = DRAWINGS / f"{stem}.png"
    table = recover_grid(image, cfg.grid, document_id=stem, title="风机组接线图")
    assign_blocks_to_cells(table, get_ocr_engine("fixture").recognize(image).blocks,
                           cfg.assignment)
    chunks, _ = built_chunks(table, [])
    rows = [c for c in chunks if c.content_type == "table_row"]
    assert rows
    for chunk in rows:
        assert chunk.page_start == 1 and chunk.source_bboxes
        assert chunk.metadata["cell_refs"]


def test_parser_source_is_carried_and_not_relabelled():
    """SYNTHETIC. A VLM reading presented as OCR is the one provenance error
    nothing downstream can undo."""
    table = header_table([["设备", "功率"]], [["A16", "45kW"]])
    paths = expand_header_paths(table, [0])
    table.entity_column = 0
    ocr_chunks = build_table_chunks(table, paths, parser_source="ocr")
    vlm_chunks = build_table_chunks(table, paths, parser_source="vlm")
    assert {c.parser_source for c in ocr_chunks} == {"ocr"}
    assert {c.parser_source for c in vlm_chunks} == {"vlm"}
