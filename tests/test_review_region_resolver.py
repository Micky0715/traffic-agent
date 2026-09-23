"""Graded review-region resolution: L1 through L6, and every failure path.

No network, no VLM, no paid API. Tests marked REAL run against actual drawings
in data/drawings and replayed OCR fixtures; tests marked SYNTHETIC build their
own geometry, and say so because this repo has no real page that exercises
them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

import cv2
import pytest

from src.tables.cells import assign_blocks_to_cells
from src.tables.config import load_tables_config
from src.tables.grid import recover_grid
from src.vision.ocr_engine import get_ocr_engine
from src.vision.review_region_config import (
    ReviewRegionConfig, load_review_region_config,
)
from src.vision.review_region_resolver import (
    ALIAS_LABEL_RIGHT, CONTEXTUAL, DIAGNOSTIC, EXACT_LABEL_RIGHT,
    FULL_PAGE_DIAGNOSTIC, PRECISE, R_DIAGNOSTIC_DISABLED, R_IMAGE_MISSING,
    STRUCTURAL, STRUCTURAL_ROW_INFERENCE, TABLE_OR_TITLE_BLOCK,
    TABLE_VALUE_CELL, UNAVAILABLE, clip_to_image, deduplicate,
    find_alias_label, identify_key_value_rows, infer_key_value_row, iou,
    resolve_field_region, resolve_review_regions,
)

ROOT = Path(__file__).resolve().parents[1]
DRAWINGS = ROOT / "data" / "drawings"
PAGE_W, PAGE_H = 1000, 800


@pytest.fixture(scope="module")
def cfg() -> ReviewRegionConfig:
    return load_review_region_config()


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------

@dataclass
class Block:
    text: str
    bbox: List[float]


@dataclass
class OCR:
    blocks: List[Block]


@dataclass
class Cell:
    row_start: int
    col_start: int
    bbox: List[float]
    text_normalized: str = ""


@dataclass
class Table:
    cells: List[Cell]
    bbox: List[float]
    col_count: int = 2
    table_id: str = "t1"
    header_rows: List[int] = None

    def __post_init__(self):
        if self.header_rows is None:
            self.header_rows = []

    def data_rows(self) -> List[int]:
        return sorted({c.row_start for c in self.cells})

    def cell_at(self, row: int, col: int) -> Optional[Cell]:
        return next((c for c in self.cells
                     if c.row_start == row and c.col_start == col), None)


def kv_table(rows, *, x0=100, y0=100, w=600, row_h=40) -> Table:
    """SYNTHETIC two-column key/value table."""
    cells = []
    for index, (left, right) in enumerate(rows):
        top = y0 + index * row_h
        cells.append(Cell(index, 0, [x0, top, x0 + w * 0.4, top + row_h], left))
        cells.append(Cell(index, 1, [x0 + w * 0.4, top, x0 + w, top + row_h], right))
    return Table(cells=cells, bbox=[x0, y0, x0 + w, y0 + len(rows) * row_h])


# --------------------------------------------------------------------------
# L1 — exact label
# --------------------------------------------------------------------------

def test_L1_exact_label_gives_a_precise_answer_eligible_region(cfg):
    """SYNTHETIC."""
    ocr = OCR([Block("控制柜编号", [100, 200, 200, 230])])
    region = resolve_field_region(
        "控制柜编号", document_id="D", page_no=1, width=PAGE_W, height=PAGE_H,
        cfg=cfg, ocr_blocks=ocr.blocks)
    assert region.strategy == EXACT_LABEL_RIGHT
    assert region.trust_level == PRECISE
    assert region.answer_eligible is True
    assert region.localization_uncertain is False
    assert region.bbox[2] > 200          # extends right of the label


def test_L1_prefers_a_real_grid_cell_over_a_guessed_extent(cfg):
    """SYNTHETIC. A cell boundary is measured; a rightward extension is a guess."""
    table = kv_table([("控制柜编号", "FAN-CAB-23")], x0=100, y0=200, w=600, row_h=40)
    ocr = OCR([Block("控制柜编号", [100, 200, 340, 240])])
    region = resolve_field_region(
        "控制柜编号", document_id="D", page_no=1, width=PAGE_W, height=PAGE_H,
        cfg=cfg, ocr_blocks=ocr.blocks, table=table)
    value_cell = table.cell_at(0, 1)
    assert region.strategy == EXACT_LABEL_RIGHT
    # The region is the value cell (plus padding), not a fixed-width sweep.
    assert region.bbox[0] <= value_cell.bbox[0]
    assert region.bbox[2] <= value_cell.bbox[2] + cfg.max_crop_padding_px


def test_L1_fires_on_a_real_page(cfg):
    """REAL IMAGE + REAL OCR FIXTURE REPLAY. FAN-A23-01's 图号 label survives."""
    image = DRAWINGS / "FAN-A23-01.png"
    frame = cv2.imread(str(image))
    height, width = frame.shape[:2]
    ocr = get_ocr_engine("fixture").recognize(image)
    region = resolve_field_region(
        "图号", document_id="FAN-A23-01", page_no=1, width=width, height=height,
        cfg=cfg, ocr_blocks=ocr.blocks)
    assert region.strategy == EXACT_LABEL_RIGHT
    assert region.trust_level == PRECISE
    assert clip_to_image(region.bbox, width, height) == region.bbox


# --------------------------------------------------------------------------
# L2 — alias / damaged label
# --------------------------------------------------------------------------

def test_L2_alias_match_can_never_claim_precise(cfg):
    """REAL IMAGE. FAN-A24-01 OCRs 电机编号 as '电r编号' (score 0.75).

    A fuzzy label is evidence about WHICH field this is, and it is weaker than
    reading the label. Grading it `precise` would make a guess indistinguishable
    from a match.
    """
    image = DRAWINGS / "FAN-A24-01.png"
    frame = cv2.imread(str(image))
    height, width = frame.shape[:2]
    ocr = get_ocr_engine("fixture").recognize(image)
    region = resolve_field_region(
        "电机编号", document_id="FAN-A24-01", page_no=1, width=width,
        height=height, cfg=cfg, ocr_blocks=ocr.blocks)
    assert region.strategy == ALIAS_LABEL_RIGHT
    assert region.trust_level == STRUCTURAL
    assert region.localization_uncertain is True
    assert region.answer_eligible is False
    assert region.alias_match["score"] >= cfg.alias_similarity_threshold


def test_L2_refuses_a_tie_rather_than_picking_the_higher_float(cfg):
    """SYNTHETIC. Two equally good candidates is a question about which field
    this is, not a near miss."""
    ocr = OCR([Block("电机编号", [100, 100, 200, 130]),
               Block("电机编号", [100, 300, 200, 330])])
    match, info = find_alias_label("电机编号", ocr.blocks, cfg)
    assert match is None
    assert info["ambiguous"] is True


def test_L2_does_not_rescue_a_label_below_the_threshold(cfg):
    """REAL IMAGE. '制编' scores 0.364 against every configured alias.

    This is the outcome the spec demands: it must NOT be hard-coded to
    控制柜编号, and the threshold refuses it without any special case.
    """
    ocr = OCR([Block("E银行 制编", [100, 600, 260, 640])])
    match, info = find_alias_label("控制柜编号", ocr.blocks, cfg)
    assert match is None


def test_the_damaged_label_is_not_in_the_config(cfg):
    """No alias list may contain the ruined spelling — that would be the answer
    written into the rules."""
    raw = (ROOT / "configs" / "review_regions.yaml").read_text(encoding="utf-8")
    assert "制编" not in raw
    for aliases in cfg.field_aliases.values():
        assert "制编" not in aliases


# --------------------------------------------------------------------------
# L3 — key/value table row.  SYNTHETIC ONLY: no real page in this repo has
# enough identified neighbours for row inference (measured, see the report).
# --------------------------------------------------------------------------

def test_L3_uses_the_value_cell_when_the_left_cell_names_the_field(cfg):
    """SYNTHETIC."""
    table = kv_table([("图号", "FAN-X"), ("控制柜编号", "FAN-CAB-1")])
    rows = identify_key_value_rows(table, cfg)
    assert rows[1] == "控制柜编号"
    region = resolve_field_region(
        "控制柜编号", document_id="D", page_no=1, width=PAGE_W, height=PAGE_H,
        cfg=cfg, ocr_blocks=[], table=table, key_value=True, identified_rows=rows)
    assert region.strategy == TABLE_VALUE_CELL
    assert region.trust_level == STRUCTURAL
    assert region.answer_eligible is False


def test_L3_infers_a_row_from_identified_neighbours_not_from_its_own_text(cfg):
    """SYNTHETIC. The middle row's label is destroyed beyond matching; the rows
    above and below place it, and the configured field order says what belongs
    between them."""
    table = kv_table([("图号", "FAN-X"), ("名称", "A风机"),
                      ("###", "M-24"), ("控制柜编号", "FAN-CAB-1")])
    rows = identify_key_value_rows(table, cfg)
    assert 2 not in rows                          # the ruined row is unidentified
    row, anchors = infer_key_value_row("电机编号", table, cfg, rows)
    assert row == 2
    assert len(anchors) == 2                      # bounded above AND below

    region = resolve_field_region(
        "电机编号", document_id="D", page_no=1, width=PAGE_W, height=PAGE_H,
        cfg=cfg, ocr_blocks=[], table=table, key_value=True, identified_rows=rows)
    assert region.strategy == STRUCTURAL_ROW_INFERENCE
    assert region.trust_level == STRUCTURAL
    assert region.localization_uncertain is True


def test_L3_refuses_when_two_rows_could_both_be_the_target(cfg):
    """SYNTHETIC. A probable row is not a row."""
    table = kv_table([("图号", "FAN-X"), ("???", "a"), ("###", "b"),
                      ("页码", "1/1")])
    rows = identify_key_value_rows(table, cfg)
    row, reasons = infer_key_value_row("电机编号", table, cfg, rows)
    assert row is None
    assert "not uniquely determined" in " ".join(reasons)


def test_L3_refuses_when_there_is_no_anchor_below(cfg):
    """REAL-SHAPED. This is exactly why FAN-A24-01's 控制柜编号 falls through to
    L4: the only identified row is above it."""
    table = kv_table([("图号", "FAN-X"), ("电机编号", "M-1"), ("###", "x")])
    rows = identify_key_value_rows(table, cfg)
    row, reasons = infer_key_value_row("控制柜编号", table, cfg, rows)
    assert row is None
    assert "above and below" in " ".join(reasons)


def test_L3_drops_a_field_that_appears_to_occupy_two_rows(cfg):
    """SYNTHETIC. One field cannot be in two places; neither row is settled."""
    table = kv_table([("电机编号", "M-1"), ("电机编号", "M-2")])
    assert identify_key_value_rows(table, cfg) == {}


# --------------------------------------------------------------------------
# L4 — table / title block
# --------------------------------------------------------------------------

def test_L4_falls_back_to_the_table_and_is_only_contextual(cfg):
    """REAL IMAGE. FAN-A24-01's 控制柜编号 cannot be located; its table can."""
    image = DRAWINGS / "FAN-A24-01.png"
    frame = cv2.imread(str(image))
    height, width = frame.shape[:2]
    tcfg = load_tables_config()
    table = recover_grid(image, tcfg.grid, document_id="FAN-A24-01")
    assign_blocks_to_cells(
        table, get_ocr_engine("fixture").recognize(image).blocks, tcfg.assignment)
    rows = identify_key_value_rows(table, cfg)

    region = resolve_field_region(
        "控制柜编号", document_id="FAN-A24-01", page_no=1, width=width,
        height=height, cfg=cfg, ocr_blocks=[], table=table, key_value=True,
        identified_rows=rows)
    assert region.strategy == TABLE_OR_TITLE_BLOCK
    assert region.trust_level == CONTEXTUAL
    assert region.answer_eligible is False
    assert region.localization_uncertain is True


def test_a_local_crop_covering_most_of_the_page_is_re_graded(cfg):
    """SYNTHETIC. A 'local' region covering 90% of the page is a full-page call
    with a friendlier name."""
    huge = kv_table([("控制柜编号", "X")], x0=0, y0=0, w=980, row_h=760)
    region = resolve_field_region(
        "控制柜编号", document_id="D", page_no=1, width=PAGE_W, height=PAGE_H,
        cfg=cfg, ocr_blocks=[], table=huge, key_value=True,
        identified_rows={0: "控制柜编号"})
    assert region.area_ratio > cfg.max_context_area_ratio
    assert region.trust_level == CONTEXTUAL      # not structural
    assert region.answer_eligible is False


# --------------------------------------------------------------------------
# L5 — full-page diagnostic
# --------------------------------------------------------------------------

def test_L5_an_untypable_page_no_longer_produces_zero_requests(cfg):
    """REAL IMAGE. FAN-A22-01-heavyblur: 0 OCR blocks, type unknown, no grid.

    The baseline produced ZERO review requests for this page, so it vanished
    from the queue entirely. It now produces exactly one diagnostic region.
    """
    image = DRAWINGS / "FAN-A22-01-heavyblur.png"
    frame = cv2.imread(str(image))
    height, width = frame.shape[:2]
    ocr = get_ocr_engine("fixture").recognize(image)
    assert len(ocr.blocks) == 0

    regions = resolve_review_regions(
        document_id="FAN-A22-01-heavyblur", page_no=1, image_size=(width, height),
        ocr_result=ocr, parsed_table=None, drawing_type="unknown",
        target_fields=[], config=cfg)
    assert len(regions) == 1
    region = regions[0]
    assert region.strategy == FULL_PAGE_DIAGNOSTIC
    assert region.trust_level == DIAGNOSTIC
    assert region.answer_eligible is False
    assert region.localization_uncertain is True


def test_L5_downscales_and_records_the_scale(cfg):
    """SYNTHETIC. The original is never overwritten; the factor is recorded so a
    returned bbox can be mapped back."""
    small = replace(cfg, diagnostic_max_long_edge=400)
    regions = resolve_review_regions(
        document_id="D", page_no=1, image_size=(2000, 1000), ocr_result=OCR([]),
        parsed_table=None, drawing_type="unknown", target_fields=[], config=small)
    assert regions[0].scale == pytest.approx(0.2)


def test_L5_is_capped_at_one_per_page(cfg):
    """SYNTHETIC. Several unresolvable fields must not buy several full-page
    calls for the same pixels."""
    regions = resolve_review_regions(
        document_id="D", page_no=1, image_size=(PAGE_W, PAGE_H),
        ocr_result=OCR([]), parsed_table=None, drawing_type="fan_wiring",
        target_fields=["图号", "名称", "控制柜编号"], config=cfg)
    diagnostics = [r for r in regions if r.strategy == FULL_PAGE_DIAGNOSTIC]
    assert len(diagnostics) == 1


def test_L5_can_be_turned_off_and_then_the_page_is_unavailable(cfg):
    off = replace(cfg, enable_full_page_diagnostic=False)
    regions = resolve_review_regions(
        document_id="D", page_no=1, image_size=(PAGE_W, PAGE_H),
        ocr_result=OCR([]), parsed_table=None, drawing_type="unknown",
        target_fields=[], config=off)
    assert regions[0].strategy == UNAVAILABLE
    assert regions[0].unavailable_reason == R_DIAGNOSTIC_DISABLED


# --------------------------------------------------------------------------
# L6 — unavailable
# --------------------------------------------------------------------------

def test_L6_a_missing_image_is_unavailable_with_a_reason_code(cfg):
    regions = resolve_review_regions(
        document_id="D", page_no=1, image_size=None, ocr_result=None,
        parsed_table=None, drawing_type="fan_wiring",
        target_fields=["图号"], config=cfg)
    assert regions[0].strategy == UNAVAILABLE
    assert regions[0].unavailable_reason == R_IMAGE_MISSING


def test_a_clean_page_produces_no_region_at_all(cfg):
    """A page with nothing missing must not be given a diagnostic out of
    politeness — that is a call nobody asked for."""
    regions = resolve_review_regions(
        document_id="CLEAN", page_no=1, image_size=(PAGE_W, PAGE_H),
        ocr_result=OCR([Block("图号", [10, 10, 60, 30])] * 5),
        parsed_table=kv_table([("图号", "X")]), drawing_type="fan_wiring",
        target_fields=[], config=cfg)
    assert regions == []


# --------------------------------------------------------------------------
# bbox safety
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bbox", [
    [-50, -40, 300, 200],          # negative origin
    [900, 700, 5000, 5000],        # runs off the far edge
    [0, 0, 1000, 800],             # exactly the page
])
def test_clipping_never_produces_a_negative_or_zero_area_box(bbox):
    out = clip_to_image(bbox, PAGE_W, PAGE_H)
    assert out and out[0] >= 0 and out[1] >= 0
    assert out[2] <= PAGE_W and out[3] <= PAGE_H
    assert out[2] > out[0] and out[3] > out[1]


@pytest.mark.parametrize("bbox", [[100, 100, 100, 100], [5000, 5000, 6000, 6000]])
def test_a_degenerate_or_off_page_box_clips_to_nothing_without_crashing(bbox):
    assert clip_to_image(bbox, PAGE_W, PAGE_H) == []


def test_an_out_of_bounds_label_still_yields_a_usable_region(cfg):
    """SYNTHETIC. Clipped, not rejected, and never crashing."""
    ocr = OCR([Block("图号", [-20, -10, 120, 40])])
    region = resolve_field_region(
        "图号", document_id="D", page_no=1, width=PAGE_W, height=PAGE_H,
        cfg=cfg, ocr_blocks=ocr.blocks)
    assert region.bbox[0] >= 0 and region.bbox[1] >= 0
    assert region.bbox[2] <= PAGE_W and region.bbox[3] <= PAGE_H


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------

def test_overlapping_regions_merge_and_keep_every_field_and_reason(cfg):
    """SYNTHETIC. The title block cropped once per missing field pays for the
    same pixels five times."""
    from src.vision.review_region_resolver import _region
    a = _region(TABLE_OR_TITLE_BLOCK, [100, 100, 500, 400], document_id="D",
                page_no=1, fields=["图号"], anchors=["table:t1"],
                reasons=["r1"], width=PAGE_W, height=PAGE_H, cfg=cfg)
    b = _region(TABLE_OR_TITLE_BLOCK, [105, 102, 498, 402], document_id="D",
                page_no=1, fields=["名称"], anchors=["table:t1"],
                reasons=["r2"], width=PAGE_W, height=PAGE_H, cfg=cfg)
    merged = deduplicate([a, b], cfg)
    assert len(merged) == 1
    assert sorted(merged[0].target_fields) == ["名称", "图号"]
    assert set(merged[0].reasons) >= {"r1", "r2"}


def test_dedup_keeps_the_precise_region_not_the_bigger_one(cfg):
    """SYNTHETIC. Merging must never coarsen a located field."""
    from src.vision.review_region_resolver import _region
    precise = _region(EXACT_LABEL_RIGHT, [100, 100, 300, 200], document_id="D",
                      page_no=1, fields=["图号"], anchors=[], reasons=[],
                      width=PAGE_W, height=PAGE_H, cfg=cfg)
    coarse = _region(TABLE_OR_TITLE_BLOCK, [100, 100, 305, 205], document_id="D",
                     page_no=1, fields=["名称"], anchors=[], reasons=[],
                     width=PAGE_W, height=PAGE_H, cfg=cfg)
    merged = deduplicate([coarse, precise], cfg)
    assert len(merged) == 1
    assert merged[0].strategy == EXACT_LABEL_RIGHT
    assert merged[0].trust_level == PRECISE


def test_distant_regions_are_not_merged(cfg):
    from src.vision.review_region_resolver import _region
    a = _region(TABLE_OR_TITLE_BLOCK, [0, 0, 200, 200], document_id="D",
                page_no=1, fields=["a"], anchors=[], reasons=[],
                width=PAGE_W, height=PAGE_H, cfg=cfg)
    b = _region(TABLE_OR_TITLE_BLOCK, [600, 600, 800, 790], document_id="D",
                page_no=1, fields=["b"], anchors=[], reasons=[],
                width=PAGE_W, height=PAGE_H, cfg=cfg)
    assert len(deduplicate([a, b], cfg)) == 2
    assert iou(a.bbox, b.bbox) == 0.0


def test_regions_from_different_pages_never_merge(cfg):
    from src.vision.review_region_resolver import _region
    a = _region(TABLE_OR_TITLE_BLOCK, [100, 100, 500, 400], document_id="D",
                page_no=1, fields=["图号"], anchors=[], reasons=[],
                width=PAGE_W, height=PAGE_H, cfg=cfg)
    b = _region(TABLE_OR_TITLE_BLOCK, [100, 100, 500, 400], document_id="D",
                page_no=2, fields=["图号"], anchors=[], reasons=[],
                width=PAGE_W, height=PAGE_H, cfg=cfg)
    assert len(deduplicate([a, b], cfg)) == 2


# --------------------------------------------------------------------------
# grading discipline
# --------------------------------------------------------------------------

def test_only_precise_regions_are_answer_eligible(cfg):
    """The single rule that stops a title-block crop being reported as a
    located field."""
    from src.vision.review_region_resolver import (
        ANSWER_ELIGIBLE_TRUST, STRATEGY_TRUST,
    )
    assert ANSWER_ELIGIBLE_TRUST == {PRECISE}
    for strategy, trust in STRATEGY_TRUST.items():
        if strategy != EXACT_LABEL_RIGHT:
            assert trust != PRECISE, f"{strategy} must not be graded precise"


def test_thresholds_come_from_config_not_code(cfg):
    raw = (ROOT / "configs" / "review_regions.yaml").read_text(encoding="utf-8")
    for key in ("alias_similarity_threshold", "dedup_iou_threshold",
                "crop_padding_ratio", "max_context_area_ratio",
                "max_regions_per_page", "diagnostic_max_long_edge"):
        assert key in raw
    assert "not calibrated" in cfg.threshold_provenance.lower()


def test_cell_spans_columns_is_still_not_a_multimodal_trigger():
    """The 57 table-stage requests measured as false positives must not leak
    into the VLM queue through this round's new paths."""
    from src.multimodal.config import load_multimodal_config
    cfg = load_multimodal_config().review
    assert cfg.priority_of("cell_assignment_uncertain") == 99
    assert cfg.priority_of("required_field_missing") < cfg.priority_of(
        "cell_assignment_uncertain")
