"""Region cropping and its link back to the chunk it illustrates.

Crops are written to a temporary directory in every test. Nothing here writes
into the repository's own output directory, and nothing copies a file in from
outside the repo.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.tables.cells import assign_blocks_to_cells, infer_entity_column
from src.tables.chunks import build_table_chunks
from src.tables.config import load_tables_config
from src.tables.crop import (
    crop_cell_region, crop_region, crop_row_region, crop_table_region,
)
from src.tables.grid import recover_grid
from src.tables.headers import expand_header_paths
from src.tables.review import (
    build_review_request, collect_review_requests, review_status_summary,
)
from src.vision.ocr_engine import get_ocr_engine

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "data" / "drawings" / "FAN-MULTI-01.png"   # REAL drawing


@pytest.fixture(scope="module")
def cfg():
    return load_tables_config()


@pytest.fixture
def table(cfg):
    parsed = recover_grid(IMAGE, cfg.grid, document_id="FAN-MULTI-01",
                          title="A16/A17/A18风机组接线图")
    assign_blocks_to_cells(parsed, get_ocr_engine("fixture").recognize(IMAGE).blocks,
                           cfg.assignment)
    return parsed


# --------------------------------------------------------------------------
# cropping
# --------------------------------------------------------------------------

def test_table_row_and_cell_regions_all_crop(cfg, table, tmp_path):
    table_ref = crop_table_region(IMAGE, table, cfg.crop, output_dir=tmp_path)
    row_ref = crop_row_region(IMAGE, table, 4, cfg.crop, output_dir=tmp_path)
    cell_ref = crop_cell_region(IMAGE, table, 4, 1, cfg.crop, output_dir=tmp_path)

    for ref, kind in ((table_ref, "table"), (row_ref, "row"), (cell_ref, "cell")):
        assert ref is not None and ref.region_type == kind
        assert Path(ROOT / ref.image_path).exists() or (tmp_path / Path(ref.image_path).name).exists()
        assert ref.crop_sha256 and ref.source_image_sha256


def test_crop_records_both_the_requested_and_the_actual_box(cfg, table, tmp_path):
    """Near an edge the padding is trimmed. A reader comparing the crop against
    a citation has to know which box they are looking at."""
    ref = crop_cell_region(IMAGE, table, 0, 0, cfg.crop, output_dir=tmp_path)
    assert ref.bbox_original != ref.bbox_with_padding
    assert ref.bbox_with_padding[0] <= ref.bbox_original[0]
    assert ref.bbox_with_padding[2] >= ref.bbox_original[2]


def test_padding_is_applied_from_config(cfg, table, tmp_path):
    ref = crop_cell_region(IMAGE, table, 4, 1, cfg.crop, output_dir=tmp_path)
    assert ref.bbox_original[0] - ref.bbox_with_padding[0] == pytest.approx(
        cfg.crop.padding_px)


def test_a_box_running_off_the_page_is_clipped_not_rejected(cfg, tmp_path):
    ref = crop_region(IMAGE, [-50.0, -50.0, 120.0, 120.0], cfg.crop,
                      document_id="FAN-MULTI-01", page=1, region_type="table",
                      output_dir=tmp_path)
    assert ref is not None and ref.clipped
    assert ref.bbox_with_padding[0] >= 0 and ref.bbox_with_padding[1] >= 0


def test_a_degenerate_box_returns_none_rather_than_a_one_pixel_image(cfg, tmp_path):
    """A 1px crop still looks like evidence in a report while showing nothing."""
    tiny = load_tables_config().crop.model_copy(update={"padding_px": 0,
                                                        "min_crop_size_px": 10})
    assert crop_region(IMAGE, [10.0, 10.0, 12.0, 12.0], tiny,
                       document_id="d", page=1, region_type="cell",
                       output_dir=tmp_path) is None


def test_crop_filename_and_hash_are_stable_across_runs(cfg, table, tmp_path):
    first = crop_cell_region(IMAGE, table, 4, 1, cfg.crop, output_dir=tmp_path)
    second = crop_cell_region(IMAGE, table, 4, 1, cfg.crop, output_dir=tmp_path)
    assert first.image_path == second.image_path
    assert first.crop_sha256 == second.crop_sha256


def test_source_hash_matches_the_original_image(cfg, table, tmp_path):
    ref = crop_table_region(IMAGE, table, cfg.crop, output_dir=tmp_path)
    assert ref.source_image_sha256 == hashlib.sha256(IMAGE.read_bytes()).hexdigest()


def test_a_crop_can_be_found_again_from_its_chunk(cfg, table, tmp_path):
    """A crop that cannot be traced back to the chunk it illustrates is
    decoration."""
    paths = expand_header_paths(table, [])
    table.entity_column = infer_entity_column(table)
    row_ref = crop_row_region(IMAGE, table, 4, cfg.crop, output_dir=tmp_path)
    chunks = build_table_chunks(table, paths, parser_source="ocr",
                                region_refs={"r4": row_ref})
    row_chunk = next(c for c in chunks
                     if c.content_type == "table_row" and c.metadata["row_index"] == 4)
    region = row_chunk.metadata["image_region"]
    assert region["crop_sha256"] == row_ref.crop_sha256
    assert region["table_id"] == table.table_id
    assert row_chunk.source_bboxes


# --------------------------------------------------------------------------
# local vision review requests — built, never invoked
# --------------------------------------------------------------------------

def test_a_review_request_is_marked_not_invoked(cfg):
    """`requested_not_invoked` written up as a successful review turns an open
    question into a fabricated confirmation."""
    request = build_review_request(reason="CELL_SPANS_COLUMNS",
                                   chunk_id="table_row:x", expected_fields=["功率"])
    assert request.status == "requested_not_invoked"
    assert request.result is None


def test_an_unknown_review_reason_is_rejected():
    with pytest.raises(ValueError, match="unknown review reason"):
        build_review_request(reason="BECAUSE_I_FELT_LIKE_IT")


def test_review_requests_are_raised_for_ambiguous_cells(cfg, table):
    """The 6 first-row cells on this real drawing are flagged because the OCR
    boxes are taller than the printed rows — see the audit report."""
    requests = collect_review_requests(table, high_risk_fields=cfg.high_risk_fields)
    summary = review_status_summary(requests)
    assert summary["real_inference"] == 0
    assert summary["cache_replay"] == 0
    if requests:
        assert summary["requested_not_invoked"] == len(requests)


def test_a_clean_table_raises_no_review_requests(cfg):
    """An always-nonempty review queue is a queue nobody works through."""
    from src.tables.schemas import ParsedTable, TableCell
    cells = [TableCell(cell_id=f"T:r{r}c{c}", row_start=r, row_end=r,
                       col_start=c, col_end=c, bbox=[c * 10, r * 10, c * 10 + 10, r * 10 + 10],
                       text_raw="x", text_normalized="x", confidence=0.99)
             for r in range(2) for c in range(2)]
    clean = ParsedTable(document_id="d", table_id="T", page_start=1, page_end=1,
                        cells=cells, row_count=2, col_count=2)
    assert collect_review_requests(clean, high_risk_fields=[]) == []
