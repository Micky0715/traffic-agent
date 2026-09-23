"""Configuration for table recovery.

Every threshold that decides structure lives here. A magic number inside the
parser is a threshold nobody can find, review, or change per document set — and
these numbers are exactly the ones that need changing when the drawings change.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "tables.yaml"


class GridConfig(BaseModel):
    # Fraction used when hunting for the table BAND on a full page.
    region_line_fraction: float = 0.30
    # Fraction used INSIDE a located region. Relative to the region, which is
    # the whole point: a table's internal divider spans its own height, not the
    # page's, so a page-relative threshold never sees it.
    cell_line_fraction: float = 0.40
    line_cluster_tolerance_px: int = 6
    page_frame_margin_fraction: float = 0.02
    min_rules_for_table: int = 3
    # A jump in row spacing this much larger than the running spacing splits
    # one band into two tables.
    row_gap_break_ratio: float = 2.5
    min_rows: int = 2
    min_cols: int = 2


class AssignmentConfig(BaseModel):
    # Share of an OCR block that must fall inside one cell for the text to be
    # assigned there outright.
    min_coverage_ratio: float = 0.60
    # Below this against every cell, the block is recorded as unassigned
    # rather than pushed into the nearest one.
    min_candidate_ratio: float = 0.15
    # A block covering two cells by at least this much each is ambiguous: it
    # is kept with both candidates and flagged, not split on a guess.
    ambiguous_ratio: float = 0.30
    # Tolerance for a block sitting slightly outside the table bbox (skew).
    outside_table_tolerance_px: int = 8


class HeaderConfig(BaseModel):
    default_header_rows: int = 1
    max_header_rows: int = 3
    # A header row's cells are mostly non-numeric; used only as a hint when the
    # caller does not declare how many header rows there are.
    numeric_ratio_for_data_row: float = 0.5


class ContinuationConfig(BaseModel):
    required_signals: List[str] = Field(default_factory=lambda: [
        "adjacent_pages", "same_document", "column_count_compatible",
        "header_path_compatible", "horizontal_extent_compatible"])
    supporting_signals: List[str] = Field(default_factory=lambda: [
        "same_section", "repeated_header_on_next_page",
        "continuation_marker", "previous_unterminated"])
    min_supporting: int = 1
    horizontal_extent_tolerance_px: int = 40


class CropConfig(BaseModel):
    padding_px: int = 8
    min_crop_size_px: int = 4
    output_dir: str = "outputs/table_crops"


class TablesConfig(BaseModel):
    grid: GridConfig = Field(default_factory=GridConfig)
    assignment: AssignmentConfig = Field(default_factory=AssignmentConfig)
    header: HeaderConfig = Field(default_factory=HeaderConfig)
    continuation: ContinuationConfig = Field(default_factory=ContinuationConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    high_risk_fields: List[str] = Field(default_factory=list)


@lru_cache(maxsize=4)
def load_tables_config(path: str | None = None) -> TablesConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"tables config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return TablesConfig(
        grid=GridConfig(**(raw.get("grid") or {})),
        assignment=AssignmentConfig(**(raw.get("assignment") or {})),
        header=HeaderConfig(**(raw.get("header") or {})),
        continuation=ContinuationConfig(**(raw.get("continuation") or {})),
        crop=CropConfig(**(raw.get("crop") or {})),
        high_risk_fields=raw.get("high_risk_fields") or [],
    )
