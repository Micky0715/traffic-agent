"""Configuration for graded review-region resolution.

Every threshold lives in configs/review_regions.yaml. None of them is
calibrated against human-reviewed bbox gold, because none exists in this repo;
they are engineering starting values, and any report quoting them has to say
so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "review_regions.yaml"


@dataclass(frozen=True)
class ReviewRegionConfig:
    enable_alias_match: bool = True
    alias_similarity_threshold: float = 0.72
    crop_padding_ratio: float = 0.12
    min_crop_padding_px: int = 8
    max_crop_padding_px: int = 48
    dedup_iou_threshold: float = 0.60
    max_regions_per_page: int = 8
    max_context_area_ratio: float = 0.45
    min_region_width: int = 48
    min_region_height: int = 24
    label_right_extension_ratio: float = 0.35
    same_row_overlap_ratio: float = 0.45
    enable_full_page_diagnostic: bool = True
    diagnostic_max_long_edge: int = 1600
    diagnostic_max_pixels: int = 2_200_000
    sparse_page_max_blocks: int = 3
    field_aliases: Dict[str, List[str]] = field(default_factory=dict)
    title_block_field_order: List[str] = field(default_factory=list)

    threshold_provenance: str = (
        "engineering starting values; NOT calibrated against human-reviewed "
        "bbox gold, which does not exist in this repo")


def load_review_region_config(path: Path | None = None) -> ReviewRegionConfig:
    raw = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    return ReviewRegionConfig(**dict(raw.get("review_region_resolution", {})))
