"""Multimodal review configuration.

Every threshold, padding and call limit comes from configs/multimodal.yaml.
Nothing in src/multimodal hard-codes one, so a reviewer can see the whole
policy in a single file instead of grepping for magic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "multimodal.yaml"

ReviewMode = Literal["disabled", "cache_replay", "live"]


@dataclass(frozen=True)
class MultimodalReviewConfig:
    mode: ReviewMode = "disabled"
    allow_live_vlm: bool = False
    max_live_calls: int = 5
    max_attempts_per_field: int = 1
    crop_padding_px: int = 24
    min_crop_width: int = 64
    min_crop_height: int = 32
    prompt_version: str = "vision_field_review_v1"
    schema_version: str = "vision_review_v1"
    require_crop_hash_match: bool = True
    require_image_hash_match: bool = True
    eligible_reasons: List[str] = field(default_factory=list)
    reason_priority: Dict[str, int] = field(default_factory=dict)
    low_confidence_threshold: float = 0.60
    max_crop_candidates: int = 2
    label_right_extension_fraction: float = 0.35
    label_context_padding_px: int = 8
    single_source_vlm_high_risk_decision: str = "human_review"
    high_risk_fields: List[str] = field(default_factory=list)

    def priority_of(self, reason: str) -> int:
        return self.reason_priority.get(reason, 1000)


@dataclass(frozen=True)
class CacheConfig:
    path: Path
    crop_dir: Path


@dataclass(frozen=True)
class MultimodalConfig:
    review: MultimodalReviewConfig
    cache: CacheConfig


def load_multimodal_config(path: Path | None = None) -> MultimodalConfig:
    raw = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    review_raw = dict(raw.get("multimodal_review", {}))
    cache_raw = dict(raw.get("cache", {}))
    return MultimodalConfig(
        review=MultimodalReviewConfig(**review_raw),
        cache=CacheConfig(
            path=ROOT / cache_raw.get(
                "path", "data/multimodal_cache/vision_review_cache.json"),
            crop_dir=ROOT / cache_raw.get(
                "crop_dir", "data/multimodal_cache/crops"),
        ),
    )
