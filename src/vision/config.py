from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "visual_parser.yaml"


class RelationConfidenceConfig(BaseModel):
    corroboration_boost: float = 0.15
    max_confidence: float = 0.95


class PreprocessConfig(BaseModel):
    """Thresholds ImageQualityAnalyzer/PreprocessDecision use to decide which
    cv2 operations to run — never run the full stack unconditionally."""

    min_sharpness_for_skip_denoise: float = 80.0  # Laplacian variance below this -> denoise
    min_contrast_for_skip_binarize: float = 35.0  # grayscale std-dev below this -> binarize
    skew_angle_threshold_deg: float = 1.5
    min_brightness_uniformity: float = 0.85  # below this -> shadow_remove


class OCRConfig(BaseModel):
    engine: Literal["mock", "paddleocr"] = "mock"
    min_confidence: float = 0.6


class VisualFallbackConfig(BaseModel):
    min_ocr_confidence: float = 0.75
    max_garbled_ratio: float = 0.15
    min_image_ratio: float = 0.3
    enable_complex_table: bool = True
    enable_engineering_drawing: bool = True
    parameter_units: Dict[str, List[str]] = Field(default_factory=dict)
    relation_confidence: RelationConfidenceConfig = Field(default_factory=RelationConfidenceConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)


def load_config(path: Path | None = None) -> VisualFallbackConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fallback = raw.get("visual_fallback", {})
    return VisualFallbackConfig(
        min_ocr_confidence=fallback.get("min_ocr_confidence", 0.75),
        max_garbled_ratio=fallback.get("max_garbled_ratio", 0.15),
        min_image_ratio=fallback.get("min_image_ratio", 0.3),
        enable_complex_table=fallback.get("enable_complex_table", True),
        enable_engineering_drawing=fallback.get("enable_engineering_drawing", True),
        parameter_units=raw.get("parameter_units", {}),
        relation_confidence=RelationConfidenceConfig(**raw.get("relation_confidence", {})),
        preprocess=PreprocessConfig(**raw.get("preprocess", {})),
        ocr=OCRConfig(**raw.get("ocr", {})),
    )
