from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

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
    """Which OCR engine backs the pipeline, and how it is initialized.

    Three engines, deliberately kept distinguishable rather than collapsed
    into one "ocr" concept — see src/vision/ocr_engine.py:
      mock      hand-authored stubs, no recognition model runs
      fixture   saved output of a real, version-stamped PaddleOCR run
      paddleocr live inference right now
    """

    engine: Literal["mock", "fixture", "paddleocr"] = "mock"
    min_confidence: float = 0.6

    # oneDNN (Intel CPU acceleration) is what actually breaks PaddleOCR on
    # this machine — not PaddleOCR itself. Default False buys working
    # inference at the cost of CPU speed; see PaddleOCREngine's docstring for
    # the measured numbers behind that trade.
    enable_mkldnn: bool = False


class DrawingTypeConfig(BaseModel):
    """What one kind of drawing is expected to carry.

    Two shapes, because two kinds of drawing exist here. Single-device
    drawings list literal labels. Multi-device drawings cannot: their labels
    are per-device prefixed, so they list required SUFFIXES and set
    per_device, and completeness is evaluated per discovered device.
    """

    title_keywords: List[str] = Field(default_factory=list)
    drawing_no_prefixes: List[str] = Field(default_factory=list)
    required_labels: List[str] = Field(default_factory=list)
    # Any ONE member satisfies the group. Not synonyms: a task that explicitly
    # asks for one member still requires that member (see
    # evaluate_field_completeness's task_required_fields).
    one_of_groups: List[List[str]] = Field(default_factory=list)
    per_device: bool = False
    required_label_suffixes: List[str] = Field(default_factory=list)
    # Per-type value rules, overriding the global defaults for named fields.
    # Naming-PREFIX conventions belong here rather than in the defaults: a
    # prefix scheme is a property of one project's equipment numbering, and
    # this repo has no such scheme document to cite, so it ships empty.
    field_value_rule_overrides: Dict[str, List[str]] = Field(default_factory=dict)
    # Entity type that unnamed per-device fields belong to on this kind of
    # drawing (功率 on a fan group drawing belongs to a fan).
    primary_entity_type: Optional[str] = None
    # Which suffix identifies one device, used to count devices independently
    # from recovered table rows and cross-check against the labels OCR found.
    identity_label_suffix: str = "设备编号"


class FieldCompletenessConfig(BaseModel):
    min_completeness: float = 0.8
    max_isolated_labels: int = 0
    max_watermark_repetition: float = 0.25
    label_value_max_gap_px: float = 260.0
    label_value_height_tolerance: float = 2.0
    label_value_min_vertical_overlap: float = 0.5
    watermark_ngram: int = 6
    watermark_min_repeats: int = 3


class ValueRuleDefinition(BaseModel):
    """One shape rule for a field value.

    `origin` records where the rule's authority comes from and is carried into
    every violation report, so nobody downstream has to guess whether a rule
    is a published standard, a project agreement, or a pattern induced from
    this dataset. Allowed values are descriptive, not decorative:

      project_domain_convention  used in this project's drawings; NOT claimed
                                 to be any national/international standard —
                                 making that claim needs a standard number,
                                 version and clause, which this repo does not
                                 have
      character_structure        the value's character set / shape, as seen in
                                 this project's equipment codes
      dataset_induced            generalized from the drawings in this repo;
                                 the weakest origin, and flagged as such
      configuration_error        referenced rule id was never defined
    """

    pattern: str
    reason: str = ""
    origin: Literal[
        "project_domain_convention", "character_structure",
        "dataset_induced", "configuration_error",
    ] = "dataset_induced"


class ValueValidationConfig(BaseModel):
    value_rule_definitions: Dict[str, ValueRuleDefinition] = Field(default_factory=dict)
    field_value_rules: Dict[str, List[str]] = Field(default_factory=dict)
    # Dash characters OCR returns in place of a hyphen. NFKC handles full-width
    # forms but does not map en/em dashes or the minus sign, so they are listed.
    hyphen_variants: List[str] = Field(default_factory=list)
    # Drop spacing adjacent to a hyphen ("M - 13" -> "M-13"). OCR inserts it
    # freely and it cannot change which piece of equipment a code names; kept
    # configurable because it is a judgement about codes, not a Unicode fold.
    collapse_spaces_around_hyphen: bool = True
    min_valid_ratio: float = 0.8
    # Below this many checked fields the ratio is noise (one reading moves it
    # between 0.0 and 1.0), so the rate-based trigger stays silent. An
    # outright-invalid value still fires its own signal.
    min_checked_fields: int = 2


class VLMFieldMapping(BaseModel):
    """How VLM output names map onto the canonical field names OCR produces.

    Without this the two sources can never be compared: the VLM returns
    drawing_id/device_id/power, OCR returns the printed Chinese labels, and a
    "conflict" between them would be undetectable.
    """

    metadata: Dict[str, str] = Field(default_factory=dict)
    device: Dict[str, str] = Field(default_factory=dict)
    parameter_names: Dict[str, str] = Field(default_factory=dict)


class EntityPatternRule(BaseModel):
    """An anchored regex mapping a device name onto an entity type.

    `pattern` must be anchored and should expose a `label` group when the name
    carries an identifier ("2A号水泵"). Anchoring is the point: a substring test
    would type 控制柜温控器 as a cabinet and 潜水泵壳体 as a pump, and a
    mistyped entity is then aligned against the wrong thing entirely.
    """

    pattern: str
    entity_type: str


class EntityTypeConfig(BaseModel):
    exact: Dict[str, str] = Field(default_factory=dict)
    patterns: List[EntityPatternRule] = Field(default_factory=list)
    # Canonical field -> entity type it belongs to on the OCR side.
    field_entity_type: Dict[str, str] = Field(default_factory=dict)
    # Fields that belong to the page rather than to any entity.
    page_level_fields: List[str] = Field(default_factory=list)
    # Entity type -> the field carrying that entity's identifier.
    identity_field: Dict[str, str] = Field(default_factory=dict)
    # Field/value overlap needed before two entities may be matched on context
    # alone. Context is the weakest method offered and is gated accordingly.
    min_context_similarity: float = 0.5


class VisualFallbackConfig(BaseModel):
    min_ocr_confidence: float = 0.75
    # Recovered-grid confidence below which the table is treated as unrecovered.
    min_table_confidence: float = 0.5
    max_garbled_ratio: float = 0.15
    min_image_ratio: float = 0.3
    enable_complex_table: bool = True
    enable_engineering_drawing: bool = True
    parameter_units: Dict[str, List[str]] = Field(default_factory=dict)
    relation_confidence: RelationConfidenceConfig = Field(default_factory=RelationConfidenceConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    # Insertion order is preserved and load-bearing: classify_drawing_type
    # takes the first title-keyword match, and group types must be tried
    # before wiring types.
    drawing_types: Dict[str, DrawingTypeConfig] = Field(default_factory=dict)
    field_completeness: FieldCompletenessConfig = Field(default_factory=FieldCompletenessConfig)
    value_validation: ValueValidationConfig = Field(default_factory=ValueValidationConfig)
    vlm_field_mapping: VLMFieldMapping = Field(default_factory=VLMFieldMapping)
    entity_types: EntityTypeConfig = Field(default_factory=EntityTypeConfig)


def load_config(path: Path | None = None) -> VisualFallbackConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fallback = raw.get("visual_fallback", {})
    return VisualFallbackConfig(
        min_ocr_confidence=fallback.get("min_ocr_confidence", 0.75),
        min_table_confidence=fallback.get("min_table_confidence", 0.5),
        max_garbled_ratio=fallback.get("max_garbled_ratio", 0.15),
        min_image_ratio=fallback.get("min_image_ratio", 0.3),
        enable_complex_table=fallback.get("enable_complex_table", True),
        enable_engineering_drawing=fallback.get("enable_engineering_drawing", True),
        parameter_units=raw.get("parameter_units", {}),
        relation_confidence=RelationConfidenceConfig(**raw.get("relation_confidence", {})),
        preprocess=PreprocessConfig(**raw.get("preprocess", {})),
        ocr=OCRConfig(**raw.get("ocr", {})),
        drawing_types={k: DrawingTypeConfig(**v) for k, v in (raw.get("drawing_types") or {}).items()},
        field_completeness=FieldCompletenessConfig(**raw.get("field_completeness", {})),
        value_validation=ValueValidationConfig(**raw.get("value_validation", {})),
        vlm_field_mapping=VLMFieldMapping(**raw.get("vlm_field_mapping", {})),
        entity_types=EntityTypeConfig(**raw.get("entity_types", {})),
    )
