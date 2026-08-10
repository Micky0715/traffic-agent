from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upstream contract: what a real OCR/Layout pipeline would need to hand this
# module. None of this is produced by a real parser in this repo (there is no
# PDF/OCR/Layout pipeline here) — these fields are hand-authored stubs in
# data/vision_fallback_cases.jsonl. Keeping the contract stable here means a
# real PageParseResult producer can be swapped in later without touching
# page_router.py / validator.py / chunk_builder.py.
# ---------------------------------------------------------------------------

class LayoutBlock(BaseModel):
    block_type: str  # e.g. "text" | "table" | "figure" | "title_block"
    bbox: List[float] = Field(default_factory=list)
    text: str = ""


class PageParseResult(BaseModel):
    page_number: int
    native_text: str = ""
    ocr_text: str = ""
    ocr_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    image_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    garbled_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    layout_blocks: List[LayoutBlock] = Field(default_factory=list)
    detected_tables: int = 0
    detected_figures: int = 0
    has_title_block: bool = False
    table_structure_recovery_failed: bool = False
    page_width: int = 0
    page_height: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegionType(str, Enum):
    TITLE_BLOCK = "title_block"
    PARAMETER_TABLE = "parameter_table"
    LEGEND = "legend"
    MAIN_DRAWING = "main_drawing"
    ANNOTATION_REGION = "annotation_region"
    LOW_CONFIDENCE_OCR_REGION = "low_confidence_ocr_region"


class Region(BaseModel):
    region_id: str
    region_type: RegionType
    bbox: List[float] = Field(default_factory=list)
    page_number: int
    image_path: str  # crop_path in the spec; a real pipeline would crop per-bbox, we point at a whole synthetic page image
    ocr_text: str = ""
    ocr_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PageRouteDecision(BaseModel):
    page_type: Literal[
        "native_text", "scanned_text", "mixed", "engineering_drawing", "complex_table"
    ]
    need_native_extract: bool = False
    need_ocr: bool = False
    need_layout: bool = False
    need_preprocess: bool = False
    need_vlm: bool = False
    vlm_regions: List[str] = Field(default_factory=list)  # region_id list
    reason: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# VLM output contract
# ---------------------------------------------------------------------------

class DrawingMetadata(BaseModel):
    drawing_id: Optional[str] = None
    drawing_name: Optional[str] = None
    drawing_type: Optional[str] = None
    revision: Optional[str] = None
    station_id: Optional[str] = None
    device_type: Optional[str] = None
    date: Optional[str] = None


class DeviceParameter(BaseModel):
    name: str  # normalized field name, e.g. "power"
    raw_name: str  # as printed on the drawing, e.g. "功率"
    raw_text: str  # full original text this was parsed from, e.g. "45kW"
    value: Optional[float] = None
    unit: Optional[str] = None


class DeviceEntity(BaseModel):
    device_id: str
    device_name: Optional[str] = None
    parameters: List[DeviceParameter] = Field(default_factory=list)


class VisualEvidence(BaseModel):
    field: str
    value: Any
    page: int
    bbox: List[float] = Field(default_factory=list)
    region_id: str
    source: Literal["vlm", "ocr", "ledger"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisualRelation(BaseModel):
    source_id: str
    relation: str
    target_id: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_bbox: List[float] = Field(default_factory=list)
    source_page: int = 0
    source: Literal["vlm_only", "corroborated"] = "vlm_only"


class ValidationResult(BaseModel):
    status: Literal["confirmed", "suspected", "conflict", "incomplete"] = "confirmed"
    warnings: List[str] = Field(default_factory=list)


class ParserTrace(BaseModel):
    ocr_used: bool = False
    vlm_used: bool = False
    vlm_model: str = ""
    regions_processed: List[str] = Field(default_factory=list)
    vlm_latency_ms: int = 0
    vlm_success: bool = True
    vlm_failure_reason: Optional[str] = None


class DrawingParseResult(BaseModel):
    document_id: str
    page_number: int

    drawing_metadata: DrawingMetadata = Field(default_factory=DrawingMetadata)
    devices: List[DeviceEntity] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    relations: List[VisualRelation] = Field(default_factory=list)
    evidence: List[VisualEvidence] = Field(default_factory=list)

    parser: ParserTrace = Field(default_factory=ParserTrace)
    validation: ValidationResult = Field(default_factory=ValidationResult)


# ---------------------------------------------------------------------------
# RAG boundary: this module's real output contract. Nothing downstream of
# this (chunking into a real vector DB / BM25 index) exists in this repo.
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_id: str
    page_number: int
    document_id: str
    content_type: Literal[
        "drawing_metadata", "drawing_parameter", "drawing_relation", "drawing_table",
        "text", "article", "table_row",
    ]


# ---------------------------------------------------------------------------
# OCR pipeline extension (this session's second round). Same rule as above:
# schemas here are a stable contract, not proof that a real engine backs them.
# See src/vision/ocr_engine.py for which engines are real vs. mock on this
# machine.
# ---------------------------------------------------------------------------

class OCRBlock(BaseModel):
    text: str
    bbox: List[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class OCRResult(BaseModel):
    text: str
    blocks: List[OCRBlock] = Field(default_factory=list)
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    engine: str = ""
    engine_available: bool = True
    error: Optional[str] = None


class ImageQualityMetrics(BaseModel):
    sharpness: float = 0.0  # Laplacian variance; lower = blurrier
    contrast: float = 0.0  # grayscale std-dev
    skew_angle_deg: float = 0.0
    brightness_uniformity: float = 1.0  # 1.0 = perfectly even illumination
    mean_brightness: float = 0.0


class PreprocessDecision(BaseModel):
    need_preprocess: bool = False
    operations: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)


class ProcessedImage(BaseModel):
    original_path: str
    processed_path: str
    operations_applied: List[str] = Field(default_factory=list)


class TableStructure(BaseModel):
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    merged_cells: List[Dict[str, Any]] = Field(default_factory=list)
    bbox: List[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RegulationClause(BaseModel):
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    text: str = ""
    line_number: int = 0


class RegulationMetadata(BaseModel):
    document_name: Optional[str] = None
    issuer: Optional[str] = None
    effective_date: Optional[str] = None
    clauses: List[RegulationClause] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)  # e.g. article-number gaps


SourceType = Literal["native_pdf", "ocr_original", "ocr_processed", "vlm", "rule", "validator", "ledger"]


class ParseEvidence(BaseModel):
    """Broader-source counterpart to VisualEvidence, used by the OCR pipeline
    (which has more source kinds than the VLM-only drawing module: native PDF
    text, OCR on the original image, OCR on a preprocessed image, etc.)."""

    field: str
    value: Any
    page: int
    bbox: List[float] = Field(default_factory=list)
    source: SourceType
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentQuality(BaseModel):
    status: Literal["ok", "warning", "error"] = "ok"
    ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    garbled_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_field_count: int = 0
    table_completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    page_continuity: bool = True
    article_continuity: bool = True
    conflict_count: int = 0
    low_confidence_count: int = 0
    warnings: List[Dict[str, str]] = Field(default_factory=list)  # {"type": ..., "message": ...}
