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
    """One retrievable unit.

    Extended in the structured-chunking round. Every field added then carries a
    default, so the existing drawing chunker keeps working unchanged and no
    caller had to be touched.

    The additions all serve one requirement: a retrieved chunk has to be able
    to say WHERE it came from and WHAT it belongs to. Text alone cannot — a row
    reading "45kW" is useless without knowing which device, which column, which
    page and which region of that page.
    """

    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_id: str
    page_number: int
    document_id: str
    content_type: Literal[
        "drawing_metadata", "drawing_parameter", "drawing_relation", "drawing_table",
        "text", "article", "table_row",
        # structured-chunking round
        "section", "clause", "drawing_field", "table_parent",
    ]

    # Deterministic, derived from document + structural position + content hash.
    # Never a random UUID: re-ingesting an unchanged document must produce the
    # same ids, or every downstream reference breaks on each rebuild.
    chunk_id: str = ""
    source_hash: str = ""          # hash of the content this chunk was built from

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # Heading trail, e.g. ["第4章 通风系统", "4.2 风机"]. Repeated into child
    # chunks so a fragment still carries the context it was split out of.
    section_path: List[str] = Field(default_factory=list)

    table_id: Optional[str] = None
    entity_id: Optional[str] = None
    field_name: Optional[str] = None
    field_value: Optional[str] = None
    # Multi-level column headers flattened per cell, e.g. [["参数","功率"]].
    header_paths: List[List[str]] = Field(default_factory=list)

    source_bboxes: List[List[float]] = Field(default_factory=list)
    # Which stage produced the text: ocr / vlm / table_parser / regulation_parser.
    # Kept explicit so VLM-generated content is never presented as OCR原文.
    parser_source: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # True when the structure itself is in doubt (e.g. a possible but
    # unconfirmed cross-page table continuation).
    structure_uncertain: bool = False


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


class FieldCompleteness(BaseModel):
    """Silent-miss signals for one page (src/vision/field_completeness.py).

    Distinct from DocumentQuality, which aggregates across a document: this is
    per-page evidence about what OCR failed to read, and it is what lets the
    router catch a page that OCR returned confidently and incompletely.
    """

    drawing_type: str = "unknown"
    type_source: Literal["title", "drawing_no", "field_combination", "none"] = "none"

    found_fields: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    # Label read, value slot empty — the signature of occlusion damage, and
    # invisible to any confidence-based check.
    isolated_labels: List[str] = Field(default_factory=list)

    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    watermark_repetition: float = Field(default=0.0, ge=0.0, le=1.0)

    # Multi-device drawings only. A count alone is not trustworthy: reading
    # 2 devices perfectly out of 3 looks identical to reading 2 of 2 unless
    # the count is corroborated, hence the explicit uncertainty flag.
    group_device_count: Optional[int] = None
    group_cardinality_uncertain: bool = False

    reasons: List[str] = Field(default_factory=list)
    # Label->value pairings behind the completeness numbers, exposed so value
    # validation reuses the same geometry instead of re-deriving it (and
    # possibly disagreeing about which block is a field's value).
    field_pairs: List["FieldPair"] = Field(default_factory=list)


class OCRRouteDecision(BaseModel):
    """Whether one page needs VLM fallback, and every reason it does.

    Reasons are kept as a list rather than collapsed to a single cause: a page
    can fail several ways at once, and the per-reason breakdown is what makes
    a false-trigger rate diagnosable instead of just a number.
    """

    need_vlm: bool = False
    reasons: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Field-value validation and field-level evidence (round 3)
# ---------------------------------------------------------------------------

class FieldPair(BaseModel):
    """One label->value pairing recovered from OCR geometry.

    entity_id names WHICH device the value belongs to on a multi-device
    drawing, and is None for page-level fields. Conflict detection keys on
    (entity_id, field_name): without it, two different motors legitimately
    carrying different 电机编号 on one drawing read as a contradiction.
    """

    entity_id: Optional[str] = None
    field_name: str
    raw_value: str = ""
    bbox: List[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class InvalidFieldDetail(BaseModel):
    field_name: str
    entity_id: Optional[str] = None
    observed_value: str          # raw, exactly as OCR returned it
    normalized_value: str        # after non-semantic normalization only
    violated_rule: str           # rule id, so a report can be grouped by cause
    rule_pattern: str
    rule_origin: str             # where the rule's authority comes from
    reason: str


class ValueValidationResult(BaseModel):
    """Per-page outcome of checking field VALUES against business-shape rules.

    valid_ratio is Optional and is None — not 0.0, not 1.0 — when nothing was
    checked. A page with no checkable fields has no validity to report, and
    collapsing that to a number would either fabricate a failure or fabricate
    a pass.
    """

    checked_fields: List[str] = Field(default_factory=list)
    valid_fields: List[str] = Field(default_factory=list)
    invalid_fields: List[str] = Field(default_factory=list)
    # No rule configured. Explicitly NOT invalid, and excluded from the ratio
    # denominator — "we never wrote a rule for this" must not read as
    # "this value is wrong".
    unknown_fields: List[str] = Field(default_factory=list)
    invalid_details: List[InvalidFieldDetail] = Field(default_factory=list)
    valid_ratio: Optional[float] = None


class FieldEvidence(BaseModel):
    """One candidate value for one field, from one source, kept verbatim.

    Both sources' values survive here. Nothing in this pipeline overwrites an
    OCR reading with a VLM reading or repairs a value from a regex — a
    disagreement is recorded as a disagreement and left for a human or a later
    adjudication step that this round deliberately does not implement.
    """

    entity_id: Optional[str] = None
    occurrence_id: str           # unique per candidate; two readings of the
                                 # same field from the same source stay distinct
    field_name: str
    raw_value: str
    normalized_value: str
    source: Literal["ocr", "vlm"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_status: Literal["valid", "invalid", "unknown"] = "unknown"
    bbox: List[float] = Field(default_factory=list)
    entity_assignment_uncertain: bool = False
    # Post-alignment handle. None for page-level fields (drawing number, title)
    # which belong to no entity. Conflict grouping keys on this, never on
    # entity_id — see src/vision/entity_ref.py.
    entity_ref: Optional[str] = None
    entity_type: Optional[str] = None


class FieldConflict(BaseModel):
    """Same (entity_ref, field_name), different normalized values.

    resolution is fixed at "unresolved" this round. Recording the conflict is
    the deliverable; picking a winner would need an adjudication policy that
    has not been designed or measured, and guessing one silently is exactly
    the failure mode this module exists to expose.
    """

    entity_ref: Optional[str] = None
    entity_id: Optional[str] = None      # kept as an observation, not the key
    entity_type: Optional[str] = None
    field_name: str
    ocr_value: Optional[str] = None
    vlm_value: Optional[str] = None
    ocr_validation: Optional[str] = None
    vlm_validation: Optional[str] = None
    resolution: Literal["unresolved"] = "unresolved"


class FieldEvidenceSet(BaseModel):
    evidence: List[FieldEvidence] = Field(default_factory=list)
    conflicts: List[FieldConflict] = Field(default_factory=list)
    entity_assignment_uncertain_count: int = 0
    alignments: List["EntityAlignment"] = Field(default_factory=list)

    def by_field(self) -> Dict[str, List[FieldEvidence]]:
        grouped: Dict[str, List[FieldEvidence]] = {}
        for item in self.evidence:
            grouped.setdefault(f"{item.entity_id or ''}|{item.field_name}", []).append(item)
        return grouped


# ---------------------------------------------------------------------------
# Entity typing and cross-source alignment (round 4)
# ---------------------------------------------------------------------------

class SourceEntity(BaseModel):
    """One entity as ONE source sees it, before any cross-source matching.

    identifier is what that source read (a motor number, a device id). It is
    kept as an observation and used as ONE alignment signal — never as the key
    that groups readings together, because a misread identifier would then
    split one entity into two and hide the very disagreement being looked for.
    """

    source: Literal["ocr", "vlm"]
    source_key: str            # stable handle for this entity within its source
    entity_type: str
    raw_entity_name: str = ""          # what the source called it, verbatim
    extracted_entity_label: str = ""   # label pulled out by a regex mapping rule
    mapping_method: Literal["exact", "regex", "unmapped", "field_rule"] = "unmapped"
    identifier: Optional[str] = None
    normalized_identifier: Optional[str] = None
    fields: Dict[str, str] = Field(default_factory=dict)  # canonical field -> normalized value
    position: Optional[List[float]] = None  # bbox when the source provides one


class EntityAlignment(BaseModel):
    """The result of trying to decide that an OCR entity and a VLM entity are
    the same physical thing.

    entity_ref is the post-alignment handle everything downstream keys on. The
    ordinal in it labels an already-decided pairing; it is never the mechanism
    that creates one, because ordinal position shifts the moment either source
    misses an entity.
    """

    # None unless the two sources were actually matched. An unaligned entity
    # must not carry a handle that downstream code could mistake for a
    # confirmed cross-source identity.
    entity_ref: Optional[str] = None
    entity_type: str
    status: Literal[
        "aligned_singleton",
        "aligned_exact_identifier",
        "aligned_spatially",
        "aligned_contextually",
        # Only one source described an entity of this type at all — there was
        # nothing to align against. Kept distinct from `unresolved`, which
        # means both sources offered candidates and no method could match
        # them; collapsing the two would make an alignment failure rate that
        # is mostly "the VLM was never called on this page".
        "single_source",
        "unresolved",
    ]
    method: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_entity_id: Optional[str] = None
    vlm_entity_id: Optional[str] = None
    ocr_source_key: Optional[str] = None
    vlm_source_key: Optional[str] = None
    # Why only one source described this entity. "single_source" on its own
    # lumps together three quite different situations — a page where fallback
    # never ran, a component OCR could not see, and one the VLM did not
    # report — and a count that mixes them measures nothing in particular.
    single_source_cause: Optional[Literal[
        "vlm_not_invoked", "ocr_side_absent", "vlm_side_absent",
    ]] = None
    reasons: List[str] = Field(default_factory=list)
