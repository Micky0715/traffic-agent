"""Types for the crop -> VLM -> merge -> re-decide loop."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RegionScope = Literal["value", "label_right", "cell", "row", "unavailable"]
InferenceMode = Literal["disabled", "cache_replay", "live"]


class ReviewTarget(BaseModel):
    """One field worth a second look, with every reason that made it so.

    A field is one target no matter how many reasons fired — otherwise a cell
    that is both low-confidence and ambiguous would spend two VLM calls to ask
    the same question about the same pixels.
    """

    request_id: str
    document_id: str
    drawing_type: Optional[str] = None
    page: int = 1
    chunk_id: Optional[str] = None
    field_name: str
    canonical_field_name: str
    review_reasons: List[str] = Field(default_factory=list)
    existing_ocr_value: Optional[str] = None
    existing_ocr_confidence: float = 0.0
    label_bbox: List[float] = Field(default_factory=list)
    value_bbox: List[float] = Field(default_factory=list)
    cell_bbox: List[float] = Field(default_factory=list)
    row_bbox: List[float] = Field(default_factory=list)
    entity_id: Optional[str] = None

    @property
    def priority_reason(self) -> str:
        return self.review_reasons[0] if self.review_reasons else ""


class CropResult(BaseModel):
    """A saved crop, or a recorded refusal to make one.

    `available=False` is a first-class outcome. The alternative — sending the
    whole page when no region could be located — asks the model to find the
    field itself, which is the failure mode this entire stage exists to avoid.
    """

    available: bool
    region_scope: RegionScope
    reason: str = ""
    source_image_path: Optional[str] = None
    source_image_sha256: Optional[str] = None
    crop_path: Optional[str] = None
    crop_sha256: Optional[str] = None
    page: int = 1
    bbox_original: List[float] = Field(default_factory=list)
    bbox_with_padding: List[float] = Field(default_factory=list)
    width: int = 0
    height: int = 0
    chunk_id: Optional[str] = None
    field_name: str = ""
    review_reasons: List[str] = Field(default_factory=list)


class VisionFieldResult(BaseModel):
    """One field as the model reported it.

    `readable=False` with `raw_value=None` is a legitimate, expected answer.
    The prompt asks for it explicitly, because a model that must always produce
    a string will produce a plausible one.
    """

    canonical_field_name: str
    raw_name: Optional[str] = None
    raw_value: Optional[str] = None
    readable: bool = False
    evidence_bbox_in_crop: List[float] = Field(default_factory=list)
    notes: Optional[str] = None


class VisionReviewResult(BaseModel):
    request_id: str
    status: Literal["success", "schema_failure", "unreadable",
                    "crop_unavailable", "not_invoked", "call_failed",
                    "budget_exhausted"] = "not_invoked"
    inference_mode: InferenceMode = "disabled"
    fields: List[VisionFieldResult] = Field(default_factory=list)
    unreadable_reasons: List[str] = Field(default_factory=list)
    # Provenance. Never inferred — copied from the call or the cache entry.
    provider: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: Optional[str] = None
    started_at: Optional[str] = None
    latency_ms: Optional[float] = None
    input_image_sha256: Optional[str] = None
    crop_sha256: Optional[str] = None
    error_type: Optional[str] = None
    raw_response: Optional[str] = None

    @property
    def is_live(self) -> bool:
        return self.inference_mode == "live"


class ReviewOutcome(BaseModel):
    """What the whole loop did for one document."""

    document_id: str
    decision_before_review: str
    decision_after_review: str
    fields_recovered: List[str] = Field(default_factory=list)
    conflicts_created: List[str] = Field(default_factory=list)
    conflicts_resolved: List[str] = Field(default_factory=list)
    still_missing_fields: List[str] = Field(default_factory=list)
    single_source_vlm_fields: List[str] = Field(default_factory=list)
    review_results: List[VisionReviewResult] = Field(default_factory=list)
    notes: Dict[str, Any] = Field(default_factory=dict)
