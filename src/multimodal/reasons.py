"""Why a field may be sent to a vision model.

These eight are the ONLY admissible reasons. A field with complete, valid,
unambiguous evidence has no entry here, which is what stops the review queue
from becoming "every field, every time".

src/tables/review.py already had its own seven-name vocabulary from the table
round. It is not renamed — it is frozen, and other code reads it — so this
module maps between the two instead. The mapping is explicit and total in one
direction: every legacy reason lands on exactly one reason here.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

REQUIRED_FIELD_MISSING = "required_field_missing"
INVALID_FIELD_VALUE = "invalid_field_value"
LABEL_VALUE_PAIRING_FAILED = "label_value_pairing_failed"
ISOLATED_LABEL_WITHOUT_VALUE = "isolated_label_without_value"
OCR_VLM_CONFLICT = "ocr_vlm_conflict"
CELL_ASSIGNMENT_UNCERTAIN = "cell_assignment_uncertain"
WATERMARK_OVERLAP = "watermark_overlap"
LOW_OCR_CONFIDENCE_ON_REQUIRED_FIELD = "low_ocr_confidence_on_required_field"

ALL_REASONS: List[str] = [
    REQUIRED_FIELD_MISSING,
    INVALID_FIELD_VALUE,
    LABEL_VALUE_PAIRING_FAILED,
    ISOLATED_LABEL_WITHOUT_VALUE,
    OCR_VLM_CONFLICT,
    CELL_ASSIGNMENT_UNCERTAIN,
    WATERMARK_OVERLAP,
    LOW_OCR_CONFIDENCE_ON_REQUIRED_FIELD,
]

REASON_MEANING: Dict[str, str] = {
    REQUIRED_FIELD_MISSING:
        "a field this drawing type requires produced no value at all",
    INVALID_FIELD_VALUE:
        "a value was read but failed its shape rule",
    LABEL_VALUE_PAIRING_FAILED:
        "a label and a value were both read but could not be paired",
    ISOLATED_LABEL_WITHOUT_VALUE:
        "a label was read with nothing beside it",
    OCR_VLM_CONFLICT:
        "two sources disagree about the same field",
    CELL_ASSIGNMENT_UNCERTAIN:
        "an OCR block could not be placed in one cell",
    WATERMARK_OVERLAP:
        "repeated boilerplate covers the value area",
    LOW_OCR_CONFIDENCE_ON_REQUIRED_FIELD:
        "a required field was read below the confidence threshold",
}

# src/tables/review.py REVIEW_REASONS -> the vocabulary above.
LEGACY_REASON_MAP: Dict[str, str] = {
    "HEADER_ATTRIBUTION_UNCERTAIN": CELL_ASSIGNMENT_UNCERTAIN,
    "CELL_SPANS_COLUMNS": CELL_ASSIGNMENT_UNCERTAIN,
    "CRITICAL_VALUE_INVALID": INVALID_FIELD_VALUE,
    "LABEL_VALUE_PAIRING_FAILED": LABEL_VALUE_PAIRING_FAILED,
    "OCR_VLM_CONFLICT": OCR_VLM_CONFLICT,
    "WATERMARK_OVER_FIELD": WATERMARK_OVERLAP,
    "LOW_CONFIDENCE_CRITICAL_FIELD": LOW_OCR_CONFIDENCE_ON_REQUIRED_FIELD,
}


def from_legacy(reason: str) -> str:
    """Translate a table-round reason. Unknown input raises rather than passing
    through: a reason that silently survives unmapped would evade the eligible
    list and buy a VLM call nobody authorised."""
    if reason not in LEGACY_REASON_MAP:
        raise ValueError(f"unmapped legacy review reason {reason!r}")
    return LEGACY_REASON_MAP[reason]


def validate(reasons: Sequence[str]) -> List[str]:
    unknown = [r for r in reasons if r not in ALL_REASONS]
    if unknown:
        raise ValueError(f"unknown review reason(s): {unknown}")
    return list(reasons)
