"""Deciding when a region deserves a second look from a vision model.

This round builds the request and stops. No model is called, and the status
says so: `requested_not_invoked` means the system judged a second look
warranted and did not take it. Writing that up as a successful review would
turn an open question into a fabricated confirmation — the one error that
cannot be detected downstream, because a confirmed value looks exactly like a
verified one.

The four statuses are kept distinct for the same reason a cache hit is not a
live call: `cache_replay` is real content from a recorded run, `real_inference`
is this run, and they support different claims.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.tables.schemas import (
    ImageRegionRef, LocalVisionReviewRequest, ParsedTable, deterministic_id,
)

# Why a region might warrant another look. Each is a condition the current
# parse can detect about itself, not a guess about correctness.
REVIEW_REASONS = {
    "HEADER_ATTRIBUTION_UNCERTAIN": "a column's header path could not be settled",
    "CELL_SPANS_COLUMNS": "an OCR block covered more than one cell",
    "CRITICAL_VALUE_INVALID": "a high-risk field failed its shape rule",
    "LABEL_VALUE_PAIRING_FAILED": "a label was read with no value beside it",
    "OCR_VLM_CONFLICT": "two sources disagree about the same field",
    "WATERMARK_OVER_FIELD": "repeated boilerplate covers the value area",
    "LOW_CONFIDENCE_CRITICAL_FIELD": "a high-risk field was read with low confidence",
}


def build_review_request(
    *,
    reason: str,
    chunk_id: Optional[str] = None,
    region: Optional[ImageRegionRef] = None,
    expected_fields: Optional[Sequence[str]] = None,
    existing_ocr_values: Optional[Dict[str, str]] = None,
) -> LocalVisionReviewRequest:
    if reason not in REVIEW_REASONS:
        raise ValueError(f"unknown review reason {reason!r}")
    return LocalVisionReviewRequest(
        request_id=f"review:{deterministic_id(chunk_id or '', reason, region.cell_id if region else '')}",
        chunk_id=chunk_id,
        image_region_ref=region,
        expected_fields=list(expected_fields or []),
        existing_ocr_values=dict(existing_ocr_values or {}),
        review_reason=reason,
        # Built, not run. See the module docstring.
        status="requested_not_invoked",
    )


def collect_review_requests(
    table: ParsedTable,
    *,
    high_risk_fields: Sequence[str],
    header_paths: Optional[Dict[int, List[str]]] = None,
    region_refs: Optional[Dict[str, ImageRegionRef]] = None,
    min_confidence: float = 0.6,
) -> List[LocalVisionReviewRequest]:
    """Every region in this table that warrants a second look.

    Deliberately returns an empty list when nothing does — an always-nonempty
    review queue is a queue nobody works through.
    """
    header_paths = header_paths or {}
    region_refs = region_refs or {}
    requests: List[LocalVisionReviewRequest] = []

    for cell in table.cells:
        if cell.is_header:
            continue
        field_name = ".".join(header_paths.get(cell.col_start, [])) or ""
        is_high_risk = any(field_name.endswith(f) or f in cell.text_normalized
                           for f in high_risk_fields)
        region = region_refs.get(cell.cell_id)

        if cell.assignment_uncertain:
            requests.append(build_review_request(
                reason="CELL_SPANS_COLUMNS", region=region,
                expected_fields=[field_name] if field_name else [],
                existing_ocr_values={field_name: cell.text_raw} if field_name else {}))
        elif cell.merge_status == "uncertain":
            requests.append(build_review_request(
                reason="HEADER_ATTRIBUTION_UNCERTAIN", region=region,
                expected_fields=[field_name] if field_name else {}))
        elif is_high_risk and cell.text_normalized and \
                cell.confidence < min_confidence:
            requests.append(build_review_request(
                reason="LOW_CONFIDENCE_CRITICAL_FIELD", region=region,
                expected_fields=[field_name] if field_name else [],
                existing_ocr_values={field_name: cell.text_raw} if field_name else {}))
    return requests


def review_status_summary(requests: Sequence[LocalVisionReviewRequest]) -> Dict[str, int]:
    summary = {"not_required": 0, "requested_not_invoked": 0,
               "cache_replay": 0, "real_inference": 0}
    for request in requests:
        summary[request.status] = summary.get(request.status, 0) + 1
    return summary
