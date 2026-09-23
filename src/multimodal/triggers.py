"""Deciding which fields get a second look.

The gate is `cfg.eligible_reasons`. A field whose evidence is present, valid,
unambiguous and confident matches none of them and is never sent anywhere,
which is what keeps this from degenerating into "review everything" — the
state the audit found the table round in, where 60 of 114 cells had a request
and 57 of those came from one known false-positive detector.

One target per field, not one per reason. A cell that is both low-confidence
and ambiguously assigned is a single question about a single region; asking it
twice would spend two calls to learn one thing.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from src.multimodal.config import MultimodalReviewConfig
from src.multimodal.reasons import (
    CELL_ASSIGNMENT_UNCERTAIN, INVALID_FIELD_VALUE,
    ISOLATED_LABEL_WITHOUT_VALUE, LOW_OCR_CONFIDENCE_ON_REQUIRED_FIELD,
    OCR_VLM_CONFLICT, REQUIRED_FIELD_MISSING, WATERMARK_OVERLAP, validate,
)
from src.multimodal.schemas import ReviewTarget
from src.vision.schemas import FieldPair


def _request_id(document_id: str, field_name: str, entity_id: Optional[str]) -> str:
    scope = f"{entity_id}." if entity_id else ""
    return f"review:{document_id}:{scope}{field_name}"


class TriggerCollector:
    """Accumulates reasons per field, then emits one target each."""

    def __init__(self, document_id: str, *, drawing_type: Optional[str] = None,
                 page: int = 1):
        self.document_id = document_id
        self.drawing_type = drawing_type
        self.page = page
        self._targets: Dict[str, ReviewTarget] = {}

    def add(
        self,
        field_name: str,
        reason: str,
        *,
        entity_id: Optional[str] = None,
        canonical_field_name: Optional[str] = None,
        ocr_value: Optional[str] = None,
        ocr_confidence: float = 0.0,
        label_bbox: Optional[Sequence[float]] = None,
        value_bbox: Optional[Sequence[float]] = None,
        cell_bbox: Optional[Sequence[float]] = None,
        row_bbox: Optional[Sequence[float]] = None,
        chunk_id: Optional[str] = None,
    ) -> ReviewTarget:
        validate([reason])
        key = f"{entity_id or ''}|{field_name}"
        target = self._targets.get(key)
        if target is None:
            target = ReviewTarget(
                request_id=_request_id(self.document_id, field_name, entity_id),
                document_id=self.document_id,
                drawing_type=self.drawing_type,
                page=self.page,
                chunk_id=chunk_id,
                field_name=field_name,
                canonical_field_name=canonical_field_name or field_name,
                review_reasons=[],
                existing_ocr_value=ocr_value,
                existing_ocr_confidence=ocr_confidence,
                entity_id=entity_id,
            )
            self._targets[key] = target

        if reason not in target.review_reasons:
            target.review_reasons.append(reason)
        # Later calls fill in geometry the first one did not have; they never
        # overwrite a box that is already known.
        for attr, value in (("label_bbox", label_bbox), ("value_bbox", value_bbox),
                            ("cell_bbox", cell_bbox), ("row_bbox", row_bbox)):
            if value and not getattr(target, attr):
                setattr(target, attr, [float(v) for v in value])
        if ocr_value and not target.existing_ocr_value:
            target.existing_ocr_value = ocr_value
        if chunk_id and not target.chunk_id:
            target.chunk_id = chunk_id
        return target

    def targets(self, cfg: MultimodalReviewConfig) -> List[ReviewTarget]:
        """Eligible targets, most important reason first, then by priority.

        Sorting matters because `max_live_calls` truncates this list. Without
        it the budget goes to whatever the iteration order happened to produce
        — in this repo, 57 ambiguous-cell requests before the first missing
        required field.
        """
        out = []
        for target in self._targets.values():
            eligible = [r for r in target.review_reasons
                        if r in cfg.eligible_reasons]
            if not eligible:
                continue
            target.review_reasons = sorted(eligible, key=cfg.priority_of)
            out.append(target)
        return sorted(out, key=lambda t: (cfg.priority_of(t.priority_reason),
                                          t.field_name))


def collect_targets(
    *,
    document_id: str,
    drawing_type: Optional[str],
    cfg: MultimodalReviewConfig,
    pairs: Sequence[FieldPair] = (),
    required_fields: Iterable[str] = (),
    invalid_field_labels: Iterable[str] = (),
    isolated_labels: Sequence[dict] = (),
    conflicted_fields: Iterable[str] = (),
    watermark_fields: Iterable[str] = (),
    uncertain_cells: Sequence[dict] = (),
    page: int = 1,
) -> List[ReviewTarget]:
    """Build every eligible target for one page from what the parse knows.

    Each argument corresponds to one admissible reason and comes from a
    detector that already exists; nothing here re-derives whether a field is
    missing or invalid, it only decides that the answer warrants a look.
    """
    collector = TriggerCollector(document_id, drawing_type=drawing_type, page=page)
    by_field = {p.field_name: p for p in pairs}

    for name in required_fields:
        pair = by_field.get(name)
        if pair is None or not pair.raw_value:
            collector.add(name, REQUIRED_FIELD_MISSING)
        elif pair.confidence and pair.confidence < cfg.low_confidence_threshold:
            collector.add(name, LOW_OCR_CONFIDENCE_ON_REQUIRED_FIELD,
                          ocr_value=pair.raw_value, ocr_confidence=pair.confidence,
                          value_bbox=pair.bbox)

    for label in invalid_field_labels:
        entity_id, _, field_name = label.rpartition(".")
        pair = by_field.get(field_name)
        collector.add(field_name, INVALID_FIELD_VALUE,
                      entity_id=entity_id or None,
                      ocr_value=pair.raw_value if pair else None,
                      value_bbox=pair.bbox if pair else None)

    for item in isolated_labels:
        collector.add(item["field_name"], ISOLATED_LABEL_WITHOUT_VALUE,
                      label_bbox=item.get("label_bbox"),
                      entity_id=item.get("entity_id"))

    for label in conflicted_fields:
        entity_id, _, field_name = label.rpartition(".")
        pair = by_field.get(field_name)
        collector.add(field_name, OCR_VLM_CONFLICT, entity_id=entity_id or None,
                      ocr_value=pair.raw_value if pair else None,
                      value_bbox=pair.bbox if pair else None)

    for name in watermark_fields:
        pair = by_field.get(name)
        collector.add(name, WATERMARK_OVERLAP,
                      value_bbox=pair.bbox if pair else None)

    for item in uncertain_cells:
        collector.add(item["field_name"], CELL_ASSIGNMENT_UNCERTAIN,
                      entity_id=item.get("entity_id"),
                      ocr_value=item.get("text"),
                      cell_bbox=item.get("cell_bbox"),
                      row_bbox=item.get("row_bbox"),
                      chunk_id=item.get("chunk_id"))

    return collector.targets(cfg)


def trigger_stats(targets: Sequence[ReviewTarget], *,
                  total_chunks: int, critical_fields: int) -> dict:
    counts: Dict[str, int] = {}
    for target in targets:
        for reason in target.review_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "total_chunks": total_chunks,
        "critical_fields": critical_fields,
        "triggered_fields": len(targets),
        "trigger_rate": round(len(targets) / critical_fields, 4)
        if critical_fields else None,
        "by_reason": dict(sorted(counts.items())),
    }
