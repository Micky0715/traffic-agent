"""VLM replies -> FieldEvidence -> merged with OCR.

Two rules carry this module.

First, a VLM value is validated by exactly the same `validate_field_values`
that OCR values go through. Skipping it "because the model is smarter" would
mean the only values in the system that never had their shape checked are the
ones no human ever saw.

Second, merging is delegated to the existing `src/vision/evidence.merge_evidence`
rather than reimplemented. It already records a disagreement without resolving
it, treats a normalization-only difference as agreement, and leaves a
single-source reading uncontested. A second merge function would drift from it,
and the drift would show up as conflicts silently appearing or vanishing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from src.multimodal.schemas import CropResult, ReviewTarget, VisionReviewResult
from src.vision.evidence import merge_evidence
from src.vision.config import ValueValidationConfig
from src.vision.schemas import FieldEvidence, FieldEvidenceSet, FieldPair
from src.vision.value_validation import (
    normalize_value, validate_field_values, validity_status,
)


def _bbox_to_page(crop: Optional[CropResult],
                  bbox_in_crop: Sequence[float]) -> List[float]:
    """Map a crop-relative box back to page coordinates.

    The model is asked for coordinates inside the crop precisely because it
    cannot know the page offset. Adding the crop origin back here is the only
    place that knows it; letting a crop-relative box travel onward as if it
    were a page box would point a reviewer at the wrong part of the drawing.
    """
    if not crop or not crop.available or not bbox_in_crop or len(bbox_in_crop) != 4:
        return list(crop.bbox_original) if crop and crop.available else []
    ox, oy = crop.bbox_with_padding[0], crop.bbox_with_padding[1]
    x0, y0, x1, y1 = bbox_in_crop
    return [ox + float(x0), oy + float(y0), ox + float(x1), oy + float(y1)]


def evidence_from_review(
    results: Sequence[Tuple[ReviewTarget, VisionReviewResult, Optional[CropResult]]],
    cfg: ValueValidationConfig,
    drawing_type_spec=None,
) -> List[FieldEvidence]:
    """Turn readable VLM fields into validated evidence.

    An unreadable field produces NO evidence. That is the point of allowing the
    model to say so — a `readable=false` turned into an empty-string value
    would be indistinguishable downstream from a field that was read as blank.
    """
    pairs: List[FieldPair] = []
    provenance: List[Tuple[ReviewTarget, VisionReviewResult, Optional[CropResult], object]] = []

    for target, result, crop in results:
        if result.status != "success":
            continue
        for item in result.fields:
            if not item.readable or not item.raw_value:
                continue
            pairs.append(FieldPair(
                entity_id=target.entity_id,
                field_name=item.canonical_field_name,
                raw_value=item.raw_value,
                bbox=_bbox_to_page(crop, item.evidence_bbox_in_crop),
                # A model's self-reported confidence is not a business
                # confidence and this pipeline never had one from the VLM.
                # 0.0 records "no measured confidence", not "no trust".
                confidence=0.0,
            ))
            provenance.append((target, result, crop, item))

    if not pairs:
        return []

    validation = validate_field_values(pairs, cfg, drawing_type_spec)

    out: List[FieldEvidence] = []
    for index, (pair, (target, result, crop, item)) in enumerate(zip(pairs, provenance)):
        label = f"{pair.entity_id}.{pair.field_name}" if pair.entity_id else pair.field_name
        out.append(FieldEvidence(
            entity_id=pair.entity_id,
            occurrence_id=f"vlm:{index}:{label}",
            field_name=pair.field_name,
            raw_value=pair.raw_value,
            normalized_value=normalize_value(
                pair.raw_value, cfg.hyphen_variants,
                cfg.collapse_spaces_around_hyphen),
            source="vlm",
            confidence=0.0,
            validation_status=validity_status(validation, label),
            bbox=list(pair.bbox),
            # A value read from a crop of one row cannot be re-attributed to a
            # device by this stage; whoever built the target said which device
            # the region belonged to, and nothing here second-guesses it.
            entity_assignment_uncertain=target.entity_id is None
            and target.review_reasons != [],
        ))
    return out


def fuse(
    ocr_evidence: Sequence[FieldEvidence],
    vlm_evidence: Sequence[FieldEvidence],
    alignments=None,
) -> FieldEvidenceSet:
    """Single entry point. Delegates to the existing merge."""
    return merge_evidence(ocr_evidence, vlm_evidence, alignments)


def single_source_fields(evidence_set: FieldEvidenceSet) -> Dict[str, List[str]]:
    """Which fields only one source ever spoke about."""
    by_key: Dict[Tuple[Optional[str], str], set] = {}
    for item in evidence_set.evidence:
        by_key.setdefault((item.entity_id, item.field_name), set()).add(item.source)
    out: Dict[str, List[str]] = {"ocr_only": [], "vlm_only": []}
    for (entity_id, field_name), sources in sorted(
            by_key.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        label = f"{entity_id}.{field_name}" if entity_id else field_name
        if sources == {"vlm"}:
            out["vlm_only"].append(label)
        elif sources == {"ocr"}:
            out["ocr_only"].append(label)
    return out
