"""Merges OCR and VLM readings into field-level evidence, and records where
they disagree.

What this replaces: the evaluation used to union the two sources by
concatenating their text and asking whether a gold string appeared anywhere in
the result. Under that scoring a page where OCR read a cabinet number as stamp
text and the VLM read it correctly is indistinguishable from a page where both
read it correctly — the gold substring is present either way. The disagreement,
which is the whole signal, is invisible.

Here each reading becomes a candidate with a source, a validation status and
its original text. Two candidates for the same field with different values
produce a conflict, and a conflict is left unresolved: this round records
disagreement, it does not adjudicate it. Choosing a winner needs a policy whose
error rate somebody has measured, and inventing one silently is the failure
this module exists to surface.

Conflicts key on (entity_ref, canonical field name). entity_ref comes out of
the alignment layer and depends on neither source's reading, while the
recognized entity_id is kept alongside as an observation. Keying on the
identifier itself would let a misread code split one entity into two and bury
the disagreement; keying on the field name alone would make every multi-device
drawing look like one permanent contradiction.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.vision.config import ValueValidationConfig, VLMFieldMapping
from src.vision.schemas import (
    EntityAlignment, FieldConflict, FieldEvidence, FieldEvidenceSet, FieldPair,
    ValueValidationResult,
)
from src.vision.value_validation import normalize_value, validity_status


def _normalize(value: str, cfg: ValueValidationConfig) -> str:
    return normalize_value(value, cfg.hyphen_variants, cfg.collapse_spaces_around_hyphen)


def _uncertain(source: str, source_key: Optional[str], entity_ref: Optional[str],
               certain_keys: Optional[set]) -> bool:
    """Whether this reading's entity attribution should be treated as unknown.

    Falls back to "aligned or not" when no certainty index was supplied, so a
    caller that has not run the alignment layer is not silently told every
    attribution is fine.
    """
    if not source_key:
        return False  # page-level field: belongs to the drawing, not an entity
    if certain_keys is None:
        return entity_ref is None
    return (source, source_key) not in certain_keys


def _qualified(entity_id: Optional[str], field_name: str) -> str:
    return f"{entity_id}{field_name}" if entity_id else field_name


def evidence_from_ocr(
    pairs: Sequence[FieldPair],
    validation: ValueValidationResult,
    cfg: ValueValidationConfig,
    pair_source_keys: Optional[Dict[int, str]] = None,
    entity_ref_index: Optional[Dict[Tuple[str, str], str]] = None,
    certain_keys: Optional[set] = None,
) -> List[FieldEvidence]:
    """One candidate per recovered OCR pair, carrying its validation verdict.

    An invalid value is kept, not dropped. It is the primary record of what the
    page actually shows, and discarding it would erase the evidence that
    anything went wrong.
    """
    pair_source_keys = pair_source_keys or {}
    entity_ref_index = entity_ref_index or {}
    certain_keys = certain_keys if certain_keys is not None else None
    out: List[FieldEvidence] = []
    for index, pair in enumerate(pairs):
        label = _qualified(pair.entity_id, pair.field_name)
        source_key = pair_source_keys.get(index)
        entity_ref = entity_ref_index.get(("ocr", source_key)) if source_key else None
        out.append(FieldEvidence(
            entity_ref=entity_ref,
            entity_type=source_key.split(":")[1] if source_key else None,
            entity_id=pair.entity_id,
            occurrence_id=f"ocr:{index}:{label}",
            field_name=pair.field_name,
            raw_value=pair.raw_value,
            normalized_value=_normalize(pair.raw_value, cfg),
            source="ocr",
            confidence=pair.confidence,
            validation_status=validity_status(validation, label),
            bbox=list(pair.bbox),
            entity_assignment_uncertain=_uncertain("ocr", source_key, entity_ref, certain_keys),
        ))
    return out


def evidence_from_vlm_entities(
    metadata_json: str,
    entities: Sequence,
    mapping: VLMFieldMapping,
    cfg: ValueValidationConfig,
    entity_ref_index: Optional[Dict[Tuple[str, str], str]] = None,
    certain_keys: Optional[set] = None,
) -> List[FieldEvidence]:
    """Candidates from a VLM response, stamped with their aligned entity_ref.

    Component-level readings are no longer flattened away. Round 3 suppressed
    them to stop a motor and a breaker being filed under one page-level field,
    and the audit of the legacy/structured gap showed the bill for that: on
    single-device drawings the VLM's component identifiers became unreachable
    as structured candidates, and that accounted for two of the three fields in
    the gap. Typing each device and aligning it first removes the need for the
    suppression — the motor's reading lands on the motor, the breaker's on the
    breaker.

    A device whose name the controlled mapping does not recognize keeps
    entity_type "unknown" and fails to align. Its readings are still recorded,
    with no entity_ref, rather than being attached to a guess.
    """
    entity_ref_index = entity_ref_index or {}
    out: List[FieldEvidence] = []

    metadata = _safe_json(metadata_json)
    for vlm_key, field_name in mapping.metadata.items():
        value = metadata.get(vlm_key)
        if value in (None, ""):
            continue
        out.append(FieldEvidence(
            entity_ref=None,  # page-level: belongs to the drawing, not an entity
            entity_type=None,
            entity_id=None,
            occurrence_id=f"vlm:meta:{field_name}",
            field_name=field_name,
            raw_value=str(value),
            normalized_value=_normalize(str(value), cfg),
            source="vlm",
            confidence=0.0,  # this gateway returns no per-field confidence
            validation_status="unknown",
        ))

    for entity in entities:
        entity_ref = entity_ref_index.get(("vlm", entity.source_key))
        for field_name, value in entity.fields.items():
            if not value:
                continue
            out.append(FieldEvidence(
                entity_ref=entity_ref,
                entity_type=entity.entity_type,
                entity_id=entity.identifier,
                occurrence_id=f"{entity.source_key}:{field_name}",
                field_name=field_name,
                raw_value=value,
                normalized_value=value,  # normalized when the entity was built
                source="vlm", confidence=0.0, validation_status="unknown",
                entity_assignment_uncertain=_uncertain(
                    "vlm", entity.source_key, entity_ref, certain_keys),
            ))

    return out


def _safe_json(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {} if default is None else default


def merge_evidence(
    ocr_evidence: Sequence[FieldEvidence],
    vlm_evidence: Sequence[FieldEvidence],
    alignments: Optional[Sequence[EntityAlignment]] = None,
) -> FieldEvidenceSet:
    """Combine both sources, recording — never resolving — disagreements.

    A conflict is raised when the two sources both spoke about the same
    (entity_ref, field_name) and their normalized values differ. Evidence whose
    entity could not be aligned carries no entity_ref and is never compared: a
    disagreement between two readings that may not even describe the same piece
    of equipment is not evidence that either was misread. Comparison uses
    the normalized form so that pure formatting differences (a full-width
    hyphen, spacing around a dash) do not masquerade as factual disagreement;
    the raw values of both sides stay on the record either way.
    """
    all_evidence = list(ocr_evidence) + list(vlm_evidence)

    grouped: Dict[Tuple[Optional[str], str], Dict[str, List[FieldEvidence]]] = {}
    for item in all_evidence:
        if item.entity_type is not None and item.entity_ref is None:
            continue  # entity-scoped but unaligned: not comparable
        key = (item.entity_ref, item.field_name)
        grouped.setdefault(key, {"ocr": [], "vlm": []})[item.source].append(item)

    conflicts: List[FieldConflict] = []
    for (entity_ref, field_name), sides in sorted(
        grouped.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])
    ):
        ocr_side, vlm_side = sides["ocr"], sides["vlm"]
        if not ocr_side or not vlm_side:
            continue  # only one source spoke; nothing to disagree about
        ocr_values = {e.normalized_value for e in ocr_side}
        vlm_values = {e.normalized_value for e in vlm_side}
        if ocr_values & vlm_values:
            continue  # at least one reading agrees
        conflicts.append(FieldConflict(
            entity_ref=entity_ref,
            entity_id=ocr_side[0].entity_id,
            entity_type=ocr_side[0].entity_type,
            field_name=field_name,
            ocr_value=ocr_side[0].raw_value,
            vlm_value=vlm_side[0].raw_value,
            ocr_validation=ocr_side[0].validation_status,
            vlm_validation=vlm_side[0].validation_status,
        ))

    return FieldEvidenceSet(
        evidence=all_evidence,
        conflicts=conflicts,
        entity_assignment_uncertain_count=sum(
            1 for e in all_evidence if e.entity_assignment_uncertain),
        alignments=list(alignments or []),
    )


def conflicted_keys(evidence_set: FieldEvidenceSet) -> set:
    return {(c.entity_ref, c.field_name) for c in evidence_set.conflicts}
