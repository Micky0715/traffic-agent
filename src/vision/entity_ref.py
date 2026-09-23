"""Entity typing and cross-source entity alignment.

Why this layer exists: conflict detection used the recognized equipment code as
its grouping key. That lets the error being looked for decide how readings are
grouped — OCR reading M-19 and a VLM reading M-I9 become two unrelated
entities, the disagreement is filed as two separate facts, and nothing is
reported. An identifier that may itself be misread cannot also be the primary
key.

So an entity gets a reference that does not depend on what either source read:
its TYPE (motor, breaker, cabinet, …) plus an ordinal assigned after the two
sources' entities have been aligned. Alignment is attempted in descending order
of how much it can be trusted, and when none of the methods applies the pair is
left `unresolved` rather than matched by position and hope.

Nothing here maps one canonical field onto another. 控制柜编号 and 控制柜型号
belong to the same cabinet entity and stay two distinct fields; a cabinet and a
controller stay two distinct entity types.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.vision.config import EntityTypeConfig
from src.vision.schemas import EntityAlignment, SourceEntity

UNKNOWN_TYPE = "unknown"

# Reserved field name for an identifier read from a device whose type the
# controlled mapping did not recognize. Not a business field and not
# configurable: the alternative is dropping the reading entirely, and the
# metric-gap audit showed exactly what silently dropped readings cost. Filing
# it under a real field name instead would be a guess about what it means.
UNMAPPED_IDENTIFIER_FIELD = "unmapped_device_id"


# ---------------------------------------------------------------------------
# device_name -> entity_type
# ---------------------------------------------------------------------------

def map_entity_type(raw_name: str, cfg: EntityTypeConfig) -> Tuple[str, str, str]:
    """Returns (entity_type, extracted_entity_label, mapping_method).

    Two mechanisms, both configured, neither of them a substring test:

      exact    the whole name is a known type word
      regex    an anchored pattern with an explicit `label` group, for names
               that carry an identifier ("2A号水泵" -> pump, label "2A")

    A plain `contains` check is deliberately not offered. "控制柜" is a
    substring of "控制柜温控器", and "泵" appears in words that are not pumps;
    a containment rule silently mistypes entities, and a mistyped entity is
    then aligned against the wrong thing. Anything neither mechanism matches is
    `unknown` — never a guess.
    """
    name = (raw_name or "").strip()
    if not name:
        return UNKNOWN_TYPE, "", "unmapped"

    exact = cfg.exact.get(name)
    if exact:
        return exact, "", "exact"

    for rule in cfg.patterns:
        match = re.match(rule.pattern, name)
        if match:
            label = ""
            if "label" in (match.groupdict() or {}):
                label = match.group("label") or ""
            return rule.entity_type, label, "regex"

    return UNKNOWN_TYPE, "", "unmapped"


def entity_type_for_field(
    field_name: str, cfg: EntityTypeConfig, primary_entity_type: Optional[str]
) -> Optional[str]:
    """Which entity a FIELD belongs to on the OCR side.

    Returns None for page-level fields (drawing number, title, page). Fields
    the config does not name fall back to the drawing type's primary entity —
    on a fan group drawing, 功率 belongs to a fan — and to None when the
    drawing type declares no primary entity.
    """
    if field_name in cfg.page_level_fields:
        return None
    mapped = cfg.field_entity_type.get(field_name)
    if mapped:
        return mapped
    return primary_entity_type


def identity_field_for_type(entity_type: str, cfg: EntityTypeConfig) -> Optional[str]:
    return cfg.identity_field.get(entity_type)


# ---------------------------------------------------------------------------
# building one source's entities
# ---------------------------------------------------------------------------

def build_ocr_entities(
    pairs: Sequence, cfg: EntityTypeConfig, primary_entity_type: Optional[str],
    normalize,
) -> Tuple[List[SourceEntity], Dict[int, str], Dict[str, str]]:
    """Group OCR field/value pairs into entities.

    Returns (entities, pair_index -> source_key, page_level_field -> value).
    Page-level fields (drawing number, title, page) belong to no entity and are
    returned separately rather than being attached to an arbitrary one.
    """
    buckets: Dict[str, Dict] = {}
    pair_keys: Dict[int, str] = {}
    page_level: Dict[str, str] = {}

    for index, pair in enumerate(pairs):
        entity_type = entity_type_for_field(pair.field_name, cfg, primary_entity_type)
        if entity_type is None:
            page_level[pair.field_name] = normalize(pair.raw_value)
            continue
        # pair.entity_id is the per-device prefix on group drawings; it is an
        # OCR reading, used here only to keep one source's own devices apart,
        # never to match against the other source.
        source_key = f"ocr:{entity_type}:{pair.entity_id or 'page'}"
        bucket = buckets.setdefault(source_key, {
            "entity_type": entity_type, "entity_id": pair.entity_id,
            "fields": {}, "position": None,
        })
        bucket["fields"][pair.field_name] = normalize(pair.raw_value)
        if pair.bbox and bucket["position"] is None:
            bucket["position"] = list(pair.bbox)
        pair_keys[index] = source_key

    entities: List[SourceEntity] = []
    for source_key, bucket in buckets.items():
        identity_field = identity_field_for_type(bucket["entity_type"], cfg)
        identifier = bucket["fields"].get(identity_field) if identity_field else None
        entities.append(SourceEntity(
            source="ocr", source_key=source_key, entity_type=bucket["entity_type"],
            raw_entity_name="", extracted_entity_label=bucket["entity_id"] or "",
            mapping_method="field_rule",
            identifier=identifier, normalized_identifier=identifier,
            fields=bucket["fields"], position=bucket["position"],
        ))
    return entities, pair_keys, page_level


def build_vlm_entities(
    devices: Sequence[dict], cfg: EntityTypeConfig, known_field_names: Sequence[str],
    parameter_names: Dict[str, str], normalize,
) -> List[SourceEntity]:
    """Turn a VLM devices[] response into typed entities.

    device_name decides the type through the controlled mapping; device_id is
    recorded as this source's reading of the identifier, not as a key. A device
    whose name maps to nothing keeps entity_type "unknown" and will simply fail
    to align — which is the honest outcome, and visible in the report.
    """
    known = set(known_field_names)
    entities: List[SourceEntity] = []
    for index, device in enumerate(devices):
        if not isinstance(device, dict):
            continue
        raw_name = str(device.get("device_name") or "")
        entity_type, label, method = map_entity_type(raw_name, cfg)
        identifier = str(device.get("device_id") or "") or None

        fields: Dict[str, str] = {}
        for parameter in device.get("parameters") or []:
            if not isinstance(parameter, dict):
                continue
            field_name = parameter_names.get(parameter.get("name", ""))
            if not field_name:
                candidate = (parameter.get("raw_name") or "").strip()
                field_name = candidate if candidate in known else ""
            if not field_name:
                continue
            value = parameter.get("raw_text") or parameter.get("value")
            if value in (None, ""):
                continue
            fields[field_name] = normalize(str(value))

        # The device's own id, expressed as its type's identity field, so the
        # two sources describe the same thing in the same vocabulary.
        identity_field = identity_field_for_type(entity_type, cfg)
        if identifier and identity_field and identity_field not in fields:
            fields[identity_field] = normalize(identifier)
        elif identifier and not identity_field:
            # Type unknown, so which business field this identifier fills is
            # unknown too. Recorded under a reserved name rather than dropped
            # or guessed at; it will not align, and that shows in the report.
            fields[UNMAPPED_IDENTIFIER_FIELD] = normalize(identifier)

        entities.append(SourceEntity(
            source="vlm", source_key=f"vlm:dev{index}", entity_type=entity_type,
            raw_entity_name=raw_name, extracted_entity_label=label,
            mapping_method=method,
            identifier=identifier,
            normalized_identifier=normalize(identifier) if identifier else None,
            fields=fields, position=None,
        ))
    return entities


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

def _normalized_ids(entities: Sequence[SourceEntity]) -> Dict[str, List[SourceEntity]]:
    grouped: Dict[str, List[SourceEntity]] = {}
    for entity in entities:
        if entity.normalized_identifier:
            grouped.setdefault(entity.normalized_identifier, []).append(entity)
    return grouped


def _context_similarity(a: SourceEntity, b: SourceEntity) -> float:
    """Share of (field, value) observations the two entities agree on.

    Compared over the union, so an entity that merely has fewer fields does not
    score highly by default.
    """
    left = {(f, v) for f, v in a.fields.items() if v}
    right = {(f, v) for f, v in b.fields.items() if v}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def align_entities(
    ocr_entities: Sequence[SourceEntity],
    vlm_entities: Sequence[SourceEntity],
    cfg: EntityTypeConfig,
    vlm_invoked: bool = True,
) -> List[EntityAlignment]:
    """Pair OCR entities with VLM entities, one entity_type at a time.

    Method priority, strongest first:

      aligned_singleton        exactly one instance of the type on each side.
                               Nothing can be confused with anything else, and
                               it holds even when the two sources disagree
                               about the identifier — which is the case this
                               layer exists to expose.
      aligned_exact_identifier the normalized identifier is identical and
                               unique on both sides.
      aligned_spatially        positional correspondence. Requires positional
                               data from BOTH sources; the VLM adapter in this
                               repo returns no bounding boxes, so this branch
                               cannot fire here. It is kept because the status
                               is part of the contract, not because it runs.
      aligned_contextually     neighbouring field/value overlap at or above a
                               configured threshold.
      single_source            only one source described this type at all, so
                               there was nothing to align against. Distinct
                               from a failure to align — see the status enum.
      unresolved               both sources offered candidates and no method
                               matched them. Ordinal position is NOT used as a
                               last resort: if one source missed an entity
                               entirely, positions shift and every later entity
                               pairs with the wrong one.

    Entities that align reliably keep their alignment even when the page holds
    several of their type — only the instances that cannot be matched are left
    unresolved.
    """
    alignments: List[EntityAlignment] = []
    types = sorted({e.entity_type for e in list(ocr_entities) + list(vlm_entities)})

    for entity_type in types:
        ocr_side = [e for e in ocr_entities if e.entity_type == entity_type]
        vlm_side = [e for e in vlm_entities if e.entity_type == entity_type]
        matched_ocr: set = set()
        matched_vlm: set = set()
        pairs: List[Tuple[Optional[SourceEntity], Optional[SourceEntity], str, str, float, List[str]]] = []

        # 1. singleton
        if len(ocr_side) == 1 and len(vlm_side) == 1:
            pairs.append((ocr_side[0], vlm_side[0], "aligned_singleton", "singleton", 1.0,
                          ["exactly one instance of this type on each side"]))
            matched_ocr.add(id(ocr_side[0]))
            matched_vlm.add(id(vlm_side[0]))
        else:
            # 2. exact, unique normalized identifier
            ocr_by_id = _normalized_ids([e for e in ocr_side if id(e) not in matched_ocr])
            vlm_by_id = _normalized_ids([e for e in vlm_side if id(e) not in matched_vlm])
            for identifier, ocr_group in ocr_by_id.items():
                vlm_group = vlm_by_id.get(identifier)
                if not vlm_group:
                    continue
                if len(ocr_group) != 1 or len(vlm_group) != 1:
                    continue  # ambiguous on one side; not a reliable match
                pairs.append((ocr_group[0], vlm_group[0], "aligned_exact_identifier",
                              "exact_identifier", 1.0,
                              [f"unique normalized identifier {identifier!r} on both sides"]))
                matched_ocr.add(id(ocr_group[0]))
                matched_vlm.add(id(vlm_group[0]))

            # 3. spatial — see docstring: unreachable while the VLM returns no
            #    positional data. Left explicit rather than silently skipped.
            spatial_possible = any(e.position is not None for e in ocr_side) and \
                any(e.position is not None for e in vlm_side)

            # 4. contextual
            for ocr_entity in ocr_side:
                if id(ocr_entity) in matched_ocr:
                    continue
                best, best_score = None, 0.0
                for vlm_entity in vlm_side:
                    if id(vlm_entity) in matched_vlm:
                        continue
                    score = _context_similarity(ocr_entity, vlm_entity)
                    if score > best_score:
                        best, best_score = vlm_entity, score
                if best is not None and best_score >= cfg.min_context_similarity:
                    pairs.append((ocr_entity, best, "aligned_contextually", "context", best_score,
                                  [f"field/value overlap {best_score:.2f} >= "
                                   f"{cfg.min_context_similarity}"]))
                    matched_ocr.add(id(ocr_entity))
                    matched_vlm.add(id(best))

            # 5. whatever is left. Two different situations, kept apart:
            #    one side simply had no entity of this type (nothing to align
            #    against), versus both sides had candidates and no method
            #    matched them. Only the second is an alignment failure.
            counterpart_exists = bool(ocr_side) and bool(vlm_side)
            leftover_status = "unresolved" if counterpart_exists else "single_source"
            for entity, side_matched, other_side in (
                (e, matched_ocr, "vlm") for e in ocr_side
            ):
                if id(entity) in side_matched:
                    continue
                pairs.append((entity, None, leftover_status, "none", 0.0,
                              _leftover_reasons(leftover_status, other_side, spatial_possible)))
            for entity, side_matched, other_side in (
                (e, matched_vlm, "ocr") for e in vlm_side
            ):
                if id(entity) in side_matched:
                    continue
                pairs.append((None, entity, leftover_status, "none", 0.0,
                              _leftover_reasons(leftover_status, other_side, spatial_possible)))

        # Ordinals are assigned AFTER alignment, and only label the resulting
        # pair. They are never used to create one.
        for ordinal, (ocr_entity, vlm_entity, status, method, confidence, reasons) in enumerate(
            sorted(pairs, key=lambda p: _ordinal_sort_key(p[0], p[1])), start=1
        ):
            cause = None
            if status == "single_source":
                if not vlm_invoked:
                    cause = "vlm_not_invoked"
                elif ocr_entity is None:
                    cause = "ocr_side_absent"
                else:
                    cause = "vlm_side_absent"
            # A ref is issued only for an actual cross-source match. Handing
            # one to an unaligned entity would let downstream code compare two
            # readings that were never established to describe the same thing.
            aligned = status.startswith("aligned_")
            alignments.append(EntityAlignment(
                entity_ref=f"{entity_type}#{ordinal}" if aligned else None,
                entity_type=entity_type,
                status=status,
                method=method,
                confidence=confidence,
                ocr_entity_id=ocr_entity.identifier if ocr_entity else None,
                vlm_entity_id=vlm_entity.identifier if vlm_entity else None,
                ocr_source_key=ocr_entity.source_key if ocr_entity else None,
                vlm_source_key=vlm_entity.source_key if vlm_entity else None,
                single_source_cause=cause,
                reasons=reasons,
            ))

    return alignments


def _leftover_reasons(status: str, other_side: str, spatial_possible: bool) -> List[str]:
    if status == "single_source":
        return [f"no entity of this type on the {other_side} side; nothing to align against"]
    reasons = ["no unique identifier match and context overlap below threshold"]
    if not spatial_possible:
        reasons.append("spatial alignment unavailable: VLM output carries no bbox")
    return reasons


def _ordinal_sort_key(ocr_entity: Optional[SourceEntity], vlm_entity: Optional[SourceEntity]):
    """Deterministic ordering for ordinal assignment.

    Sorts by normalized identifier where one exists so the same page always
    yields the same refs across runs. This orders labels; it does not decide
    which entities pair with which — that is already settled above.
    """
    entity = ocr_entity or vlm_entity
    return (entity.normalized_identifier or "", entity.source_key or "") if entity else ("", "")


def assignment_certainty_index(alignments: Sequence[EntityAlignment]) -> set:
    """(source, source_key) pairs whose entity attribution can be relied on.

    Two situations qualify, and the difference between them is the whole point
    of keeping single_source_cause:

      the entity aligned across both sources, or
      only one source described it BECAUSE the router decided no second source
      was needed (single_source_cause == "vlm_not_invoked")

    The second is a deliberate judgement, not a loophole: a tiered pipeline
    whose cheap path can never produce an answerable value has no tiers. It
    does mean decision-readiness inherits the router's accuracy — if the router
    waves through a page it should have escalated, this index calls the result
    certain. That is the same bet the tiering itself makes.

    A page that WAS escalated and still yielded an entity only one source saw
    does not qualify: it was escalated precisely because something looked
    wrong there.
    """
    certain: set = set()
    for alignment in alignments:
        qualifies = (
            alignment.status.startswith("aligned_")
            or alignment.single_source_cause == "vlm_not_invoked"
        )
        if not qualifies:
            continue
        if alignment.ocr_source_key is not None:
            certain.add(("ocr", alignment.ocr_source_key))
        if alignment.vlm_source_key is not None:
            certain.add(("vlm", alignment.vlm_source_key))
    return certain


def alignment_index(alignments: Sequence[EntityAlignment]) -> Dict[Tuple[str, str], str]:
    """(source, source_key) -> entity_ref, for stamping evidence."""
    index: Dict[Tuple[str, str], str] = {}
    for alignment in alignments:
        if alignment.entity_ref is None:
            continue  # unaligned entities get no handle
        if alignment.ocr_source_key is not None:
            index[("ocr", alignment.ocr_source_key)] = alignment.entity_ref
        if alignment.vlm_source_key is not None:
            index[("vlm", alignment.vlm_source_key)] = alignment.entity_ref
    return index
