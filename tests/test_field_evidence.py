"""Tests for field-level evidence merging and conflict recording.

Rewritten for round 4. The previous version flattened every VLM device onto
page-level fields; evidence is now grouped by an entity_ref produced by the
alignment layer, so these tests go through that layer rather than around it.

The behaviour under test stays narrow: keep both readings, key disagreements on
(entity_ref, canonical field name), and never pick a winner.
"""
from __future__ import annotations

import pytest

from src.vision.config import load_config
from src.vision.entity_ref import align_entities, alignment_index, build_vlm_entities
from src.vision.evidence import (
    evidence_from_ocr, evidence_from_vlm_entities, merge_evidence,
)
from src.vision.schemas import FieldPair, SourceEntity, ValueValidationResult
from src.vision.value_validation import normalize_value, validate_field_values


@pytest.fixture
def full():
    return load_config()


@pytest.fixture
def cfg(full):
    return full.value_validation


@pytest.fixture
def entity_cfg(full):
    return full.entity_types


@pytest.fixture
def mapping(full):
    return full.vlm_field_mapping


def norm_with(cfg):
    return lambda v: normalize_value(v, cfg.hyphen_variants, cfg.collapse_spaces_around_hyphen)


def pair(field_name: str, value: str, entity_id=None) -> FieldPair:
    return FieldPair(entity_id=entity_id, field_name=field_name, raw_value=value, confidence=0.9)


def ocr_entity(key, entity_type, identifier, fields) -> SourceEntity:
    return SourceEntity(source="ocr", source_key=key, entity_type=entity_type,
                        identifier=identifier, normalized_identifier=identifier, fields=fields)


def build(ocr_pairs, pair_keys, ocr_entities, devices, full, cfg, entity_cfg, mapping):
    """Run the real path: build VLM entities, align, stamp, merge."""
    normalize = norm_with(cfg)
    vlm_entities = build_vlm_entities(
        devices, entity_cfg, list(cfg.field_value_rules), mapping.parameter_names, normalize)
    alignments = align_entities(ocr_entities, vlm_entities, entity_cfg)
    index = alignment_index(alignments)

    validation = validate_field_values(ocr_pairs, cfg)
    ocr_evidence = evidence_from_ocr(ocr_pairs, validation, cfg, pair_keys, index)
    vlm_evidence = evidence_from_vlm_entities("{}", vlm_entities, mapping, cfg, index)
    return merge_evidence(ocr_evidence, vlm_evidence, alignments)


# --------------------------------------------------------------------------
# agreeing sources
# --------------------------------------------------------------------------

def test_matching_values_on_one_aligned_entity_produce_no_conflict(full, cfg, entity_cfg, mapping):
    merged = build(
        [pair("电机编号", "M-19")], {0: "ocr:motor:page"},
        [ocr_entity("ocr:motor:page", "motor", "M-19", {"电机编号": "M-19"})],
        [{"device_id": "M-19", "device_name": "电机", "parameters": []}],
        full, cfg, entity_cfg, mapping)

    assert merged.conflicts == []
    assert {e.source for e in merged.evidence} == {"ocr", "vlm"}
    refs = {e.entity_ref for e in merged.evidence}
    assert refs == {"motor#1"}  # both readings landed on the same entity


def test_component_readings_are_no_longer_flattened_away(full, cfg, entity_cfg, mapping):
    """Round 3 suppressed VLM component identifiers on single-device drawings,
    which the metric-gap audit showed cost two of the three missing fields.
    A motor and a breaker now keep separate refs instead of colliding."""
    merged = build(
        [], {}, [],
        [{"device_id": "M-23", "device_name": "电机", "parameters": []},
         {"device_id": "QF-23", "device_name": "断路器", "parameters": []}],
        full, cfg, entity_cfg, mapping)

    values = {(e.field_name, e.raw_value) for e in merged.evidence}
    assert ("电机编号", "M-23") in values
    assert ("断路器编号", "QF-23") in values
    assert merged.conflicts == []


def test_formatting_only_differences_are_not_conflicts(full, cfg, entity_cfg, mapping):
    merged = build(
        [pair("电机编号", "Ｍ－19")], {0: "ocr:motor:page"},
        [ocr_entity("ocr:motor:page", "motor", "M-19", {"电机编号": "M-19"})],
        [{"device_id": "M-19", "device_name": "电机", "parameters": []}],
        full, cfg, entity_cfg, mapping)
    assert merged.conflicts == []


# --------------------------------------------------------------------------
# disagreeing sources
# --------------------------------------------------------------------------

def test_differing_values_on_one_entity_produce_an_unresolved_conflict(
        full, cfg, entity_cfg, mapping):
    merged = build(
        [pair("控制柜编号", "审核专用毫AB9")], {0: "ocr:cabinet:page"},
        [ocr_entity("ocr:cabinet:page", "cabinet", "审核专用毫AB9",
                    {"控制柜编号": "审核专用毫AB9"})],
        [{"device_id": "FAN-CAB-19", "device_name": "控制柜", "parameters": []}],
        full, cfg, entity_cfg, mapping)

    assert len(merged.conflicts) == 1
    conflict = merged.conflicts[0]
    assert conflict.entity_ref == "cabinet#1"
    assert conflict.field_name == "控制柜编号"
    assert conflict.ocr_value == "审核专用毫AB9"
    assert conflict.vlm_value == "FAN-CAB-19"
    assert conflict.ocr_validation == "invalid"
    assert conflict.resolution == "unresolved"


def test_a_conflict_keeps_both_values_and_picks_neither(full, cfg, entity_cfg, mapping):
    merged = build(
        [pair("控制柜编号", "审核专用毫AB9")], {0: "ocr:cabinet:page"},
        [ocr_entity("ocr:cabinet:page", "cabinet", "审核专用毫AB9",
                    {"控制柜编号": "审核专用毫AB9"})],
        [{"device_id": "FAN-CAB-19", "device_name": "控制柜", "parameters": []}],
        full, cfg, entity_cfg, mapping)

    values = {(e.source, e.raw_value) for e in merged.evidence}
    assert ("ocr", "审核专用毫AB9") in values
    assert ("vlm", "FAN-CAB-19") in values
    assert all(c.resolution == "unresolved" for c in merged.conflicts)


def test_a_misread_identifier_no_longer_hides_the_disagreement(
        full, cfg, entity_cfg, mapping):
    """The failure round 4 exists to fix. Keyed on the recognized code, M-19
    and M-I9 were two unrelated entities and nothing was reported."""
    merged = build(
        [pair("电机编号", "M-I9")], {0: "ocr:motor:page"},
        [ocr_entity("ocr:motor:page", "motor", "M-I9", {"电机编号": "M-I9"})],
        [{"device_id": "M-19", "device_name": "电机", "parameters": []}],
        full, cfg, entity_cfg, mapping)

    assert len(merged.conflicts) == 1
    assert merged.conflicts[0].entity_ref == "motor#1"
    assert {merged.conflicts[0].ocr_value, merged.conflicts[0].vlm_value} == {"M-I9", "M-19"}


# --------------------------------------------------------------------------
# entity scoping
# --------------------------------------------------------------------------

def test_two_devices_of_one_type_are_not_a_conflict(full, cfg, entity_cfg, mapping):
    """A group drawing carries one identifier per device. Keyed on the field
    name alone every multi-device page would be a permanent contradiction."""
    merged = build(
        [pair("设备编号", "A16", entity_id="A16"), pair("设备编号", "A17", entity_id="A17")],
        {0: "ocr:fan:A16", 1: "ocr:fan:A17"},
        [ocr_entity("ocr:fan:A16", "fan", "A16", {"设备编号": "A16"}),
         ocr_entity("ocr:fan:A17", "fan", "A17", {"设备编号": "A17"})],
        [{"device_id": "A16", "device_name": "风机", "parameters": []},
         {"device_id": "A17", "device_name": "风机", "parameters": []}],
        full, cfg, entity_cfg, mapping)

    assert merged.conflicts == []
    assert {e.entity_ref for e in merged.evidence} == {"fan#1", "fan#2"}


def test_unaligned_entity_readings_are_recorded_but_never_compared(
        full, cfg, entity_cfg, mapping):
    """Two readings that were never established to describe the same equipment
    must not be reported as a disagreement about it."""
    merged = build(
        [pair("设备编号", "A16", entity_id="A16"), pair("设备编号", "A17", entity_id="A17")],
        {0: "ocr:fan:A16", 1: "ocr:fan:A17"},
        [ocr_entity("ocr:fan:A16", "fan", "A16", {"设备编号": "A16"}),
         ocr_entity("ocr:fan:A17", "fan", "A17", {"设备编号": "A17"})],
        [{"device_id": "A99", "device_name": "风机", "parameters": []}],
        full, cfg, entity_cfg, mapping)

    assert merged.conflicts == []
    unaligned = [e for e in merged.evidence if e.entity_ref is None]
    assert unaligned  # still on the record
    assert all(e.entity_assignment_uncertain for e in unaligned)


def test_an_unmapped_device_name_yields_no_entity_ref(full, cfg, entity_cfg, mapping):
    merged = build(
        [], {}, [],
        [{"device_id": "X-1", "device_name": "配电箱", "parameters": []}],
        full, cfg, entity_cfg, mapping)
    assert all(e.entity_ref is None for e in merged.evidence if e.entity_type)
    assert merged.entity_assignment_uncertain_count >= 1


# --------------------------------------------------------------------------
# source fidelity
# --------------------------------------------------------------------------

def test_every_candidate_records_source_raw_and_normalized_value(cfg):
    validation = validate_field_values([pair("图号", "ＦＡＮ－A13－02")], cfg)
    evidence = evidence_from_ocr([pair("图号", "ＦＡＮ－A13－02")], validation, cfg)[0]
    assert evidence.source == "ocr"
    assert evidence.raw_value == "ＦＡＮ－A13－02"
    assert evidence.normalized_value == "FAN-A13-02"
    assert evidence.validation_status == "valid"
    assert evidence.occurrence_id


def test_malformed_vlm_metadata_yields_no_metadata_evidence(cfg, mapping):
    assert evidence_from_vlm_entities("not json", [], mapping, cfg) == []
