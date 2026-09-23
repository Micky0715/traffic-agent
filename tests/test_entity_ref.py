"""Tests for entity typing and cross-source alignment.

The property that matters most here is negative: a recognized equipment code
must never be the thing that decides whether two readings describe the same
entity. Several tests below deliberately make the two sources disagree about
the identifier and assert they still land on one entity_ref.
"""
from __future__ import annotations

import pytest

from src.vision.config import EntityTypeConfig, load_config
from src.vision.entity_ref import (
    align_entities, alignment_index, build_ocr_entities, build_vlm_entities,
    entity_type_for_field, map_entity_type,
)
from src.vision.schemas import FieldPair, SourceEntity


@pytest.fixture
def cfg() -> EntityTypeConfig:
    return load_config().entity_types


def identity(value: str) -> str:
    return value


def pair(field_name: str, value: str, entity_id=None, bbox=None) -> FieldPair:
    return FieldPair(entity_id=entity_id, field_name=field_name, raw_value=value,
                     bbox=bbox or [], confidence=0.9)


def entity(source, key, entity_type, identifier=None, fields=None, position=None) -> SourceEntity:
    return SourceEntity(
        source=source, source_key=key, entity_type=entity_type,
        identifier=identifier, normalized_identifier=identifier,
        fields=fields or {}, position=position)


# --------------------------------------------------------------------------
# controlled type mapping
# --------------------------------------------------------------------------

def test_exact_names_map_to_their_type(cfg):
    for name, expected in [("电机", "motor"), ("断路器", "breaker"),
                           ("控制柜", "cabinet"), ("控制器", "controller"),
                           ("风机", "fan"), ("水泵", "pump")]:
        entity_type, label, method = map_entity_type(name, cfg)
        assert (entity_type, method) == (expected, "exact"), name
        assert label == ""


def test_anchored_regex_extracts_the_identifier_from_the_name(cfg):
    entity_type, label, method = map_entity_type("2A号水泵", cfg)
    assert (entity_type, label, method) == ("pump", "2A", "regex")
    assert map_entity_type("20号水泵", cfg)[1] == "20"


def test_substring_lookalikes_are_not_typed(cfg):
    """A `contains` rule would call 控制柜温控器 a cabinet and 潜水泵壳体 a pump.
    A mistyped entity is then aligned against the wrong thing, so the mapping
    refuses rather than guesses."""
    for name in ["控制柜温控器", "潜水泵壳体", "风机房照明", "A16风机组"]:
        entity_type, _, method = map_entity_type(name, cfg)
        assert (entity_type, method) == ("unknown", "unmapped"), name


def test_empty_name_is_unknown_not_a_default_type(cfg):
    assert map_entity_type("", cfg)[0] == "unknown"


def test_cabinet_and_controller_stay_distinct_types(cfg):
    assert map_entity_type("控制柜", cfg)[0] != map_entity_type("控制器", cfg)[0]


def test_cabinet_number_and_cabinet_model_share_an_entity_but_not_a_field(cfg):
    """Same cabinet, two different canonical fields. Neither is ever rewritten
    into the other — they describe different properties."""
    assert entity_type_for_field("控制柜编号", cfg, None) == "cabinet"
    assert entity_type_for_field("控制柜型号", cfg, None) == "cabinet"
    assert "控制柜编号" != "控制柜型号"


def test_page_level_fields_belong_to_no_entity(cfg):
    for field in ["图号", "名称", "页码", "版本"]:
        assert entity_type_for_field(field, cfg, "fan") is None


def test_unnamed_field_falls_back_to_the_drawing_types_primary_entity(cfg):
    assert entity_type_for_field("功率", cfg, "fan") == "fan"
    assert entity_type_for_field("功率", cfg, None) is None


# --------------------------------------------------------------------------
# the core property: the identifier is not the key
# --------------------------------------------------------------------------

def test_sources_disagreeing_about_the_identifier_still_align(cfg):
    """The whole reason this layer exists. OCR reads M-19, the VLM reads M-I9.
    Keyed on the identifier they are two unrelated entities and the
    disagreement is never reported; typed and aligned, they are one motor."""
    ocr = [entity("ocr", "ocr:motor:page", "motor", "M-19", {"电机编号": "M-19"})]
    vlm = [entity("vlm", "vlm:dev0", "motor", "M-I9", {"电机编号": "M-I9"})]

    alignments = align_entities(ocr, vlm, cfg)
    assert len(alignments) == 1
    assert alignments[0].status == "aligned_singleton"
    assert alignments[0].ocr_entity_id == "M-19"
    assert alignments[0].vlm_entity_id == "M-I9"
    index = alignment_index(alignments)
    assert index[("ocr", "ocr:motor:page")] == index[("vlm", "vlm:dev0")]


def test_singleton_alignment_requires_one_instance_on_each_side(cfg):
    ocr = [entity("ocr", "o1", "fan", "A16"), entity("ocr", "o2", "fan", "A17")]
    vlm = [entity("vlm", "v1", "fan", "A16")]
    statuses = {a.status for a in align_entities(ocr, vlm, cfg)}
    assert "aligned_singleton" not in statuses


# --------------------------------------------------------------------------
# multi-device alignment
# --------------------------------------------------------------------------

def test_multi_device_aligns_on_unique_identifiers(cfg):
    """Round 4's correction to an earlier proposal: a page holding several
    entities of one type is NOT blanket-marked uncertain. Instances that match
    on a unique identifier stay aligned."""
    ocr = [entity("ocr", f"o{i}", "fan", d, {"设备编号": d})
           for i, d in enumerate(["A16", "A17", "A18"])]
    vlm = [entity("vlm", f"v{i}", "fan", d, {"设备编号": d})
           for i, d in enumerate(["A18", "A16", "A17"])]  # different order

    alignments = align_entities(ocr, vlm, cfg)
    assert all(a.status == "aligned_exact_identifier" for a in alignments)
    assert len(alignments) == 3
    # Order on either side is irrelevant: matching is by identifier, not index.
    for a in alignments:
        assert a.ocr_entity_id == a.vlm_entity_id


def test_only_the_unmatched_instance_is_left_unresolved(cfg):
    """One device the VLM missed must not invalidate the two that matched."""
    ocr = [entity("ocr", f"o{i}", "fan", d, {"设备编号": d})
           for i, d in enumerate(["A16", "A17", "A18"])]
    vlm = [entity("vlm", f"v{i}", "fan", d, {"设备编号": d})
           for i, d in enumerate(["A16", "A17"])]

    alignments = align_entities(ocr, vlm, cfg)
    aligned = [a for a in alignments if a.status == "aligned_exact_identifier"]
    unresolved = [a for a in alignments if a.status == "unresolved"]
    assert len(aligned) == 2
    assert len(unresolved) == 1
    assert unresolved[0].ocr_entity_id == "A18"
    assert unresolved[0].entity_ref is None


def test_ordinal_position_is_never_used_to_pair_entities(cfg):
    """If the VLM missed the first device, pairing by position would match
    OCR's A16 against the VLM's A17 and report two bogus conflicts."""
    ocr = [entity("ocr", f"o{i}", "fan", d, {"设备编号": d})
           for i, d in enumerate(["A16", "A17"])]
    vlm = [entity("vlm", "v0", "fan", "A17", {"设备编号": "A17"})]

    alignments = align_entities(ocr, vlm, cfg)
    pairs = {(a.ocr_entity_id, a.vlm_entity_id) for a in alignments}
    assert ("A16", "A17") not in pairs
    assert ("A17", "A17") in pairs


def test_context_overlap_can_align_when_identifiers_are_unusable(cfg):
    ocr = [entity("ocr", "o1", "fan", None, {"功率": "45kW", "风量": "28000m3/h"}),
           entity("ocr", "o2", "fan", None, {"功率": "55kW", "风量": "32000m3/h"})]
    vlm = [entity("vlm", "v1", "fan", None, {"功率": "55kW", "风量": "32000m3/h"}),
           entity("vlm", "v2", "fan", None, {"功率": "45kW", "风量": "28000m3/h"})]

    alignments = [a for a in align_entities(ocr, vlm, cfg)
                  if a.status == "aligned_contextually"]
    assert len(alignments) == 2


# --------------------------------------------------------------------------
# single source vs. failure to align
# --------------------------------------------------------------------------

def test_one_sided_type_is_single_source_not_an_alignment_failure(cfg):
    """A page where the VLM was never called has no VLM entities at all.
    Reporting that as an alignment failure would make the failure rate mostly
    a measure of how often fallback was skipped."""
    ocr = [entity("ocr", "o1", "cabinet", "FAN-CAB-19", {"控制柜编号": "FAN-CAB-19"})]
    alignments = align_entities(ocr, [], cfg)
    assert [a.status for a in alignments] == ["single_source"]
    assert alignments[0].entity_ref is None


def test_unaligned_entities_receive_no_entity_ref(cfg):
    """An unaligned entity must not carry a handle downstream code could
    mistake for a confirmed cross-source identity."""
    ocr = [entity("ocr", "o1", "fan", "A16", {"设备编号": "A16"}),
           entity("ocr", "o2", "fan", "A17", {"设备编号": "A17"})]
    vlm = [entity("vlm", "v1", "fan", "A16", {"设备编号": "A16"})]
    alignments = align_entities(ocr, vlm, cfg)
    for alignment in alignments:
        if alignment.status.startswith("aligned_"):
            assert alignment.entity_ref
        else:
            assert alignment.entity_ref is None


def test_spatial_alignment_reports_why_it_could_not_run(cfg):
    """The VLM adapter returns no bounding boxes, so the spatial method cannot
    fire. That is stated in the reasons rather than silently skipped."""
    ocr = [entity("ocr", "o1", "fan", None, {"功率": "45kW"}, position=[0, 0, 10, 10]),
           entity("ocr", "o2", "fan", None, {"功率": "55kW"})]
    vlm = [entity("vlm", "v1", "fan", None, {"风量": "9000m3/h"})]
    reasons = [r for a in align_entities(ocr, vlm, cfg) for r in a.reasons]
    assert any("no bbox" in r for r in reasons)


# --------------------------------------------------------------------------
# entity construction from each source
# --------------------------------------------------------------------------

def test_ocr_entities_group_fields_and_split_off_page_level_ones(cfg):
    pairs = [pair("图号", "FAN-A13-02"), pair("电机编号", "M-13"),
             pair("断路器编号", "QF-13"), pair("控制柜编号", "FAN-CAB-3")]
    entities, pair_keys, page_level = build_ocr_entities(pairs, cfg, "fan", identity)

    assert page_level == {"图号": "FAN-A13-02"}
    assert {e.entity_type for e in entities} == {"motor", "breaker", "cabinet"}
    assert {e.identifier for e in entities} == {"M-13", "QF-13", "FAN-CAB-3"}
    assert 0 not in pair_keys  # the page-level field belongs to no entity


def test_vlm_entities_record_how_their_type_was_decided(cfg):
    devices = [
        {"device_id": "M-23", "device_name": "电机", "parameters": []},
        {"device_id": "2A", "device_name": "2A号水泵", "parameters": []},
        {"device_id": "X", "device_name": "配电箱", "parameters": []},
    ]
    entities = build_vlm_entities(devices, cfg, [], {}, identity)
    by_type = {e.entity_type: e for e in entities}

    assert by_type["motor"].mapping_method == "exact"
    assert by_type["motor"].raw_entity_name == "电机"
    assert by_type["pump"].mapping_method == "regex"
    assert by_type["pump"].extracted_entity_label == "2A"
    assert by_type["unknown"].mapping_method == "unmapped"
    assert by_type["unknown"].raw_entity_name == "配电箱"


def test_vlm_device_id_is_expressed_as_its_types_identity_field(cfg):
    """So both sources describe the same thing in the same vocabulary — without
    this the identifier readings can never be compared."""
    devices = [{"device_id": "M-23", "device_name": "电机", "parameters": []}]
    entity_out = build_vlm_entities(devices, cfg, [], {}, identity)[0]
    assert entity_out.fields["电机编号"] == "M-23"


# --------------------------------------------------------------------------
# which attributions may be relied on
# --------------------------------------------------------------------------

def test_single_source_cause_separates_three_different_situations(cfg):
    from src.vision.entity_ref import align_entities as align

    not_invoked = align([entity("ocr", "o1", "motor", "M-13")], [], cfg, vlm_invoked=False)
    assert not_invoked[0].single_source_cause == "vlm_not_invoked"

    ocr_absent = align([], [entity("vlm", "v1", "motor", "M-23")], cfg, vlm_invoked=True)
    assert ocr_absent[0].single_source_cause == "ocr_side_absent"

    vlm_absent = align([entity("ocr", "o1", "cabinet", "FAN-CAB-19")], [], cfg, vlm_invoked=True)
    assert vlm_absent[0].single_source_cause == "vlm_side_absent"


def test_attribution_is_certain_when_the_router_decided_no_fallback_was_needed(cfg):
    """A tiered pipeline whose cheap path can never produce an answerable value
    has no tiers. Decision-readiness therefore inherits the router's accuracy,
    which is the same bet the tiering already makes."""
    from src.vision.entity_ref import assignment_certainty_index

    alignments = align_entities([entity("ocr", "o1", "motor", "M-13")], [], cfg,
                                vlm_invoked=False)
    assert ("ocr", "o1") in assignment_certainty_index(alignments)


def test_attribution_is_uncertain_when_an_escalated_page_saw_it_only_once(cfg):
    """The page was escalated because something looked wrong on it; a reading
    only one source produced there has not been corroborated."""
    from src.vision.entity_ref import assignment_certainty_index

    alignments = align_entities([], [entity("vlm", "v1", "motor", "M-23")], cfg,
                                vlm_invoked=True)
    assert assignment_certainty_index(alignments) == set()


def test_aligned_entities_are_always_certain(cfg):
    from src.vision.entity_ref import assignment_certainty_index

    alignments = align_entities(
        [entity("ocr", "o1", "motor", "M-19", {"电机编号": "M-19"})],
        [entity("vlm", "v1", "motor", "M-I9", {"电机编号": "M-I9"})], cfg)
    certain = assignment_certainty_index(alignments)
    assert ("ocr", "o1") in certain and ("vlm", "v1") in certain
