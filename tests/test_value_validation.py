"""Tests for field-value shape checking.

Written against synthetic values rather than the eval images wherever
possible, so the suite keeps testing the rules rather than one PaddleOCR
version's output. The one test that touches a real drawing asserts the
behaviour that motivated the module, and reaches it through OCR geometry — it
never looks at a file name or a sample id.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.vision.config import (
    DrawingTypeConfig, ValueRuleDefinition, ValueValidationConfig, load_config,
)
from src.vision.field_completeness import evaluate_field_completeness
from src.vision.ocr_engine import get_ocr_engine
from src.vision.schemas import FieldPair
from src.vision.table_structure import detect_table_structure
from src.vision.value_validation import (
    field_pairs_from_completeness, normalize_value, validate_field_values,
    value_validity_below_threshold,
)

DRAWINGS = Path(__file__).resolve().parents[1] / "data" / "drawings"


@pytest.fixture
def cfg() -> ValueValidationConfig:
    return load_config().value_validation


def pair(field_name: str, value: str, entity_id=None) -> FieldPair:
    return FieldPair(entity_id=entity_id, field_name=field_name, raw_value=value, confidence=1.0)


# --------------------------------------------------------------------------
# 1-2. well-formed values pass
# --------------------------------------------------------------------------

def test_wellformed_drawing_number_passes(cfg):
    result = validate_field_values([pair("图号", "FAN-A13-02")], cfg)
    assert result.valid_fields == ["图号"]
    assert result.invalid_fields == []
    assert result.valid_ratio == 1.0


def test_wellformed_page_number_and_revision_pass(cfg):
    result = validate_field_values([pair("页码", "4/6"), pair("版本", "V3")], cfg)
    assert set(result.valid_fields) == {"页码", "版本"}
    assert result.valid_ratio == 1.0


def test_spacing_and_full_width_forms_do_not_make_a_value_invalid(cfg):
    """OCR returns full-width hyphens and stray spacing around dashes on
    perfectly legible codes. Normalization has to absorb that, or the checker
    reports formatting noise as a business error."""
    result = validate_field_values([pair("电机编号", "Ｍ － 13")], cfg)
    assert result.valid_fields == ["电机编号"]


# --------------------------------------------------------------------------
# 3. stamp-corrupted value is rejected
# --------------------------------------------------------------------------

def test_stamp_corrupted_equipment_code_is_invalid(cfg):
    """Value slot filled, OCR confident, completeness 1.0 — and the value is
    stamp text. This is the class of failure every earlier signal misses."""
    result = validate_field_values([pair("控制柜编号", "审核专用毫AB9")], cfg)
    assert result.invalid_fields == ["控制柜编号"]
    detail = result.invalid_details[0]
    assert detail.observed_value == "审核专用毫AB9"
    assert detail.violated_rule == "equipment_code_charset"
    assert detail.rule_origin == "character_structure"


def test_an_invalid_value_is_reported_never_repaired(cfg):
    """The detail carries the original text unchanged. Nothing anywhere
    rewrites it toward the pattern — a repaired value would erase the evidence
    that the page needs a second look."""
    corrupt = "审核专用毫AB9"
    result = validate_field_values([pair("控制柜编号", corrupt)], cfg)
    assert result.invalid_details[0].observed_value == corrupt
    assert result.invalid_details[0].normalized_value == corrupt


# --------------------------------------------------------------------------
# 4. no rule -> unknown, never invalid
# --------------------------------------------------------------------------

def test_field_without_a_rule_is_unknown_not_invalid(cfg):
    result = validate_field_values([pair("功率", "45kW"), pair("名称", "A13风机接线图")], cfg)
    assert set(result.unknown_fields) == {"功率", "名称"}
    assert result.invalid_fields == []
    assert result.checked_fields == []


def test_valid_ratio_is_none_when_nothing_was_checked(cfg):
    """Not 0.0 and not 1.0. A page with nothing checkable has no validity to
    report, and either number would be a fabricated verdict."""
    result = validate_field_values([pair("功率", "45kW")], cfg)
    assert result.valid_ratio is None


def test_rate_trigger_stays_silent_below_the_minimum_checked_fields(cfg):
    """With one checkable field the ratio can only be 0.0 or 1.0, so a
    threshold over it measures nothing."""
    result = validate_field_values([pair("控制柜编号", "审核专用毫AB9")], cfg)
    assert result.valid_ratio == 0.0
    assert len(result.checked_fields) < cfg.min_checked_fields
    assert value_validity_below_threshold(result, cfg) is False


def test_rate_trigger_fires_once_enough_fields_are_checked(cfg):
    result = validate_field_values([
        pair("控制柜编号", "审核专用毫AB9"),
        pair("电机编号", "严禁外传"),
        pair("图号", "FAN-A13-02"),
    ], cfg)
    assert result.valid_ratio == pytest.approx(1 / 3)
    assert value_validity_below_threshold(result, cfg) is True


# --------------------------------------------------------------------------
# per-type overrides; no dataset answers baked into the defaults
# --------------------------------------------------------------------------

def test_default_cabinet_rule_accepts_any_project_prefix(cfg):
    """The shipped default checks character structure only. A FAN-CAB-* rule
    would be this dataset's answer written back as a rule, and would reject a
    pump cabinet out of hand."""
    result = validate_field_values([
        pair("控制柜编号", "FAN-CAB-19"), pair("控制柜编号", "PUMP-CAB-02"),
    ], cfg)
    assert result.invalid_fields == []


def test_a_project_can_enforce_its_own_prefix_through_type_overrides(cfg):
    cfg.value_rule_definitions["fan_cabinet_prefix"] = ValueRuleDefinition(
        pattern="^FAN-CAB-", reason="project scheme", origin="project_domain_convention")
    spec = DrawingTypeConfig(
        field_value_rule_overrides={"控制柜编号": ["equipment_code_charset", "fan_cabinet_prefix"]})
    result = validate_field_values([pair("控制柜编号", "PUMP-CAB-02")], cfg, spec)
    assert result.invalid_fields == ["控制柜编号"]
    assert result.invalid_details[0].rule_origin == "project_domain_convention"


def test_a_referenced_but_undefined_rule_is_reported_not_ignored(cfg):
    cfg.field_value_rules["图号"] = ["no_such_rule"]
    result = validate_field_values([pair("图号", "FAN-A13-02")], cfg)
    assert result.invalid_fields == ["图号"]
    assert result.invalid_details[0].rule_origin == "configuration_error"


# --------------------------------------------------------------------------
# entity-scoped naming
# --------------------------------------------------------------------------

def test_per_device_fields_are_reported_under_their_entity(cfg):
    result = validate_field_values([
        pair("设备编号", "A16", entity_id="A16"), pair("设备编号", "A17", entity_id="A17"),
    ], cfg)
    assert set(result.valid_fields) == {"A16设备编号", "A17设备编号"}


def test_empty_values_are_not_validated(cfg):
    """An empty slot is a completeness fault, already reported there. Running
    shape rules over "" would count one fault twice as two signals."""
    assert field_pairs_from_completeness([pair("图号", "   ")]) == []


# --------------------------------------------------------------------------
# 8. reached through geometry, never through a file name or sample id
# --------------------------------------------------------------------------

@pytest.mark.fixture_ocr
def test_stamped_page_is_flagged_without_any_filename_or_id_lookup():
    """The whole path on a real page: OCR -> label/value geometry -> value
    rules. The image path is only ever opened, never parsed; nothing in the
    chain branches on the file name or a case id."""
    full = load_config()
    image = DRAWINGS / "FAN-A19-01.png"
    ocr = get_ocr_engine("fixture").recognize(image)
    assert ocr.average_confidence > 0.98  # confidence says the page is fine

    completeness = evaluate_field_completeness(
        ocr, full.drawing_types, full.field_completeness,
        table=detect_table_structure(image, ocr_result=ocr),
        min_table_confidence=full.min_table_confidence)
    assert completeness.completeness == 1.0  # and so does completeness

    result = validate_field_values(
        field_pairs_from_completeness(completeness.field_pairs),
        full.value_validation, full.drawing_types.get(completeness.drawing_type))
    assert "控制柜编号" in result.invalid_fields


def test_normalization_never_invents_business_characters():
    """Normalization may fold encodings and spacing. It may not add, delete or
    substitute characters that carry meaning — a value cleaned into shape is a
    defect hidden rather than found."""
    variants = load_config().value_validation.hyphen_variants
    assert normalize_value("ＦＡＮ－Ａ13－02", variants) == "FAN-A13-02"
    assert normalize_value("审核专用毫AB9", variants) == "审核专用毫AB9"
    assert normalize_value("QF-I3", variants) == "QF-I3"  # OCR misread stays a misread
