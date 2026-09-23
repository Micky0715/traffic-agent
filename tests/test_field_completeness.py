"""Tests for the silent-miss signals.

Most of these are written against synthetic OCRResults rather than the eval
fixtures, so a change in PaddleOCR's output cannot quietly turn them green.
The two that do use a fixture are marked, and they assert the specific real
behaviour that motivated the module.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.vision.config import DrawingTypeConfig, FieldCompletenessConfig, load_config
from src.vision.field_completeness import (
    classify_drawing_type, evaluate_field_completeness, find_value_block,
    watermark_repetition,
)
from src.vision.ocr_engine import get_ocr_engine
from src.vision.schemas import OCRBlock, OCRResult, TableStructure
from src.vision.table_structure import detect_table_structure

DRAWINGS = Path(__file__).resolve().parents[1] / "data" / "drawings"


def block(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 1.0) -> OCRBlock:
    return OCRBlock(text=text, bbox=[x0, y0, x1, y1], confidence=conf)


def ocr_of(*blocks: OCRBlock) -> OCRResult:
    return OCRResult(
        text="\n".join(b.text for b in blocks), blocks=list(blocks),
        average_confidence=1.0, engine="test",
    )


@pytest.fixture
def cfg() -> FieldCompletenessConfig:
    return load_config().field_completeness


@pytest.fixture
def types():
    return load_config().drawing_types


# --------------------------------------------------------------------------
# label -> value adjacency
# --------------------------------------------------------------------------

def test_value_is_found_to_the_right_on_the_same_line(cfg):
    label = block("图号", 39, 502, 84, 525)
    value = block("FAN-A13-02", 208, 503, 326, 524)
    assert find_value_block(label, [label, value], cfg) is value


def test_a_page_spanning_watermark_block_is_not_accepted_as_a_value(cfg):
    """The exact failure this height test exists for: on FAN-A23-01 a 192px
    tall watermark block overlaps every label on the page and sits to their
    right. Without the height check it satisfies the value test for all of
    them and the occlusion becomes invisible."""
    label = block("图号", 43, 486, 96, 520)
    watermark = block("内部资料严禁外传内部资料", 385, 508, 711, 700)
    assert find_value_block(label, [label, watermark], cfg) is None


def test_a_block_on_a_different_line_is_not_a_value(cfg):
    label = block("图号", 39, 502, 84, 525)
    other = block("4/6", 205, 564, 244, 592)
    assert find_value_block(label, [label, other], cfg) is None


def test_a_block_to_the_left_is_not_a_value(cfg):
    label = block("图号", 300, 502, 350, 525)
    left = block("FAN-A13-02", 39, 503, 200, 524)
    assert find_value_block(label, [label, left], cfg) is None


# --------------------------------------------------------------------------
# watermark repetition
# --------------------------------------------------------------------------

def test_a_title_echoed_into_a_field_does_not_count_as_a_watermark():
    """Every drawing prints its name twice — once as a title, once as the 名称
    value. An earlier version counted any n-gram appearing twice and fired on
    every clean page because of exactly this."""
    text = "A13风机接线图\n图号 FAN-A13-02\n名称 A13风机接线图\n页码 4/6"
    assert watermark_repetition(text, ngram=6, min_repeats=3) == 0.0


def test_a_tiled_notice_counts_as_a_watermark():
    text = "严禁外传内部资料" * 6
    assert watermark_repetition(text, ngram=6, min_repeats=3) > 0.5


def test_short_text_cannot_produce_a_repetition_score():
    assert watermark_repetition("图号", ngram=6, min_repeats=3) == 0.0


# --------------------------------------------------------------------------
# drawing type classification
# --------------------------------------------------------------------------

def test_group_title_wins_over_wiring_title(types):
    """A group drawing's title legitimately contains both words. Reading it as
    a plain wiring diagram would check it against single-device fields it does
    not have."""
    ocr = ocr_of(block("A16/A17/A18风机组接线图", 0, 0, 300, 30))
    assert classify_drawing_type(ocr, types) == ("fan_group", "title")


def test_a_fan_prefix_alone_does_not_make_something_a_wiring_diagram(types):
    """FAN- is carried by wiring diagrams, parameter tables AND group
    drawings; classifying on the prefix alone would give three different kinds
    of drawing the same field checklist."""
    ocr = ocr_of(block("A28风机参数表", 0, 0, 200, 30), block("FAN-A28-01", 0, 40, 200, 70))
    name, source = classify_drawing_type(ocr, types)
    assert name == "fan_param_table" and source == "title"


def test_drawing_number_prefix_is_used_when_there_is_no_usable_title(types):
    ocr = ocr_of(block("PUMP-MULTI-02", 0, 0, 200, 30))
    assert classify_drawing_type(ocr, types) == ("pump_group", "drawing_no")


def test_unreadable_page_is_unknown_rather_than_guessed(types):
    assert classify_drawing_type(OCRResult(text="", engine="test"), types) == ("unknown", "none")


# --------------------------------------------------------------------------
# completeness, one_of groups, task fields
# --------------------------------------------------------------------------

def _wiring_page(page_label: str = "页码") -> OCRResult:
    return ocr_of(
        block("A13风机接线图", 16, 24, 233, 55),
        block("图号", 39, 502, 84, 525), block("FAN-A13-02", 208, 503, 326, 524),
        block("名称", 38, 532, 85, 558), block("A13风机接线图", 207, 534, 341, 558),
        block(page_label, 39, 565, 83, 589), block("4/6", 205, 564, 244, 592),
        block("电机编号", 39, 593, 120, 622), block("M-13", 206, 597, 261, 622),
        block("断路器编号", 39, 624, 139, 651), block("QF-13", 205, 627, 268, 656),
        block("控制柜编号", 39, 653, 140, 681), block("FAN-CAB-3", 207, 660, 315, 683),
    )


def test_complete_wiring_page_scores_one(types, cfg):
    result = evaluate_field_completeness(_wiring_page(), types, cfg)
    assert result.drawing_type == "fan_wiring"
    assert result.completeness == 1.0
    assert result.isolated_labels == []
    assert result.reasons == []


def test_revision_satisfies_the_page_or_revision_group(types, cfg):
    """FAN-A26-01 prints 版本 and no 页码. Against a flat required-field list
    it would be scored incomplete for a field it was never supposed to have."""
    result = evaluate_field_completeness(_wiring_page(page_label="版本"), types, cfg)
    assert result.completeness == 1.0
    assert "版本" in result.found_fields


def test_task_required_fields_override_the_type_default(types, cfg):
    """版本 satisfies basic completeness, but it cannot stand in for 页码 when
    the user actually asked which page this is."""
    page = _wiring_page(page_label="版本")
    result = evaluate_field_completeness(page, types, cfg, task_required_fields=["页码"])
    assert result.completeness == 0.0
    assert "using_task_required_fields" in result.reasons


def test_label_without_value_is_reported_as_an_isolated_label(types, cfg):
    page = ocr_of(
        block("A13风机接线图", 16, 24, 233, 55),
        block("图号", 39, 502, 84, 525),  # no value to its right
        block("名称", 38, 532, 85, 558), block("A13风机接线图", 207, 534, 341, 558),
        block("页码", 39, 565, 83, 589), block("4/6", 205, 564, 244, 592),
        block("电机编号", 39, 593, 120, 622), block("M-13", 206, 597, 261, 622),
        block("断路器编号", 39, 624, 139, 651), block("QF-13", 205, 627, 268, 656),
        block("控制柜编号", 39, 653, 140, 681), block("FAN-CAB-3", 207, 660, 315, 683),
    )
    result = evaluate_field_completeness(page, types, cfg)
    assert result.isolated_labels == ["图号"]
    assert "isolated_labels_without_values" in result.reasons
    assert result.completeness < 1.0


def test_unknown_type_short_circuits_to_zero_completeness(types, cfg):
    result = evaluate_field_completeness(OCRResult(text="", engine="test"), types, cfg)
    assert result.drawing_type == "unknown"
    assert result.completeness == 0.0
    assert result.reasons == ["drawing_type_unknown"]


# --------------------------------------------------------------------------
# multi-device cardinality
# --------------------------------------------------------------------------

def _group_page() -> OCRResult:
    blocks = [block("A16/A17/A18风机组接线图", 0, 0, 300, 30)]
    y = 100
    for device in ("A16", "A17", "A18"):
        for suffix, value in (("设备编号", device), ("功率", "45kW"), ("风量", "28000m3/h")):
            blocks.append(block(f"{device}{suffix}", 39, y, 160, y + 25))
            blocks.append(block(value, 205, y, 330, y + 25))
            y += 30
    return ocr_of(*blocks)


def test_group_page_is_always_cardinality_uncertain(types, cfg):
    """Known limitation, asserted so it cannot be forgotten: this pipeline has
    no device count independent of the OCR text, so "every device I found is
    complete" can never rule out a device row that was never found. Multi-
    device pages therefore always escalate."""
    result = evaluate_field_completeness(_group_page(), types, cfg)
    assert result.drawing_type == "fan_group"
    assert result.group_device_count == 3
    assert result.completeness == 1.0  # every device it CAN see is complete
    assert result.group_cardinality_uncertain is True
    assert "device_count_not_independently_verifiable" in result.reasons


def test_group_page_missing_a_whole_device_still_looks_complete(types, cfg):
    """The trap itself, pinned down: drop one device entirely and per-device
    completeness stays 1.0. Only the cardinality flag catches it, which is why
    that flag cannot be made conditional on completeness."""
    blocks = [b for b in _group_page().blocks if not b.text.startswith("A17")]
    result = evaluate_field_completeness(ocr_of(*blocks), types, cfg)
    assert result.completeness == 1.0
    assert result.group_device_count == 2
    assert result.group_cardinality_uncertain is True


def test_device_missing_one_field_lowers_completeness(types, cfg):
    blocks = [b for b in _group_page().blocks if b.text != "A17功率"]
    result = evaluate_field_completeness(ocr_of(*blocks), types, cfg)
    assert result.completeness < 1.0
    assert "A17功率" in result.missing_fields


# --------------------------------------------------------------------------
# against the real fixture
# --------------------------------------------------------------------------

@pytest.mark.fixture_ocr
def test_watermarked_page_is_caught_despite_high_ocr_confidence(types, cfg):
    """The case this whole module exists for. Real PaddleOCR returns 0.996
    average confidence on FAN-A23-01 while missing 3 of 4 target fields,
    because it confidently read the watermark instead."""
    image = DRAWINGS / "FAN-A23-01.png"
    ocr = get_ocr_engine("fixture").recognize(image)
    assert ocr.average_confidence > 0.99  # confidence says everything is fine

    result = evaluate_field_completeness(
        ocr, types, cfg, table=detect_table_structure(image, ocr_result=ocr))
    assert result.drawing_type == "fan_wiring"
    assert result.completeness < cfg.min_completeness
    assert result.isolated_labels  # labels read, value slots empty
    assert result.watermark_repetition > cfg.max_watermark_repetition


@pytest.mark.fixture_ocr
def test_clean_page_stays_silent(types, cfg):
    image = DRAWINGS / "FAN-A13-02.png"
    ocr = get_ocr_engine("fixture").recognize(image)
    result = evaluate_field_completeness(
        ocr, types, cfg, table=detect_table_structure(image, ocr_result=ocr))
    assert result.completeness == 1.0
    assert result.reasons == []
