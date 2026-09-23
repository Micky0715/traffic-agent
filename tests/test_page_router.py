from src.vision.config import VisualFallbackConfig
from src.vision.page_router import route_page, route_regions
from src.vision.schemas import ImageQualityMetrics, LayoutBlock, PageParseResult, Region, RegionType

CFG = VisualFallbackConfig()


def test_native_text_page_does_not_need_vlm():
    page = PageParseResult(page_number=1, native_text="x" * 1000, ocr_text="", ocr_confidence=1.0, image_ratio=0.05)
    decision = route_page(page, CFG)
    assert decision.page_type == "native_text"
    assert decision.need_vlm is False


def test_engineering_drawing_triggers_vlm():
    page = PageParseResult(
        page_number=2, native_text="", ocr_text="some ocr", ocr_confidence=0.9, image_ratio=0.8,
        layout_blocks=[LayoutBlock(block_type="title_block"), LayoutBlock(block_type="figure")],
        detected_figures=1, has_title_block=True,
    )
    decision = route_page(page, CFG)
    assert decision.page_type == "engineering_drawing"
    assert decision.need_vlm is True
    assert "engineering_drawing_detected" in decision.reason


def test_low_ocr_confidence_and_garbled_text_trigger_vlm():
    page = PageParseResult(page_number=3, native_text="", ocr_text="blah", ocr_confidence=0.4,
                            image_ratio=0.6, garbled_ratio=0.2)
    decision = route_page(page, CFG)
    assert decision.need_vlm is True
    assert "ocr_confidence_below_threshold" in decision.reason
    assert "garbled_text_ratio_above_threshold" in decision.reason


def test_thresholds_are_config_driven_not_hardcoded():
    page = PageParseResult(page_number=4, native_text="", ocr_text="blah", ocr_confidence=0.6,
                            image_ratio=0.6, garbled_ratio=0.05)
    lenient_cfg = VisualFallbackConfig(min_ocr_confidence=0.5)
    strict_cfg = VisualFallbackConfig(min_ocr_confidence=0.9)
    assert route_page(page, lenient_cfg).need_vlm is False
    assert route_page(page, strict_cfg).need_vlm is True


def test_native_text_page_needs_native_extract_not_layout():
    page = PageParseResult(page_number=1, native_text="x" * 1000, ocr_text="", ocr_confidence=1.0, image_ratio=0.05)
    decision = route_page(page, CFG)
    assert decision.need_native_extract is True
    assert decision.need_layout is False


def test_engineering_drawing_needs_layout():
    page = PageParseResult(
        page_number=2, native_text="", ocr_text="some ocr", ocr_confidence=0.9, image_ratio=0.8,
        layout_blocks=[LayoutBlock(block_type="title_block"), LayoutBlock(block_type="figure")],
        detected_figures=1, has_title_block=True,
    )
    decision = route_page(page, CFG)
    assert decision.need_layout is True


def test_no_quality_signal_means_no_preprocess_decision():
    page = PageParseResult(page_number=3, native_text="", ocr_text="blah", ocr_confidence=0.4,
                            image_ratio=0.6, garbled_ratio=0.2)
    decision = route_page(page, CFG)  # quality omitted
    assert decision.need_preprocess is False


def test_low_sharpness_triggers_preprocess_when_ocr_needed():
    page = PageParseResult(page_number=3, native_text="", ocr_text="blah", ocr_confidence=0.4,
                            image_ratio=0.6, garbled_ratio=0.2)
    blurry = ImageQualityMetrics(sharpness=10.0, contrast=60.0, skew_angle_deg=0.0, brightness_uniformity=1.0)
    decision = route_page(page, CFG, quality=blurry)
    assert decision.need_preprocess is True
    assert "low_sharpness_needs_denoise" in decision.reason


def test_good_quality_image_does_not_trigger_preprocess():
    page = PageParseResult(page_number=3, native_text="", ocr_text="blah", ocr_confidence=0.4,
                            image_ratio=0.6, garbled_ratio=0.2)
    clean = ImageQualityMetrics(sharpness=200.0, contrast=80.0, skew_angle_deg=0.1, brightness_uniformity=0.98)
    decision = route_page(page, CFG, quality=clean)
    assert decision.need_preprocess is False


def test_route_regions_only_selects_eligible_types():
    regions = [
        Region(region_id="r1", region_type=RegionType.TITLE_BLOCK, page_number=1, image_path="p.png", ocr_confidence=0.95),
        Region(region_id="r2", region_type=RegionType.LEGEND, page_number=1, image_path="p.png", ocr_confidence=0.2),
        Region(region_id="r3", region_type=RegionType.LOW_CONFIDENCE_OCR_REGION, page_number=1, image_path="p.png", ocr_confidence=0.3),
        Region(region_id="r4", region_type=RegionType.ANNOTATION_REGION, page_number=1, image_path="p.png", ocr_confidence=0.9),
    ]
    selected = route_regions(regions, CFG)
    assert "r1" in selected  # title_block always needs structural read
    assert "r2" not in selected  # legend is never VLM-eligible
    assert "r3" in selected  # low OCR confidence region
    assert "r4" not in selected  # annotation region with fine OCR confidence


# ---------------------------------------------------------------------------
# route_ocr_page: VLM fallback decided from OCR evidence rather than
# confidence alone
# ---------------------------------------------------------------------------

from src.vision.config import load_config  # noqa: E402
from src.vision.page_router import route_ocr_page  # noqa: E402
from src.vision.schemas import FieldCompleteness, OCRResult, TableStructure  # noqa: E402


def _clean_completeness(**overrides) -> FieldCompleteness:
    base = dict(drawing_type="fan_wiring", type_source="title", completeness=1.0,
                watermark_repetition=0.0)
    base.update(overrides)
    return FieldCompleteness(**base)


def _confident_ocr() -> OCRResult:
    return OCRResult(text="x", average_confidence=0.99, engine="test")


def _recovered_table() -> TableStructure:
    return TableStructure(rows=[["a"]], confidence=0.8)


def test_clean_confident_complete_page_is_not_escalated():
    cfg = load_config()
    decision = route_ocr_page(_confident_ocr(), _recovered_table(), _clean_completeness(), cfg)
    assert decision.need_vlm is False
    assert decision.reasons == []


def test_failed_table_recovery_alone_does_not_escalate():
    """Table recovery is a MEANS of getting fields; completeness measures the
    END. Most drawings here are title blocks, not bordered tables, and scoring
    grid regularity as a fallback trigger fired on 7 of 7 development pages —
    reproducing the 100% fallback rate this router exists to fix."""
    cfg = load_config()
    no_table = TableStructure(rows=[], confidence=0.0)
    decision = route_ocr_page(_confident_ocr(), no_table, _clean_completeness(), cfg)
    assert decision.need_vlm is False


def test_high_confidence_but_incomplete_page_is_escalated():
    """The silent-miss case: OCR is sure about what it returned and returned
    almost nothing. Confidence-only routing waves this straight through."""
    cfg = load_config()
    damaged = _clean_completeness(completeness=0.17, isolated_labels=["图号", "断路器编号"],
                                  watermark_repetition=0.57)
    decision = route_ocr_page(_confident_ocr(), _recovered_table(), damaged, cfg)
    assert decision.need_vlm is True
    assert "field_completeness_below_threshold" in decision.reasons
    assert "isolated_labels_without_values" in decision.reasons
    assert "repeated_boilerplate_text" in decision.reasons
    assert "ocr_confidence_below_threshold" not in decision.reasons


def test_low_confidence_still_escalates():
    cfg = load_config()
    blurred = OCRResult(text="", average_confidence=0.0, engine="test")
    decision = route_ocr_page(blurred, _recovered_table(), _clean_completeness(), cfg)
    assert decision.need_vlm is True
    assert "ocr_confidence_below_threshold" in decision.reasons


def test_unknown_drawing_type_escalates():
    cfg = load_config()
    unknown = _clean_completeness(drawing_type="unknown", type_source="none", completeness=0.0)
    decision = route_ocr_page(_confident_ocr(), _recovered_table(), unknown, cfg)
    assert decision.need_vlm is True
    assert "drawing_type_unknown" in decision.reasons


def test_uncertain_device_count_escalates_even_when_everything_found_looks_complete():
    cfg = load_config()
    group = _clean_completeness(drawing_type="fan_group", completeness=1.0,
                                group_device_count=3, group_cardinality_uncertain=True)
    decision = route_ocr_page(_confident_ocr(), _recovered_table(), group, cfg)
    assert decision.need_vlm is True
    assert decision.reasons == ["group_cardinality_uncertain"]


def test_invalid_field_value_escalates_even_when_every_other_signal_is_clean():
    """The round-3 gap: label present, value present, completeness 1.0, OCR
    confidence high — and the value is stamp text rather than an equipment
    code. Nothing before this signal notices."""
    from src.vision.schemas import ValueValidationResult
    cfg = load_config()
    validation = ValueValidationResult(
        checked_fields=["控制柜编号"], invalid_fields=["控制柜编号"], valid_ratio=0.0)
    decision = route_ocr_page(
        _confident_ocr(), _recovered_table(), _clean_completeness(), cfg, validation)
    assert decision.need_vlm is True
    assert decision.reasons == ["invalid_field_value"]


def test_validity_rate_trigger_is_separate_from_the_per_field_trigger():
    from src.vision.schemas import ValueValidationResult
    cfg = load_config()
    validation = ValueValidationResult(
        checked_fields=["a", "b", "c"], valid_fields=["a"],
        invalid_fields=["b", "c"], valid_ratio=1 / 3)
    decision = route_ocr_page(
        _confident_ocr(), _recovered_table(), _clean_completeness(), cfg, validation)
    assert "invalid_field_value" in decision.reasons
    assert "value_validity_below_threshold" in decision.reasons


def test_no_validation_supplied_leaves_the_decision_unchanged():
    """Callers that have not run value validation must not be handed a
    different routing verdict by accident."""
    cfg = load_config()
    assert route_ocr_page(
        _confident_ocr(), _recovered_table(), _clean_completeness(), cfg).need_vlm is False
