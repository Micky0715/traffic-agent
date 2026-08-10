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
