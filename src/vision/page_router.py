from __future__ import annotations

from typing import List, Optional

from src.vision.config import VisualFallbackConfig
from src.vision.schemas import (
    ImageQualityMetrics, PageParseResult, PageRouteDecision, Region, RegionType,
)


def _native_text_ratio(page: PageParseResult) -> float:
    total = len(page.native_text) + len(page.ocr_text)
    if total == 0:
        return 0.0
    return len(page.native_text) / total


def _has_drawing_signal(page: PageParseResult) -> bool:
    block_types = {b.block_type for b in page.layout_blocks}
    return "figure" in block_types or "title_block" in block_types or page.detected_figures > 0


def route_page(
    page: PageParseResult,
    cfg: VisualFallbackConfig,
    quality: Optional[ImageQualityMetrics] = None,
) -> PageRouteDecision:
    """Classify a page and decide whether it needs native extraction, OCR,
    layout recovery, preprocessing, and/or VLM fallback.

    Deterministic and config-driven: every threshold comes from cfg, nothing
    is a literal here. need_vlm is never decided by "page has an image" alone
    — it combines native-text ratio, image ratio, OCR confidence, garbled
    ratio, table-recovery failure, and drawing/title-block signals.

    `quality` is optional: a real ImageQualityAnalyzer measurement (Phase 3)
    that only exists once the page has already been rendered to an image. A
    page router call before that step (or a purely native-text page that was
    never rasterized) simply omits it, and need_preprocess stays False —
    preprocessing is never decided from page_type alone.
    """
    reasons: List[str] = []
    native_ratio = _native_text_ratio(page)
    has_drawing_signal = _has_drawing_signal(page)

    need_ocr = native_ratio < 0.5 and page.image_ratio > 0
    if need_ocr:
        reasons.append("low_native_text_ratio")

    is_engineering_drawing = (
        cfg.enable_engineering_drawing
        and has_drawing_signal
        and page.image_ratio >= cfg.min_image_ratio
    )
    if is_engineering_drawing:
        reasons.append("engineering_drawing_detected")

    is_complex_table = (
        cfg.enable_complex_table
        and (page.detected_tables > 0 and page.table_structure_recovery_failed)
    )
    if is_complex_table:
        reasons.append("complex_table_detected")

    low_ocr_confidence = page.ocr_text != "" and page.ocr_confidence < cfg.min_ocr_confidence
    if low_ocr_confidence:
        reasons.append("ocr_confidence_below_threshold")

    high_garbled = page.garbled_ratio > cfg.max_garbled_ratio
    if high_garbled:
        reasons.append("garbled_text_ratio_above_threshold")

    if is_engineering_drawing:
        page_type = "engineering_drawing"
    elif is_complex_table:
        page_type = "complex_table"
    elif native_ratio >= 0.9 and page.image_ratio < cfg.min_image_ratio:
        page_type = "native_text"
    elif need_ocr and page.image_ratio >= cfg.min_image_ratio:
        page_type = "scanned_text"
    else:
        page_type = "mixed"

    # Native-text pages never need VLM: OCR/text extraction is already
    # reliable there, and VLM is reserved for pages where structure would
    # otherwise be lost.
    need_vlm = page_type != "native_text" and (
        is_engineering_drawing or is_complex_table or low_ocr_confidence or high_garbled
    )

    # native text extraction runs whenever there is native text worth taking,
    # which includes "mixed" pages (native text + local OCR), not just pure
    # native_text pages.
    need_native_extract = page_type in {"native_text", "mixed"} and native_ratio > 0
    need_layout = page_type in {"engineering_drawing", "complex_table", "mixed"}

    need_preprocess = False
    if quality is not None and need_ocr:
        pp = cfg.preprocess
        if quality.sharpness < pp.min_sharpness_for_skip_denoise:
            need_preprocess = True
            reasons.append("low_sharpness_needs_denoise")
        if quality.contrast < pp.min_contrast_for_skip_binarize:
            need_preprocess = True
            reasons.append("low_contrast_needs_binarize")
        if abs(quality.skew_angle_deg) > pp.skew_angle_threshold_deg:
            need_preprocess = True
            reasons.append("skew_angle_above_threshold")
        if quality.brightness_uniformity < pp.min_brightness_uniformity:
            need_preprocess = True
            reasons.append("uneven_illumination_needs_shadow_remove")

    return PageRouteDecision(
        page_type=page_type,
        need_native_extract=need_native_extract,
        need_ocr=need_ocr,
        need_layout=need_layout,
        need_preprocess=need_preprocess,
        need_vlm=need_vlm,
        vlm_regions=[],
        reason=reasons,
    )


# Only these region types are ever eligible for VLM calls; a legend or a
# generic annotation region is left to OCR/text unless explicitly listed.
_VLM_ELIGIBLE_REGION_TYPES = {
    RegionType.TITLE_BLOCK,
    RegionType.PARAMETER_TABLE,
    RegionType.MAIN_DRAWING,
    RegionType.LOW_CONFIDENCE_OCR_REGION,
}


def route_regions(regions: List[Region], cfg: VisualFallbackConfig) -> List[str]:
    """Return the region_ids that should be sent to the VLM.

    A region only qualifies if its type is one the VLM is meant to help with
    AND (its OCR confidence is below threshold OR its type inherently needs
    visual/structural understanding OCR can't recover, e.g. a parameter
    table or title block).
    """
    selected: List[str] = []
    for region in regions:
        if region.region_type not in _VLM_ELIGIBLE_REGION_TYPES:
            continue
        needs_structural_read = region.region_type in {
            RegionType.TITLE_BLOCK,
            RegionType.PARAMETER_TABLE,
            RegionType.MAIN_DRAWING,
        }
        low_confidence = region.ocr_confidence < cfg.min_ocr_confidence
        if needs_structural_read or low_confidence:
            selected.append(region.region_id)
    return selected
