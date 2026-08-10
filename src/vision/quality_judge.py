from __future__ import annotations

from typing import Dict, List, Optional

from src.vision.schemas import DocumentQuality, DrawingParseResult, PageParseResult, RegulationMetadata


class DocumentQualityJudge:
    """Pure rule-based aggregator — no model calls. Combines signals already
    produced by other real modules (PageParseResult.ocr_confidence/
    garbled_ratio, VisualParseValidator's conflict/incomplete status,
    RegulationMetadata's article-gap warnings) into one DocumentQuality
    summary. Quality is judged on more than OCR confidence alone, per the
    plan's explicit requirement.
    """

    # An escalation threshold, not a single hard-error trigger: several
    # simultaneously-minor issues (a bit of OCR noise here, one suspected
    # device there, a page gap) can add up to something that needs human
    # review just as much as one hard conflict does. An earlier version of
    # this judge only escalated to "error" on conflict_count>0, so 1 warning
    # and 5 stacked warnings both silently reported the same "warning"
    # status — exactly the predicted bad case in the plan (#7); this
    # threshold is the fix, not an afterthought.
    _STACKED_WARNING_ESCALATION_THRESHOLD = 3

    def judge(
        self,
        pages: Optional[List[PageParseResult]] = None,
        drawing_results: Optional[List[DrawingParseResult]] = None,
        regulation_metadata: Optional[RegulationMetadata] = None,
        table_completeness_scores: Optional[List[float]] = None,
    ) -> DocumentQuality:
        pages = pages or []
        drawing_results = drawing_results or []
        table_completeness_scores = table_completeness_scores or []
        warnings: List[Dict[str, str]] = []

        ocr_confidences = [p.ocr_confidence for p in pages if p.ocr_text]
        avg_ocr_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 1.0
        if ocr_confidences and avg_ocr_confidence < 0.6:
            warnings.append({"type": "low_ocr_confidence", "message": f"平均OCR置信度仅{avg_ocr_confidence:.2f}"})

        garbled_ratios = [p.garbled_ratio for p in pages]
        avg_garbled_ratio = sum(garbled_ratios) / len(garbled_ratios) if garbled_ratios else 0.0
        if avg_garbled_ratio > 0.2:
            warnings.append({"type": "high_garbled_ratio", "message": f"平均乱码比例达{avg_garbled_ratio:.2f}"})

        conflict_count = sum(1 for r in drawing_results if r.validation.status == "conflict")
        incomplete_results = [r for r in drawing_results if r.validation.status == "incomplete"]
        low_confidence_count = sum(1 for r in drawing_results if r.validation.status == "suspected")
        missing_field_count = sum(len(r.validation.warnings) for r in incomplete_results)

        if conflict_count:
            warnings.append({"type": "conflict", "message": f"{conflict_count} 处结果存在无法自动判定的冲突"})
        if incomplete_results:
            warnings.append({"type": "incomplete", "message": f"{len(incomplete_results)} 处结果缺少预期字段"})

        table_completeness = (
            sum(table_completeness_scores) / len(table_completeness_scores) if table_completeness_scores else 1.0
        )
        if table_completeness < 0.8:
            warnings.append({"type": "low_table_completeness", "message": f"表格结构恢复完整度仅{table_completeness:.2f}"})

        page_numbers = sorted(p.page_number for p in pages)
        page_continuity = all(b - a == 1 for a, b in zip(page_numbers, page_numbers[1:])) if len(page_numbers) > 1 else True
        if not page_continuity:
            warnings.append({"type": "page_gap", "message": f"页码不连续：{page_numbers}"})

        article_continuity = True
        if regulation_metadata is not None and regulation_metadata.warnings:
            article_continuity = False
            for w in regulation_metadata.warnings:
                warnings.append({"type": "article_gap", "message": w})

        if conflict_count > 0 or len(warnings) >= self._STACKED_WARNING_ESCALATION_THRESHOLD:
            status = "error"
        elif warnings:
            status = "warning"
        else:
            status = "ok"

        return DocumentQuality(
            status=status,
            ocr_confidence=avg_ocr_confidence,
            garbled_ratio=avg_garbled_ratio,
            missing_field_count=missing_field_count,
            table_completeness=table_completeness,
            page_continuity=page_continuity,
            article_continuity=article_continuity,
            conflict_count=conflict_count,
            low_confidence_count=low_confidence_count,
            warnings=warnings,
        )
