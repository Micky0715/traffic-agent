from src.vision.quality_judge import DocumentQualityJudge
from src.vision.schemas import DrawingParseResult, PageParseResult, RegulationMetadata, ValidationResult

JUDGE = DocumentQualityJudge()


def _page(page_number: int, ocr_confidence=1.0, garbled_ratio=0.0, ocr_text="x") -> PageParseResult:
    return PageParseResult(page_number=page_number, ocr_confidence=ocr_confidence, garbled_ratio=garbled_ratio, ocr_text=ocr_text)


def _drawing_result(status: str, warnings=None) -> DrawingParseResult:
    return DrawingParseResult(
        document_id="d", page_number=1,
        validation=ValidationResult(status=status, warnings=warnings or []),
    )


def test_clean_input_is_ok():
    quality = JUDGE.judge(pages=[_page(1), _page(2)])
    assert quality.status == "ok"
    assert quality.warnings == []


def test_low_ocr_confidence_triggers_warning():
    quality = JUDGE.judge(pages=[_page(1, ocr_confidence=0.4)])
    assert quality.status == "warning"
    assert any(w["type"] == "low_ocr_confidence" for w in quality.warnings)


def test_conflict_status_escalates_to_error_even_alone():
    quality = JUDGE.judge(drawing_results=[_drawing_result("conflict")])
    assert quality.status == "error"
    assert quality.conflict_count == 1


def test_incomplete_status_is_warning_not_error():
    quality = JUDGE.judge(drawing_results=[_drawing_result("incomplete", ["缺少字段 air_volume"])])
    assert quality.status == "warning"
    assert quality.missing_field_count == 1


def test_page_gap_is_detected():
    quality = JUDGE.judge(pages=[_page(1), _page(3)])
    assert quality.page_continuity is False
    assert any(w["type"] == "page_gap" for w in quality.warnings)


def test_regulation_article_gap_propagates_into_quality_warnings():
    reg_meta = RegulationMetadata(warnings=["第十二条后直接出现第十四条，中间条款可能缺失或编号不连续"])
    quality = JUDGE.judge(regulation_metadata=reg_meta)
    assert quality.article_continuity is False
    assert any(w["type"] == "article_gap" for w in quality.warnings)


def test_single_minor_warning_stays_warning_not_error():
    quality = JUDGE.judge(pages=[_page(1, garbled_ratio=0.3)])
    assert quality.status == "warning"


def test_several_stacked_minor_issues_escalate_to_error():
    """Predicted bad case #7 from the plan, confirmed and fixed in this
    Phase: a document with several simultaneously-minor problems (low OCR
    confidence, high garbled ratio, an incomplete extraction, a page gap)
    needs human review just as much as one hard conflict — an earlier
    version of this judge silently reported the same 'warning' status for
    one issue and four stacked issues alike."""
    quality = JUDGE.judge(
        # both pages carry the same low confidence / high garbled ratio so
        # the average actually crosses both thresholds, plus a page gap
        # (1 -> 3) and an incomplete extraction: four independently-minor
        # signals stacked together.
        pages=[_page(1, ocr_confidence=0.5, garbled_ratio=0.25), _page(3, ocr_confidence=0.5, garbled_ratio=0.25)],
        drawing_results=[_drawing_result("incomplete", ["缺少字段 air_volume"])],
    )
    assert len(quality.warnings) >= 3
    assert quality.status == "error"


def test_no_regulation_metadata_means_article_continuity_defaults_true():
    quality = JUDGE.judge()
    assert quality.article_continuity is True
    assert quality.status == "ok"
