import pytest
from pydantic import ValidationError

from src.vision.schemas import (
    DocumentChunk, DocumentQuality, ImageQualityMetrics, OCRBlock, OCRResult,
    ParseEvidence, PreprocessDecision, ProcessedImage, RegulationClause,
    RegulationMetadata, TableStructure,
)


def test_ocr_result_defaults_are_empty_not_none():
    result = OCRResult(text="", average_confidence=0.0, engine="mock")
    assert result.blocks == []
    assert result.engine_available is True
    assert result.error is None


def test_ocr_result_can_record_engine_unavailable():
    result = OCRResult(text="", engine="paddleocr", engine_available=False, error="oneDNN crash")
    assert result.engine_available is False
    assert "oneDNN" in result.error


def test_ocr_block_confidence_bounds():
    with pytest.raises(ValidationError):
        OCRBlock(text="x", confidence=1.2)


def test_image_quality_metrics_default_construction():
    metrics = ImageQualityMetrics()
    assert metrics.sharpness == 0.0
    assert metrics.brightness_uniformity == 1.0


def test_preprocess_decision_default_is_no_preprocess():
    decision = PreprocessDecision()
    assert decision.need_preprocess is False
    assert decision.operations == []


def test_processed_image_never_shares_original_and_processed_path_by_default():
    img = ProcessedImage(original_path="a.png", processed_path="a.processed.png", operations_applied=["denoise"])
    assert img.original_path != img.processed_path


def test_table_structure_confidence_bounds():
    with pytest.raises(ValidationError):
        TableStructure(confidence=-0.1)


def test_regulation_clause_all_location_fields_optional():
    clause = RegulationClause(text="应急照明供电应具备独立或备用电源")
    assert clause.article is None
    assert clause.text


def test_regulation_metadata_collects_clauses_and_warnings():
    meta = RegulationMetadata(document_name="JTG D81-2017", clauses=[
        RegulationClause(article="第十二条", text="..."),
        RegulationClause(article="第十四条", text="..."),
    ], warnings=["第十二条后直接出现第十四条"])
    assert len(meta.clauses) == 2
    assert meta.warnings


def test_parse_evidence_source_type_is_restricted():
    with pytest.raises(ValidationError):
        ParseEvidence(field="drawing_id", value="FAN-A12-01", page=1, source="made_up_source")  # type: ignore[arg-type]


def test_parse_evidence_accepts_all_documented_source_types():
    for source in ["native_pdf", "ocr_original", "ocr_processed", "vlm", "rule", "validator", "ledger"]:
        ParseEvidence(field="x", value="y", page=1, source=source)  # type: ignore[arg-type]


def test_document_quality_default_status_is_ok():
    quality = DocumentQuality()
    assert quality.status == "ok"
    assert quality.conflict_count == 0


def test_document_quality_rejects_unknown_status():
    with pytest.raises(ValidationError):
        DocumentQuality(status="critical")  # type: ignore[arg-type]


def test_document_chunk_accepts_new_ocr_pipeline_content_types():
    for content_type in ["text", "article", "table_row"]:
        DocumentChunk(text="x", parent_id="p", page_number=1, document_id="d", content_type=content_type)  # type: ignore[arg-type]
