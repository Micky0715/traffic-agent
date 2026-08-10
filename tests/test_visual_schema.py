import pytest
from pydantic import ValidationError

from src.vision.schemas import (
    DeviceEntity, DeviceParameter, DrawingMetadata, DrawingParseResult,
    DocumentChunk, VisualRelation,
)


def test_drawing_metadata_all_fields_optional():
    meta = DrawingMetadata()
    assert meta.drawing_id is None
    assert meta.revision is None


def test_drawing_parse_result_defaults_are_empty_not_none():
    result = DrawingParseResult(document_id="d1", page_number=1)
    assert result.devices == []
    assert result.relations == []
    assert result.validation.status == "confirmed"


def test_device_parameter_requires_name_raw_name_raw_text():
    with pytest.raises(ValidationError):
        DeviceParameter(name="power")  # missing raw_name/raw_text


def test_device_entity_nests_parameters():
    dev = DeviceEntity(device_id="A12", parameters=[
        DeviceParameter(name="power", raw_name="功率", raw_text="45kW", value=45, unit="kW"),
    ])
    assert dev.parameters[0].unit == "kW"


def test_visual_relation_confidence_bounds():
    with pytest.raises(ValidationError):
        VisualRelation(source_id="A", relation="connected_to", target_id="B", confidence=1.5)


def test_document_chunk_content_type_is_restricted():
    with pytest.raises(ValidationError):
        DocumentChunk(
            text="x", parent_id="p", page_number=1, document_id="d",
            content_type="not_a_real_type",  # type: ignore[arg-type]
        )
