from src.vision.chunk_builder import build_chunks
from src.vision.schemas import (
    DeviceEntity, DeviceParameter, DrawingMetadata, DrawingParseResult, TableStructure, VisualRelation,
)


def _sample_result() -> DrawingParseResult:
    return DrawingParseResult(
        document_id="doc1", page_number=3,
        drawing_metadata=DrawingMetadata(drawing_id="FAN-A13-02", revision="V1", station_id="A"),
        devices=[DeviceEntity(device_id="A13", device_name="排风机", parameters=[
            DeviceParameter(name="power", raw_name="功率", raw_text="45kW", value=45, unit="kW"),
        ])],
        relations=[VisualRelation(source_id="A13", relation="connected_to", target_id="CTRL-01", confidence=0.8)],
    )


def test_produces_one_chunk_per_content_type_present():
    chunks = build_chunks(_sample_result())
    content_types = {c.content_type for c in chunks}
    assert content_types == {"drawing_metadata", "drawing_parameter", "drawing_relation"}


def test_chunks_carry_traceability_metadata():
    chunks = build_chunks(_sample_result())
    for c in chunks:
        assert c.document_id == "doc1"
        assert c.page_number == 3
        assert c.parent_id == "doc1:p3"
        assert c.metadata["document_id"] == "doc1"


def test_metadata_chunk_text_is_natural_language_not_raw_json():
    chunks = build_chunks(_sample_result())
    meta_chunk = next(c for c in chunks if c.content_type == "drawing_metadata")
    assert "{" not in meta_chunk.text
    assert "FAN-A13-02" in meta_chunk.text


def test_build_text_chunk_wraps_plain_page_text():
    from src.vision.chunk_builder import build_text_chunk
    chunks = build_text_chunk("这是一段原生文本页面内容。", document_id="d1", page_number=3)
    assert len(chunks) == 1
    assert chunks[0].content_type == "text"
    assert chunks[0].page_number == 3
    assert "原生文本" in chunks[0].text


def test_build_text_chunk_skips_blank_text():
    from src.vision.chunk_builder import build_text_chunk
    assert build_text_chunk("   ", document_id="d1", page_number=1) == []


def test_build_regulation_chunks_one_per_clause_with_location_metadata():
    from src.vision.chunk_builder import build_regulation_chunks
    from src.vision.regulation_parser import parse_regulation_text
    meta = parse_regulation_text("第一章 总则\n第一条 本规程适用范围。\n第二条 维护记录要求。\n")
    chunks = build_regulation_chunks(meta, document_id="reg1")
    assert len(chunks) == 3  # chapter heading line + 2 articles all become clauses
    article_chunks = [c for c in chunks if c.metadata.get("article")]
    assert len(article_chunks) == 2
    assert all(c.content_type == "article" for c in chunks)
    assert article_chunks[0].metadata["article"] == "第一条"


def test_build_table_row_chunks_uses_headers_when_available():
    from src.vision.chunk_builder import build_table_row_chunks
    table = TableStructure(headers=["设备编号", "功率"], rows=[["A12", "45kW"], ["A13", "55kW"]], confidence=0.9)
    chunks = build_table_row_chunks(table, document_id="d1", page_number=2, table_id="t1")
    assert len(chunks) == 2
    assert "设备编号：A12" in chunks[0].text
    assert "功率：45kW" in chunks[0].text
    assert chunks[0].metadata["row_index"] == 0
    assert chunks[1].metadata["row_index"] == 1


def test_build_table_row_chunks_without_headers_joins_cell_values():
    from src.vision.chunk_builder import build_table_row_chunks
    table = TableStructure(headers=[], rows=[["图号", "FAN-A13-02"]], confidence=0.8)
    chunks = build_table_row_chunks(table, document_id="d1", page_number=2)
    assert "图号" in chunks[0].text and "FAN-A13-02" in chunks[0].text


def test_build_table_row_chunks_skips_empty_rows():
    from src.vision.chunk_builder import build_table_row_chunks
    table = TableStructure(headers=[], rows=[["", ""], ["a", "b"]], confidence=0.8)
    chunks = build_table_row_chunks(table, document_id="d1", page_number=1)
    assert len(chunks) == 1


def test_empty_result_produces_no_chunks():
    result = DrawingParseResult(document_id="doc2", page_number=1)
    assert build_chunks(result) == []
