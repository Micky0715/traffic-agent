"""Structured chunking.

All inputs here are SYNTHETIC TEST FIXTURES constructed inline. They are not
enterprise data, are not derived from any system output, and are never used to
compute an effectiveness metric — only to pin structural behaviour.
"""
from __future__ import annotations

import pytest

from src.rag.chunking import (
    assess_continuation, build_clause_chunks, build_drawing_field_chunks,
    build_section_chunks, build_table_chunks, looks_like_repeated_header,
    stable_chunk_id,
)
from src.rag.config import ChunkingConfig, load_rag_config
from src.vision.schemas import (
    DeviceEntity, DeviceParameter, DrawingMetadata, DrawingParseResult,
    RegulationClause, RegulationMetadata, TableStructure, ValidationResult,
    VisualEvidence,
)

SYNTHETIC_TEST_FIXTURE = True   # nothing in this module is real data


@pytest.fixture
def cfg() -> ChunkingConfig:
    return load_rag_config().chunking


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def test_two_sections_never_merge_across_a_heading(cfg):
    chunks = build_section_chunks("doc1", [
        {"section_path": ["第4章 通风", "4.1 概述"], "text": "本章说明通风系统。", "page": 1},
        {"section_path": ["第4章 通风", "4.2 风机"], "text": "风机应定期检修。", "page": 1},
    ], cfg)
    assert len(chunks) == 2
    assert chunks[0].section_path != chunks[1].section_path
    assert "4.2" not in chunks[0].text


def test_long_section_splits_at_sentence_boundaries(cfg):
    small = ChunkingConfig(section_max_tokens=40, section_overlap_tokens=0)
    body = "".join(f"第{i}句内容说明设备维护要求。" for i in range(10))
    chunks = build_section_chunks("doc1", [
        {"section_path": ["第4章"], "text": body, "page": 2}], small)
    assert len(chunks) > 1
    for chunk in chunks:
        # A split mid-sentence would leave a fragment that reads as a different
        # claim than the sentence it came from.
        assert chunk.text.rstrip().endswith("。")
        assert chunk.section_path == ["第4章"]   # trail repeated into every piece


def test_every_section_piece_keeps_the_full_heading_trail(cfg):
    small = ChunkingConfig(section_max_tokens=30, section_overlap_tokens=0)
    chunks = build_section_chunks("doc1", [
        {"section_path": ["第4章 通风", "4.2 风机"],
         "text": "".join(f"要求{i}必须执行。" for i in range(8)), "page": 1}], small)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section_path == ["第4章 通风", "4.2 风机"]
        assert "4.2 风机" in chunk.text    # searchable, not just metadata


# --------------------------------------------------------------------------
# clauses
# --------------------------------------------------------------------------

def test_clause_number_and_body_stay_in_one_chunk():
    metadata = RegulationMetadata(document_name="JTG D81-2017", clauses=[
        RegulationClause(chapter="第4章", article="4.2.1",
                         text="应急照明应保证不小于30分钟的持续供电。", line_number=12),
        RegulationClause(chapter="第4章", article="4.2.2",
                         text="疏散指示标志应连续设置。", line_number=13),
    ])
    chunks = build_clause_chunks("reg1", metadata, regulation_code="JTG D81-2017")
    assert len(chunks) == 2
    first = chunks[0]
    assert "4.2.1" in first.text and "30分钟" in first.text
    assert "4.2.2" not in first.text
    assert first.metadata["regulation_code"] == "JTG D81-2017"


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def two_level_table() -> TableStructure:
    return TableStructure(
        headers=[],
        rows=[
            ["设备", "参数", "", "备注"],       # level 1; blanks = merged span
            ["", "功率", "风量", ""],            # level 2
            ["A16", "45kW", "28000m3/h", "常用"],
            ["A17", "55kW", "32000m3/h", "备用"],
            ["A18", "37kW", "24000m3/h", "常用"],
        ],
        bbox=[10.0, 20.0, 700.0, 400.0], confidence=0.8)


def test_two_level_headers_expand_into_header_paths():
    chunks = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                                header_row_count=2, table_title="风机参数表")
    parent = chunks[0]
    assert parent.content_type == "table_parent"
    assert ["参数", "功率"] in parent.header_paths
    assert ["参数", "风量"] in parent.header_paths


def test_merged_header_cell_is_inherited_by_every_column_under_it():
    """The blank in row 1 under 风量 is a merged 参数 cell. Without inheritance
    the second column under it loses its parent and becomes just "风量"."""
    chunks = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                                header_row_count=2)
    paths = chunks[0].header_paths
    assert paths[1][:1] == ["参数"] and paths[2][:1] == ["参数"]


def test_each_device_row_becomes_its_own_child_chunk():
    chunks = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                                header_row_count=2, table_title="风机参数表")
    rows = [c for c in chunks if c.content_type == "table_row"]
    assert len(rows) == 3
    assert {c.entity_id for c in rows} == {"A16", "A17", "A18"}


def test_one_device_values_never_appear_in_another_devices_chunk():
    chunks = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                                header_row_count=2)
    rows = {c.entity_id: c.text for c in chunks if c.content_type == "table_row"}
    assert "45kW" in rows["A16"] and "55kW" not in rows["A16"]
    assert "55kW" in rows["A17"] and "45kW" not in rows["A17"]


def test_row_text_names_the_column_not_just_the_value():
    """"A16 45kW 28000m3/h" cannot answer "A16 的功率" — nothing in it says
    which number is the power."""
    chunks = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                                header_row_count=2)
    a16 = next(c for c in chunks if c.entity_id == "A16")
    assert "参数.功率=45kW" in a16.text
    assert "参数.风量=28000m3/h" in a16.text


def test_vertically_merged_entity_column_is_carried_down():
    table = TableStructure(headers=[], rows=[
        ["设备", "参数"],
        ["A16", "45kW"],
        ["", "28000m3/h"],      # same device, merged entity cell
    ], confidence=0.7)
    chunks = build_table_chunks("doc1", table, table_id="T2", page=1, header_row_count=1)
    rows = [c for c in chunks if c.content_type == "table_row"]
    assert [c.entity_id for c in rows] == ["A16", "A16"]


def test_a_repeated_header_row_is_recognised_as_a_header_not_data():
    header = ["设备", "参数", "备注"]
    assert looks_like_repeated_header(["设备", "参数", "备注"], [header])
    assert not looks_like_repeated_header(["A16", "45kW", "常用"], [header])


# --------------------------------------------------------------------------
# cross-page continuation
# --------------------------------------------------------------------------

def base_page(page: int, **overrides):
    payload = {"page": page, "headers": ["设备", "功率"], "column_count": 2,
               "section_path": ["第4章"], "continuation_marker": False,
               "unterminated": False}
    payload.update(overrides)
    return payload


def test_matching_column_count_alone_does_not_merge_two_tables(cfg):
    """Two unrelated 2-column tables on facing pages would otherwise merge,
    inventing rows that belong to a different table."""
    result = assess_continuation(
        base_page(1, headers=["设备", "功率"], section_path=["第4章"]),
        base_page(2, headers=["站点", "编号"], section_path=["第7章"]), cfg)
    assert result["merged"] is False
    assert "not_header_compatible" in result["reasons"]


def test_non_adjacent_pages_never_merge(cfg):
    result = assess_continuation(base_page(1), base_page(5), cfg)
    assert result["merged"] is False
    assert "not_adjacent_pages" in result["reasons"]


def test_merge_requires_a_supporting_signal_beyond_shape(cfg):
    """Header and column count match and the pages are adjacent, but nothing
    says the table actually continued. Left separate and flagged."""
    result = assess_continuation(
        base_page(1, section_path=["第4章"]),
        base_page(2, section_path=["第9章"]), cfg)
    assert result["merged"] is False
    assert result["continuation_uncertain"] is True


def test_merge_happens_when_an_explicit_continuation_marker_is_present(cfg):
    result = assess_continuation(
        base_page(1, unterminated=True),
        base_page(2, continuation_marker=True), cfg)
    assert result["merged"] is True


def test_uncertain_continuation_is_marked_on_the_chunk():
    chunks = build_table_chunks("doc1", two_level_table(), table_id="T1", page=4,
                                header_row_count=2, continuation_uncertain=True)
    assert all(c.structure_uncertain for c in chunks)


# --------------------------------------------------------------------------
# drawings, provenance, ids
# --------------------------------------------------------------------------

def drawing_result() -> DrawingParseResult:
    return DrawingParseResult(
        document_id="drawing_001", page_number=3,
        drawing_metadata=DrawingMetadata(drawing_id="FAN-A16-01"),
        devices=[DeviceEntity(device_id="A16", parameters=[
            DeviceParameter(name="power", raw_name="功率", raw_text="45kW",
                            value=45.0, unit="kW")])],
        evidence=[VisualEvidence(field="power", value="45kW", page=3,
                                 bbox=[120, 340, 930, 410], region_id="r1",
                                 source="vlm", confidence=0.9)],
        validation=ValidationResult(status="confirmed"))


def test_drawing_field_chunk_carries_entity_field_and_bbox():
    chunks = build_drawing_field_chunks(drawing_result())
    chunk = chunks[0]
    assert chunk.entity_id == "A16"
    assert chunk.field_name == "功率" and chunk.field_value == "45kW"
    assert chunk.page_start == 3
    assert chunk.source_bboxes == [[120, 340, 930, 410]]
    assert "图号 FAN-A16-01" in chunk.text     # drawing number repeated in


def test_vlm_output_is_not_labelled_as_ocr():
    """A generated value presented as OCR原文 would make a model's guess look
    like something read off the page."""
    chunks = build_drawing_field_chunks(drawing_result(), parser_source="vlm")
    assert all(c.parser_source == "vlm" for c in chunks)
    ocr_chunks = build_drawing_field_chunks(drawing_result(), parser_source="ocr")
    assert all(c.parser_source == "ocr" for c in ocr_chunks)
    assert chunks[0].chunk_id == ocr_chunks[0].chunk_id  # id keys on content


def test_chunk_ids_are_stable_across_rebuilds():
    """A random UUID would break every stored reference on each re-ingest and
    make two runs undiffable."""
    first = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                               header_row_count=2)
    second = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                                header_row_count=2)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert all(c.chunk_id for c in first)


def test_chunk_id_changes_when_content_changes():
    assert (stable_chunk_id("d", "table_row", "T1.0", "A16 45kW")
            != stable_chunk_id("d", "table_row", "T1.0", "A16 55kW"))


def test_every_chunk_can_be_traced_back_to_a_page():
    chunks = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                                header_row_count=2)
    chunks += build_drawing_field_chunks(drawing_result())
    for chunk in chunks:
        assert chunk.document_id and chunk.page_start is not None
        assert chunk.source_bboxes, chunk.chunk_id


def test_merged_header_does_not_leak_into_the_next_top_level_column():
    """Found by running the real pipeline: the 备注 column came out as
    备注.风量, because a blank sub-header inherited the value to its left
    across a top-level boundary. A query for 风量 would then match a remarks
    cell."""
    chunks = build_table_chunks("doc1", two_level_table(), table_id="T1", page=3,
                                header_row_count=2)
    paths = chunks[0].header_paths
    assert paths[3] == ["备注"], paths
    a16 = next(c for c in chunks if c.entity_id == "A16")
    assert "备注.风量" not in a16.text
    assert "备注=常用" in a16.text
