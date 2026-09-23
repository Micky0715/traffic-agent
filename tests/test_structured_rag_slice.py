"""The structured-BM25 slice, end to end over REAL parsed output.

Corpus: data/ocr_fixtures/*.json — a replay of a recorded real PaddleOCR 3.7.0
run, each file stamped with image sha256, library/model versions and device.
Real recognition output, replayed; not live inference and not hand-written
mock. Both halves are asserted below so neither can be overstated later.

Nothing here reads a case id or a file name to decide an answer.
"""
from __future__ import annotations

import pytest

from src.rag.config import load_rag_config
from src.rag.eligibility import evaluate_candidate_eligibility
from src.rag.ingest import build_chunks_from_real_parsed_documents, entity_document_index
from src.rag.query import parse_query, resolve_entity
from src.rag.retrieval import BM25Retriever, tokenize
from src.rag.service import StructuredRagService
from src.vision.config import load_config as load_vision_config


@pytest.fixture(scope="module")
def service() -> StructuredRagService:
    svc = StructuredRagService()
    svc.load_corpus()
    return svc


@pytest.fixture(scope="module")
def cfg():
    return load_rag_config()


# --------------------------------------------------------------------------
# corpus from real parsed output
# --------------------------------------------------------------------------

def test_corpus_is_built_from_real_parsed_documents(service):
    assert service.stats.real_documents >= 9
    assert service.stats.real_chunks > 0
    assert service.stats.synthetic_chunks == 0    # counted apart, and none here
    assert service.stats.mock_chunks == 0


def test_every_field_chunk_keeps_its_page_and_bbox(service):
    fields = [c for c in service.chunks if c.content_type == "drawing_field"]
    assert fields
    for chunk in fields:
        assert chunk.page_start is not None
        assert chunk.source_bboxes and len(chunk.source_bboxes[0]) == 4


def test_chunk_ids_are_stable_across_rebuilds(cfg):
    vision_cfg = load_vision_config()
    first, _ = build_chunks_from_real_parsed_documents(vision_cfg, cfg.chunking)
    second, _ = build_chunks_from_real_parsed_documents(vision_cfg, cfg.chunking)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert all(c.chunk_id for c in first)


def test_multi_device_drawing_keeps_devices_apart(service):
    rows = [c for c in service.chunks
            if c.document_id == "FAN-MULTI-01" and c.entity_id]
    by_entity = {}
    for chunk in rows:
        by_entity.setdefault(chunk.entity_id, []).append(chunk.text)
    assert len(by_entity) >= 3
    for entity, texts in by_entity.items():
        for other in by_entity:
            if other != entity:
                assert not any(other in text for text in texts), (entity, other)


def test_raw_and_normalized_values_are_both_kept(service):
    chunk = next(c for c in service.chunks if c.field_name == "图号")
    assert chunk.field_value                       # raw, as read
    assert chunk.metadata["normalized_value"]      # canonical form


def test_parser_source_says_ocr_not_vlm(service):
    """These fixtures ARE OCR output. Labelling a model's reading as text read
    off the page is the one provenance error nothing downstream can undo."""
    assert {c.parser_source for c in service.chunks} == {"ocr"}


def test_provenance_of_the_real_fixtures_is_carried(service):
    chunk = next(c for c in service.chunks if c.content_type == "drawing_field")
    provenance = chunk.metadata["provenance"]
    assert provenance["paddleocr_version"] == "3.7.0"
    assert provenance["image_sha256"]
    assert chunk.metadata["ocr_engine"] == "paddleocr-fixture"   # replay, not live


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

def test_exact_device_code_is_retrieved(service):
    hits = service.retriever.search("FAN-A13-02", 10)
    assert hits
    assert any(h.chunk.document_id == "FAN-A13-02" for h in hits)


def test_retrieval_is_reproducible(service):
    first = [(h.chunk.chunk_id, round(h.score, 9))
             for h in service.retriever.search("电机编号 M-13", 10)]
    second = [(h.chunk.chunk_id, round(h.score, 9))
              for h in service.retriever.search("电机编号 M-13", 10)]
    assert first == second


def test_stopwords_stop_a_function_word_only_query(cfg, service):
    """Root cause of "an unrelated query still returns three results": CJK
    tokenizes per character and 的/不 are shared with almost any Chinese text."""
    assert tokenize("完全不存在的查询词组合", cfg.retrieval.stopwords) != \
        tokenize("完全不存在的查询词组合")
    assert service.retriever.search("完全不存在的查询词组合", 10) == []


def test_no_score_threshold_is_applied(cfg):
    """A floor would need calibration on a dev set. It is configured as
    uncalibrated and deliberately not used."""
    assert cfg.retrieval.uncalibrated_min_bm25_score is None
    assert cfg.retrieval.uncalibrated_status == "uncalibrated"


# --------------------------------------------------------------------------
# eligibility: retrieved != evidence
# --------------------------------------------------------------------------

def test_a_high_scoring_chunk_for_the_wrong_device_is_not_evidence(service, cfg):
    plan = parse_query("查询A13风机接线图的电机编号", cfg.query_parsing)
    other = next(c for c in service.chunks
                 if c.document_id == "FAN-A25-01" and c.field_name == "电机编号")
    result = evaluate_candidate_eligibility(
        plan, other, resolved_entity="A13风机接线图",
        resolved_documents=["FAN-A13-02"])
    assert not result.eligible
    assert "ENTITY_MISMATCH" in result.reasons


def test_a_chunk_for_a_different_field_is_not_evidence(service, cfg):
    plan = parse_query("查询A13风机接线图的电机编号", cfg.query_parsing)
    wrong_field = next(c for c in service.chunks
                       if c.document_id == "FAN-A13-02" and c.field_name == "页码")
    result = evaluate_candidate_eligibility(
        plan, wrong_field, resolved_documents=["FAN-A13-02"])
    assert not result.eligible
    assert "FIELD_NOT_COVERED" in result.reasons


def test_an_invalid_value_is_never_eligible(service, cfg):
    """Real data: FAN-A19-01's cabinet number is stamp-corrupted, and the
    frozen value validation already marked it invalid."""
    plan = parse_query("查询A19风机接线图的控制柜编号", cfg.query_parsing)
    chunk = next(c for c in service.chunks
                 if c.document_id == "FAN-A19-01" and c.field_name == "控制柜编号")
    assert chunk.metadata["validation_status"] == "invalid"
    result = evaluate_candidate_eligibility(
        plan, chunk, resolved_documents=["FAN-A19-01"])
    assert not result.eligible
    assert "FIELD_VALUE_INVALID" in result.reasons


def test_retrieved_and_eligible_are_counted_separately(service):
    result = service.answer("查询A99风机的电机编号")
    assert result.retrieved_candidate_count > 0     # BM25 returned its top K
    assert result.eligible_evidence_count == 0      # none of it is evidence
    assert result.decision["decision"] == "abstain"


# --------------------------------------------------------------------------
# decisions, end to end
# --------------------------------------------------------------------------

def test_execute_returns_value_page_and_bbox(service):
    result = service.answer("查询A13风机接线图的电机编号")
    decision = result.decision
    assert decision["decision"] == "execute"
    assert decision["answerable_fields"]["电机编号"] == "M-13"
    ref = decision["evidence_refs"][0]
    assert ref["document_id"] == "FAN-A13-02" and ref["page"] == 1
    assert len(ref["bbox"]) == 4 and ref["chunk_id"]


def test_missing_device_id_asks_the_user(service):
    result = service.answer("查询电机编号")
    assert result.decision["decision"] == "clarify"
    assert "ENTITY_MISSING_FROM_QUERY" in result.decision["reason_codes"]


def test_partial_answers_what_is_supported_and_names_what_is_not(service):
    """A28's drawing carries no page-number field at all."""
    result = service.answer("查询A28风机参数表的控制柜编号和页码")
    decision = result.decision
    assert decision["decision"] == "partial"
    assert decision["answerable_fields"] == {"控制柜编号": "FAN-CAB-28"}
    assert decision["missing_fields"] == ["页码"]


def test_unknown_device_abstains_rather_than_answering_from_neighbours(service):
    result = service.answer("查询A99风机的电机编号")
    assert result.decision["decision"] == "abstain"
    assert "ENTITY_NOT_IN_CORPUS" in result.decision["reason_codes"]
    assert result.decision["answerable_fields"] == {}


def test_invalid_value_abstains_with_no_second_source(service):
    result = service.answer("查询A19风机接线图的控制柜编号")
    assert result.decision["decision"] == "abstain"
    assert result.decision["answerable_fields"] == {}
    assert "控制柜编号" in result.decision["missing_fields"]


def test_synonym_reaches_the_same_field(service, cfg):
    plan = parse_query("查询A16的额定功率", cfg.query_parsing)
    assert plan.required_fields == ["功率"]
    assert plan.field_requests[0].match_method == "alias"


def test_field_term_the_config_does_not_know_is_marked_unmatched(cfg):
    plan = parse_query("查询A16的转速", cfg.query_parsing)
    assert plan.required_fields == []
    assert "no_target_field_specified" in plan.notes


def test_decision_reports_candidate_and_evidence_counts(service):
    decision = service.answer("查询A13风机接线图的电机编号").decision
    assert decision["retrieved_candidate_count"] >= decision["eligible_evidence_count"]
    assert decision["eligible_evidence_count"] == 1


# --------------------------------------------------------------------------
# honesty of the capability report
# --------------------------------------------------------------------------

def test_capability_report_names_everything_that_did_not_run(service):
    report = service.capability_report()
    assert report["elasticsearch"] == "not_connected"
    assert report["milvus"] == "not_connected"
    assert report["bge_embedding"] == "not_loaded"
    assert report["reranker"] == "not_invoked"
    assert report["vlm"] == "not_invoked"
    assert report["human_reviewed_gold"] is False
    assert report["recall_at_5"] == "not_computed_no_independent_gold"


def test_feature_flag_defaults_to_legacy(cfg):
    """The existing pipeline must keep its behaviour untouched; this slice is
    reached only by asking for it."""
    assert cfg.mode == "legacy"


def test_pipeline_does_not_import_the_rag_package():
    import inspect

    import src.pipeline as pipeline
    assert "src.rag" not in inspect.getsource(pipeline)
