"""Hybrid retrieval: BM25 + offline dense + RRF + parent expansion.

SYNTHETIC TEST FIXTURES only. No Elasticsearch, no Milvus, no embedding model
and no reranker are contacted; the dense side is the declared deterministic
stand-in, and these tests assert that it says so.
"""
from __future__ import annotations

import pytest

from src.rag.chunking import build_drawing_field_chunks, build_table_chunks
from src.rag.config import RetrievalConfig, load_rag_config
from src.rag.retrieval import (
    BM25Retriever, OfflineDeterministicDenseRetriever, build_hybrid_retriever,
    expand_parent_context, reciprocal_rank_fusion, tokenize,
)
from src.vision.schemas import TableStructure

SYNTHETIC_TEST_FIXTURE = True


@pytest.fixture
def cfg() -> RetrievalConfig:
    return load_rag_config().retrieval


def fan_table() -> TableStructure:
    return TableStructure(headers=[], rows=[
        ["设备", "参数", "", "备注"],
        ["", "功率", "风量", ""],
        ["A16", "45kW", "28000m3/h", "常用"],
        ["A17", "55kW", "32000m3/h", "备用"],
        ["A18", "37kW", "24000m3/h", "常用"],
    ], bbox=[10.0, 20.0, 700.0, 400.0], confidence=0.8)


@pytest.fixture
def corpus():
    chunks = build_table_chunks("doc1", fan_table(), table_id="T1", page=3,
                                header_row_count=2, table_title="风机参数表",
                                section_path=["第4章 通风"])
    chunks += build_table_chunks("doc1", TableStructure(headers=[], rows=[
        ["设备", "扬程"], ["2A", "22m"], ["2B", "24m"],
    ], confidence=0.7), table_id="T2", page=9, table_title="水泵参数表")
    return chunks


@pytest.fixture
def retriever(cfg, corpus):
    hybrid = build_hybrid_retriever(cfg, synonyms={"功率": ["电机功率", "额定功率"]})
    hybrid.index(corpus)
    return hybrid


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

def test_codes_survive_tokenization_as_single_tokens():
    """Splitting FAN-A13-02 into pieces is exactly how an exact-identifier
    query stops being exact."""
    assert "fan-a13-02" in tokenize("查 FAN-A13-02 图纸")
    assert "jtg" in tokenize("JTG D81-2017")


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

def test_exact_device_id_is_found_by_bm25(cfg, corpus):
    bm25 = BM25Retriever(cfg)
    bm25.index(corpus)
    hits = bm25.search("A17", top_k=5)
    assert hits
    assert hits[0].chunk.entity_id == "A17"


def test_bm25_returns_nothing_rather_than_inventing_a_hit(cfg, corpus):
    bm25 = BM25Retriever(cfg)
    bm25.index(corpus)
    assert bm25.search("完全不相关的词汇组合", top_k=5) == []


def test_bm25_is_deterministic(cfg, corpus):
    bm25 = BM25Retriever(cfg)
    bm25.index(corpus)
    first = [(h.chunk.chunk_id, round(h.score, 9)) for h in bm25.search("A16 功率", 5)]
    second = [(h.chunk.chunk_id, round(h.score, 9)) for h in bm25.search("A16 功率", 5)]
    assert first == second


# --------------------------------------------------------------------------
# dense stand-in
# --------------------------------------------------------------------------

def test_dense_stand_in_declares_it_is_not_a_real_model(cfg, corpus):
    dense = OfflineDeterministicDenseRetriever(cfg)
    assert dense.is_real_model is False and dense.is_real_service is False


def test_configured_synonym_reaches_the_right_candidate(cfg, corpus):
    """Declared expansion from configuration, not learned similarity. Asserted
    so the report cannot overstate what the dense side does."""
    dense = OfflineDeterministicDenseRetriever(cfg, synonyms={"额定功率": ["功率"]})
    dense.index(corpus)
    hits = dense.search("A16 额定功率", top_k=5)
    assert any(h.chunk.entity_id == "A16" for h in hits)


# --------------------------------------------------------------------------
# RRF
# --------------------------------------------------------------------------

def test_rrf_is_reproducible(retriever):
    first = [(h.chunk.chunk_id, round(h.rrf_score, 12)) for h in retriever.search("A17 功率")]
    second = [(h.chunk.chunk_id, round(h.rrf_score, 12)) for h in retriever.search("A17 功率")]
    assert first == second


def test_rrf_score_matches_the_formula(cfg, corpus):
    bm25 = BM25Retriever(cfg)
    bm25.index(corpus)
    dense = OfflineDeterministicDenseRetriever(cfg)
    dense.index(corpus)
    sparse_hits = bm25.search("A16", cfg.sparse_top_k)
    dense_hits = dense.search("A16", cfg.dense_top_k)
    fused = reciprocal_rank_fusion(sparse_hits, dense_hits, cfg)

    top = fused[0]
    expected = 0.0
    if top.sparse_rank:
        expected += 1 / (cfg.rrf_k + top.sparse_rank)
    if top.dense_rank:
        expected += 1 / (cfg.rrf_k + top.dense_rank)
    assert top.rrf_score == pytest.approx(expected)


def test_rrf_score_is_not_a_probability(cfg, corpus):
    """Rank-one from both retrievers scores ~0.0328 with k=60. Reading that as
    "3% confident" — or thresholding at 0.8 — is meaningless."""
    bm25 = BM25Retriever(cfg)
    bm25.index(corpus)
    dense = OfflineDeterministicDenseRetriever(cfg)
    dense.index(corpus)
    fused = reciprocal_rank_fusion(bm25.search("A16", 20), dense.search("A16", 20), cfg)
    assert fused[0].rrf_score < 0.05
    assert all(0 < h.rrf_score <= 2 / (cfg.rrf_k + 1) for h in fused)


def test_every_candidate_reports_which_retriever_found_it(retriever):
    hits = retriever.search("A16 功率")
    assert hits
    for hit in hits:
        assert hit.sparse_rank is not None or hit.dense_rank is not None
        assert hit.fused_rank >= 1


def test_duplicate_chunks_are_merged_but_different_devices_are_not(retriever):
    hits = retriever.search("功率")
    ids = [h.chunk.chunk_id for h in hits]
    assert len(ids) == len(set(ids))            # deduplicated
    entities = {h.chunk.entity_id for h in hits if h.chunk.entity_id}
    assert len(entities) > 1                    # not collapsed into one row


# --------------------------------------------------------------------------
# parent expansion
# --------------------------------------------------------------------------

def test_parent_expansion_adds_the_tables_own_header(retriever):
    hits = retriever.search("A16 功率")
    a16 = next(h for h in hits if h.chunk.entity_id == "A16")
    assert a16.parent_context and "风机参数表" in a16.parent_context


def test_parent_expansion_never_splices_in_another_device(corpus, cfg):
    """Neighbouring chunks in a table are OTHER DEVICES; blindly attaching
    surrounding chunks is how one device's figures get attributed to another."""
    bm25 = BM25Retriever(cfg)
    bm25.index(corpus)
    fused = reciprocal_rank_fusion(bm25.search("A16", 20), [], cfg)
    expanded = expand_parent_context(fused, corpus)
    a16 = next(h for h in expanded if h.chunk.entity_id == "A16")
    assert "A17" not in (a16.parent_context or "")
    assert "55kW" not in (a16.parent_context or "")


def test_multi_level_header_field_is_retrievable_by_its_full_path(retriever):
    hits = retriever.search("参数.风量")
    assert any(h.chunk.entity_id for h in hits)


def test_empty_query_returns_no_evidence(retriever):
    assert retriever.search("") == []


# --------------------------------------------------------------------------
# capability honesty
# --------------------------------------------------------------------------

def test_capability_report_states_what_did_not_run(retriever):
    report = retriever.capability_report()
    assert report["elasticsearch"]["status"] == "not_connected"
    assert report["milvus"]["status"] == "not_connected"
    assert report["reranker"]["status"] == "not_invoked"
    assert report["sparse"]["real_service"] is False
    assert report["dense"]["real_model"] is False
