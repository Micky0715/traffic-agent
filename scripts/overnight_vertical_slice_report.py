"""Build the overnight vertical-slice audit report.

Runs the chunk -> retrieve -> decide slice end to end over the repo's own
parsed drawing output plus clearly-labelled synthetic structural fixtures, and
records what actually ran versus what did not.

Zero API calls. No Elasticsearch, no Milvus, no embedding model, no reranker,
no VLM.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.chunking import (  # noqa: E402
    build_clause_chunks, build_drawing_field_chunks, build_section_chunks,
    build_table_chunks,
)
from src.rag.config import load_rag_config  # noqa: E402
from src.rag.policy import (  # noqa: E402
    EvidenceBundle, FieldEvidence, RequiredEvidencePolicy,
)
from src.rag.retrieval import build_hybrid_retriever  # noqa: E402
from src.vision.schemas import (  # noqa: E402
    DeviceEntity, DeviceParameter, DrawingMetadata, DrawingParseResult,
    RegulationClause, RegulationMetadata, TableStructure, ValidationResult,
    VisualEvidence,
)

OUT_JSON = ROOT / "outputs" / "overnight_vertical_slice_report.json"
cfg = load_rag_config()


# ---------------------------------------------------------------------------
# corpus: real repo parse output + labelled synthetic structural fixtures
# ---------------------------------------------------------------------------

def real_drawing_results():
    """Built from data/mock_drawings.json — repo MOCK data, not enterprise
    data, and labelled as such everywhere it is reported."""
    rows = json.loads((ROOT / "data" / "mock_drawings.json").read_text(encoding="utf-8"))
    results = []
    for index, row in enumerate(rows):
        parameters, evidence = [], []
        for name, value in (row.get("key_fields") or {}).items():
            parameters.append(DeviceParameter(
                name=name, raw_name=name, raw_text=str(value)))
            evidence.append(VisualEvidence(
                field=name, value=str(value), page=int(row.get("page", 1) or 1),
                bbox=[40.0, 100.0 + 30 * len(evidence), 600.0, 130.0 + 30 * len(evidence)],
                region_id=f"r{len(evidence)}", source="ocr", confidence=0.9))
        results.append(DrawingParseResult(
            document_id=f"drawing_{index:03d}",
            page_number=int(row.get("page", 1) or 1),
            drawing_metadata=DrawingMetadata(
                drawing_id=row.get("drawing_no"), drawing_name=row.get("title")),
            devices=[DeviceEntity(device_id=row.get("asset_id", "unknown"),
                                  parameters=parameters)],
            evidence=evidence,
            validation=ValidationResult(status="confirmed")))
    return results


def synthetic_fixtures():
    """SYNTHETIC_TEST_FIXTURE — structural shapes the repo's real data does not
    contain (multi-level headers, merged cells). Used for structural counts
    only; never for an effectiveness claim."""
    fan_table = TableStructure(headers=[], rows=[
        ["设备", "参数", "", "备注"],
        ["", "功率", "风量", ""],
        ["A16", "45kW", "28000m3/h", "常用"],
        ["A17", "55kW", "32000m3/h", "备用"],
        ["A18", "37kW", "24000m3/h", "常用"],
    ], bbox=[10.0, 20.0, 700.0, 400.0], confidence=0.8)
    pump_table = TableStructure(headers=[], rows=[
        ["设备", "流量", "扬程"],
        ["2A", "80m3/h", "22m"],
        ["2B", "80m3/h", "24m"],
    ], bbox=[10.0, 20.0, 700.0, 300.0], confidence=0.75)
    regulation = RegulationMetadata(document_name="JTG D81-2017", clauses=[
        RegulationClause(chapter="第4章 照明", article="4.2.1",
                         text="应急照明应保证不小于30分钟的持续供电。", line_number=12),
        RegulationClause(chapter="第4章 照明", article="4.2.2",
                         text="疏散指示标志应连续设置，间距不大于20米。", line_number=13),
    ])
    sections = [
        {"section_path": ["第4章 通风", "4.1 概述"],
         "text": "本章规定隧道通风系统的设计与检修要求。", "page": 1,
         "bbox": [30.0, 60.0, 700.0, 120.0]},
        {"section_path": ["第4章 通风", "4.2 风机"],
         "text": "风机应每季度检修一次。检修内容包括轴承润滑与振动检测。", "page": 2,
         "bbox": [30.0, 60.0, 700.0, 160.0]},
    ]
    return fan_table, pump_table, regulation, sections


def build_corpus():
    chunks, provenance = [], Counter()
    for result in real_drawing_results():
        built = build_drawing_field_chunks(result, parser_source="ocr")
        chunks += built
        provenance["repo_mock_drawings"] += len(built)

    fan_table, pump_table, regulation, sections = synthetic_fixtures()
    synthetic = []
    synthetic += build_table_chunks(
        "synthetic_fan_doc", fan_table, table_id="T1", page=3, header_row_count=2,
        table_title="风机参数表", section_path=["第4章 通风"])
    synthetic += build_table_chunks(
        "synthetic_pump_doc", pump_table, table_id="T2", page=9, header_row_count=1,
        table_title="水泵参数表")
    synthetic += build_clause_chunks("synthetic_reg_doc", regulation,
                                     regulation_code="JTG D81-2017")
    synthetic += build_section_chunks("synthetic_text_doc", sections, cfg.chunking)
    chunks += synthetic
    provenance["synthetic_test_fixture"] += len(synthetic)
    return chunks, provenance


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def main() -> None:
    chunks, provenance = build_corpus()
    retriever = build_hybrid_retriever(
        cfg.retrieval, synonyms={"额定功率": ["功率"], "风量": ["排风量"]})
    retriever.index(chunks)
    policy = RequiredEvidencePolicy(cfg.evidence_policy)

    # ---- structural metrics -------------------------------------------
    by_type = Counter(c.content_type for c in chunks)
    table_like = [c for c in chunks if c.content_type in {"table_row", "table_parent"}]
    entity_scoped = [c for c in chunks
                     if c.content_type in {"table_row", "drawing_field"}]
    structural = {
        "chunk_count": len(chunks),
        "chunk_type_distribution": dict(sorted(by_type.items())),
        "chunk_provenance": dict(provenance),
        "header_path_retention": {
            "numerator": sum(1 for c in table_like if c.header_paths),
            "denominator": len(table_like)},
        "entity_id_retention": {
            "numerator": sum(1 for c in entity_scoped if c.entity_id),
            "denominator": len(entity_scoped)},
        "page_traceability": {
            "numerator": sum(1 for c in chunks if c.page_start is not None),
            "denominator": len(chunks)},
        "bbox_traceability": {
            "numerator": sum(1 for c in chunks if c.source_bboxes),
            "denominator": len(chunks)},
        "stable_chunk_id_coverage": {
            "numerator": sum(1 for c in chunks if c.chunk_id),
            "denominator": len(chunks)},
    }

    # Cross-device contamination: a device-scoped chunk containing another
    # device's identifier.
    entity_ids = {c.entity_id for c in chunks if c.entity_id}
    contamination = [
        c.chunk_id for c in entity_scoped
        if any(other != c.entity_id and other in c.text for other in entity_ids)]
    structural["cross_device_contamination"] = {
        "numerator": len(contamination), "denominator": len(entity_scoped),
        "offending_chunk_ids": contamination}

    # ---- retrieval probes ---------------------------------------------
    probes = [
        ("exact_device_id", "A17"),
        ("exact_drawing_no", "FAN-A12-01"),
        ("field_path", "参数.功率"),
        ("synonym_expansion", "A16 额定功率"),
        ("clause_lookup", "应急照明 持续供电"),
        ("no_match", "完全不存在的查询词组合"),
    ]
    latencies, probe_rows = [], []
    for name, query in probes:
        started = time.perf_counter()
        hits = retriever.search(query)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        probe_rows.append({
            "probe": name, "query": query, "hit_count": len(hits),
            "latency_ms": round(elapsed, 2),
            "top": [{
                "chunk_id": h.chunk.chunk_id, "content_type": h.chunk.content_type,
                "entity_id": h.chunk.entity_id, "text": h.chunk.text[:110],
                "sparse_rank": h.sparse_rank, "dense_rank": h.dense_rank,
                "rrf_score": round(h.rrf_score, 6), "fused_rank": h.fused_rank,
                "rerank_score": h.rerank_score, "rerank_status": h.rerank_status,
                "parent_context": (h.parent_context or "")[:80],
                "page": h.chunk.page_start, "bbox": h.chunk.source_bboxes[:1],
                "parser_source": h.chunk.parser_source,
            } for h in hits[:3]],
        })

    sparse_only = sum(1 for r in probe_rows for h in r["top"]
                      if h["sparse_rank"] and not h["dense_rank"])
    dense_only = sum(1 for r in probe_rows for h in r["top"]
                     if h["dense_rank"] and not h["sparse_rank"])
    both = sum(1 for r in probe_rows for h in r["top"]
               if h["sparse_rank"] and h["dense_rank"])

    # ---- decision scenarios -------------------------------------------
    def field(name, value, **kwargs):
        base = {"source": "ocr", "document_id": "drawing_000", "page": 1,
                "bbox": [40.0, 100.0, 600.0, 130.0], "entity_id": "A12风机",
                "confidence": 0.9, "validation_status": "valid"}
        base.update(kwargs)
        return FieldEvidence(field_name=name, value=value, **base)

    scenarios = {
        "complete_evidence": EvidenceBundle(
            task_type="drawing_field_query", entity_id="A12风机",
            requested_fields=["功率"], fields=[field("功率", "45kW")]),
        "missing_user_parameter": EvidenceBundle(
            task_type="equipment_metric_query", entity_id=None,
            metric_name="告警次数", time_range="最近3天",
            requested_fields=["告警次数"], fields=[field("告警次数", "12", source="tool")]),
        "partial_coverage": EvidenceBundle(
            task_type="drawing_field_query", entity_id="A12风机",
            requested_fields=["功率", "控制柜编号"], fields=[field("功率", "45kW")]),
        "non_high_risk_conflict": EvidenceBundle(
            task_type="drawing_field_query", entity_id="A12风机",
            requested_fields=["功率"],
            fields=[field("功率", "45kW", conflict_with="55kW")]),
        "high_risk_conflict": EvidenceBundle(
            task_type="drawing_field_query", entity_id="A12风机",
            requested_fields=["控制柜编号"],
            fields=[field("控制柜编号", "FAN-CAB-12", conflict_with="审核专用章")]),
        "untraceable_value": EvidenceBundle(
            task_type="drawing_field_query", entity_id="A12风机",
            requested_fields=["功率"],
            fields=[field("功率", "45kW", document_id=None, page=None)]),
        "remediation_exhausted": EvidenceBundle(
            task_type="drawing_field_query", entity_id="A12风机",
            requested_fields=["控制柜编号"], fields=[], remediation_exhausted=True),
    }
    decisions = {name: policy.evaluate(bundle).to_dict()
                 for name, bundle in scenarios.items()}

    report = {
        "note": ("Overnight vertical slice: structured chunking -> hybrid retrieval "
                 "-> evidence policy. Zero API calls."),
        "data_provenance": {
            "repo_mock_drawings": "data/mock_drawings.json — repo MOCK data, not enterprise data",
            "synthetic_test_fixture": ("structural shapes constructed for this run "
                                       "(multi-level headers, merged cells); "
                                       "structural counts only, never an effectiveness claim"),
            "independent_gold": None,
            "recall_at_5": ("NOT COMPUTED: no independent retrieval gold exists in this "
                            "repo. Deriving one from the system's own output would make "
                            "any recall number circular."),
        },
        "capabilities": retriever.capability_report(),
        "structural_metrics": structural,
        "retrieval_probes": probe_rows,
        "retrieval_source_mix": {"sparse_only": sparse_only, "dense_only": dense_only,
                                 "both": both},
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[max(0, int(0.95 * (len(latencies) - 1)))], 2),
            "note": "in-process retrieval only; no network, no model inference",
        },
        "decision_scenarios": decisions,
        "decision_distribution": dict(Counter(d["decision"] for d in decisions.values())),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"chunks {structural['chunk_count']}  types {structural['chunk_type_distribution']}")
    for key in ["header_path_retention", "entity_id_retention", "page_traceability",
                "bbox_traceability", "stable_chunk_id_coverage",
                "cross_device_contamination"]:
        m = structural[key]
        total = m["denominator"] or 1
        print(f"  {key:<32} {m['numerator']}/{m['denominator']} = {m['numerator']/total:.1%}")
    print(f"\nsource mix sparse_only={sparse_only} dense_only={dense_only} both={both}")
    print(f"latency p50={report['latency_ms']['p50']}ms p95={report['latency_ms']['p95']}ms")
    print(f"decisions {report['decision_distribution']}")
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
