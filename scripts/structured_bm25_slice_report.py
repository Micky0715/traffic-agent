"""Audit report for the structured-BM25 vertical slice.

Runs the real CLI path (StructuredRagService, the same object the CLI uses) over
the derived regression set and over five hand-picked end-to-end cases.

Zero API calls. No Elasticsearch, Milvus, embedding model, reranker or VLM.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.retrieval import tokenize  # noqa: E402
from src.rag.service import StructuredRagService  # noqa: E402

OUT_JSON = ROOT / "outputs" / "structured_bm25_vertical_slice_report.json"
DERIVED = ROOT / "data" / "rag_regression_derived.jsonl"
CANDIDATES = ROOT / "data" / "rag_eval_candidates_unreviewed.jsonl"

SHOWCASE = [
    ("execute", "查询A13风机接线图的电机编号"),
    ("clarify_missing_device", "查询电机编号"),
    ("partial", "查询A28风机参数表的控制柜编号和页码"),
    ("abstain_unknown_device", "查询A99风机的电机编号"),
    ("abstain_invalid_value", "查询A19风机接线图的控制柜编号"),
    ("irrelevant_query", "完全不存在的查询词组合"),
]


def main() -> None:
    service = StructuredRagService()
    service.load_corpus()
    stats = service.stats

    # ---- end-to-end showcase -------------------------------------------
    showcase = {}
    for name, query in SHOWCASE:
        result = service.answer(query)
        payload = result.to_dict()
        # Keep the full middle of the pipeline, trimmed only for size.
        payload["retrieved_candidates"] = payload["retrieved_candidates"][:5]
        payload["rejected_candidates"] = payload["rejected_candidates"][:5]
        showcase[name] = payload

    # ---- derived regression set ----------------------------------------
    derived_rows = [json.loads(l) for l in DERIVED.open(encoding="utf-8") if l.strip()]
    derived_results, latencies = [], []
    for row in derived_rows:
        result = service.answer(row["query"])
        decision = result.decision
        answered = decision["answerable_fields"].get(row["expected_field"])
        latencies.append(result.latency_ms)
        derived_results.append({
            "case_id": row["case_id"], "query": row["query"],
            "expected_field": row["expected_field"],
            "expected_value": row["expected_value"],
            "decision": decision["decision"],
            "answered_value": answered,
            "value_matches": answered == row["expected_value"],
            "retrieved": result.retrieved_candidate_count,
            "eligible": result.eligible_evidence_count,
            "evidence_document": (decision["evidence_refs"][0]["document_id"]
                                  if decision["evidence_refs"] else None),
            "document_matches": bool(decision["evidence_refs"]) and
            decision["evidence_refs"][0]["document_id"] == row["expected_document_id"],
        })

    executed = [r for r in derived_results if r["decision"] == "execute"]
    report = {
        "note": ("Structured BM25 vertical slice over REAL parsed drawing output. "
                 "Zero API calls."),
        "capabilities": service.capability_report(),
        "corpus": {
            "source": ("data/ocr_fixtures/*.json — replay of a recorded real "
                       "PaddleOCR 3.7.0 run, stamped with image sha256, library "
                       "and model versions, device and generation time"),
            "is_live_inference": False,
            "is_enterprise_data": False,
            "real_documents": stats.real_documents,
            "real_chunks": stats.real_chunks,
            "synthetic_chunks": stats.synthetic_chunks,
            "mock_chunks": stats.mock_chunks,
            "field_pairs": stats.field_pairs,
            "pairs_with_bbox": stats.pairs_with_bbox,
            "chunk_type_distribution": dict(Counter(
                c.content_type for c in service.chunks)),
            "documents": stats.documents,
            "skipped": stats.skipped,
        },
        "traceability": {
            "page_coverage": {
                "numerator": sum(1 for c in service.chunks if c.page_start is not None),
                "denominator": len(service.chunks)},
            "bbox_coverage_field_chunks": {
                "numerator": sum(1 for c in service.chunks
                                 if c.content_type == "drawing_field" and c.source_bboxes),
                "denominator": sum(1 for c in service.chunks
                                   if c.content_type == "drawing_field")},
        },
        "irrelevant_recall_root_cause": {
            "query": "完全不存在的查询词组合",
            "tokens_without_stopwords": tokenize("完全不存在的查询词组合"),
            "tokens_with_stopwords": tokenize(
                "完全不存在的查询词组合", service.cfg.retrieval.stopwords),
            "explanation": ("CJK tokenizes per character; 的/不 are shared with "
                            "almost any Chinese text, so BM25 scored above zero "
                            "against everything and returned its top K"),
            "primary_defence": ("eligibility gate — a chunk must be about the right "
                                "entity and cover a requested field"),
            "secondary_defence": "configured stopwords in the tokenizer",
            "score_threshold": {
                "value": service.cfg.retrieval.uncalibrated_min_bm25_score,
                "status": service.cfg.retrieval.uncalibrated_status,
                "reason": ("a floor needs calibration on a dev set; choosing one "
                           "because a single query misbehaved is guesswork"),
            },
        },
        "showcase": showcase,
        "derived_regression": {
            "label_source": "derived_from_existing_gold",
            "purpose": "regression_only",
            "caveat": ("questions were written FROM the answers; this catches "
                       "regressions and cannot show generalization"),
            "case_count": len(derived_results),
            "decision_distribution": dict(Counter(r["decision"] for r in derived_results)),
            "execute_count": len(executed),
            "value_match_within_execute": {
                "numerator": sum(1 for r in executed if r["value_matches"]),
                "denominator": len(executed)},
            "document_match_within_execute": {
                "numerator": sum(1 for r in executed if r["document_matches"]),
                "denominator": len(executed)},
            "retrieved_vs_eligible": {
                "mean_retrieved": round(statistics.mean(
                    r["retrieved"] for r in derived_results), 2),
                "mean_eligible": round(statistics.mean(
                    r["eligible"] for r in derived_results), 2)},
            "latency_ms": {"p50": round(statistics.median(latencies), 3),
                           "p95": round(sorted(latencies)[
                               max(0, int(0.95 * (len(latencies) - 1)))], 3)},
            "rows": derived_results,
        },
        "unreviewed_candidates": {
            "path": "data/rag_eval_candidates_unreviewed.jsonl",
            "count": sum(1 for l in CANDIDATES.open(encoding="utf-8") if l.strip()),
            "label_status": "unreviewed",
            "human_reviewed_gold": False,
            "warning": ("machine-proposed expectations. No Recall@5, refusal rate or "
                        "accuracy may be computed from these until a person has "
                        "reviewed them. They are NOT gold."),
        },
        "recall_at_5": "not_computed_no_human_reviewed_gold",
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"corpus: {stats.real_documents} real docs, {stats.real_chunks} chunks, "
          f"{stats.pairs_with_bbox}/{stats.field_pairs} pairs with bbox")
    print(f"chunk types: {report['corpus']['chunk_type_distribution']}")
    print(f"\nshowcase decisions: "
          f"{ {k: v['decision']['decision'] for k, v in showcase.items()} }")
    d = report["derived_regression"]
    print(f"\nderived regression ({d['case_count']} cases, regression_only):")
    print(f"  decisions {d['decision_distribution']}")
    print(f"  value match within execute    "
          f"{d['value_match_within_execute']['numerator']}/"
          f"{d['value_match_within_execute']['denominator']}")
    print(f"  document match within execute "
          f"{d['document_match_within_execute']['numerator']}/"
          f"{d['document_match_within_execute']['denominator']}")
    print(f"  retrieved {d['retrieved_vs_eligible']['mean_retrieved']} vs "
          f"eligible {d['retrieved_vs_eligible']['mean_eligible']} (mean)")
    print(f"  latency p50 {d['latency_ms']['p50']}ms p95 {d['latency_ms']['p95']}ms")
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
