"""What the shadow candidate generator proposes. Nothing here is scored.

With no human bbox gold, this report is only allowed to say how many candidates
exist, what evidence they rest on, and how ambiguous they are. It may NOT say
which candidate is correct — there is nothing to check that against, and
"the one the system liked best" is not an answer.

Production is untouched: these candidates are generated here and discarded.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tables.cells import assign_blocks_to_cells, detect_key_value_layout  # noqa: E402
from src.tables.config import load_tables_config  # noqa: E402
from src.tables.grid import recover_grid  # noqa: E402
from src.vision.config import load_config  # noqa: E402
from src.vision.field_completeness import evaluate_field_completeness  # noqa: E402
from src.vision.ocr_engine import get_ocr_engine  # noqa: E402
from src.vision.review_region_config import load_review_region_config  # noqa: E402
from src.vision.review_region_resolver import identify_key_value_rows  # noqa: E402
from src.vision.structural_region_candidates import (  # noqa: E402
    generate_structural_candidates, load_shadow_config, score_gap,
)
from src.vision.table_structure import detect_table_structure  # noqa: E402

BASELINE = ROOT / "outputs" / "review_region_baseline_audit.json"
OUT_JSON = ROOT / "outputs" / "structural_region_shadow_report.json"
OUT_MD = ROOT / "outputs" / "structural_region_shadow_report.md"


def main() -> None:
    vcfg = load_config()
    tcfg = load_tables_config()
    rcfg = load_review_region_config()
    scfg = load_shadow_config()
    engine = get_ocr_engine("fixture")

    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) \
        if BASELINE.exists() else {"documents": []}
    documents = sorted({d["document_id"] for d in baseline.get("documents", [])})

    pages = []
    for document_id in documents:
        image = ROOT / "data" / "drawings" / f"{document_id}.png"
        if not image.exists():
            continue
        ocr = engine.recognize(image)
        struct = detect_table_structure(image, ocr_result=ocr)
        completeness = evaluate_field_completeness(
            ocr, vcfg.drawing_types, vcfg.field_completeness, table=struct,
            min_table_confidence=vcfg.min_table_confidence)
        table = recover_grid(image, tcfg.grid, document_id=document_id)
        if table is not None:
            assign_blocks_to_cells(table, ocr.blocks, tcfg.assignment)

        key_value = bool(table is not None and detect_key_value_layout(table))
        identified = identify_key_value_rows(table, rcfg) if key_value else {}

        fields = []
        for name in sorted(completeness.missing_fields):
            candidates = generate_structural_candidates(
                table, ocr.blocks, name, completeness.drawing_type, scfg,
                aliases=rcfg.field_aliases.get(name, [name]),
                identified_rows=identified)
            fields.append({
                "target_field": name,
                "candidate_count": len(candidates),
                "ambiguity_count": candidates[0].ambiguity_count if candidates else 0,
                "top1_minus_top2": score_gap(candidates),
                "candidates": [c.to_dict() for c in candidates],
                "no_candidate_reason": (
                    None if candidates else
                    ("page has no confirmed key/value table"
                     if not key_value else
                     "every row scored below min_candidate_score")),
            })

        pages.append({
            "document_id": document_id,
            "drawing_type": completeness.drawing_type,
            "key_value_table": key_value,
            "grid_shape": f"{table.row_count}x{table.col_count}" if table else None,
            "identified_rows": {str(k): v for k, v in identified.items()},
            "field_order_used": scfg.order_for(completeness.drawing_type),
            "fields": fields,
        })

    all_fields = [f for p in pages for f in p["fields"]]
    with_candidates = [f for f in all_fields if f["candidate_count"] > 0]
    gaps = [f["top1_minus_top2"] for f in with_candidates
            if f["top1_minus_top2"] is not None]
    strategies = Counter(c["strategy"] for f in all_fields
                         for c in f["candidates"])
    assumptions = Counter(a for f in all_fields for c in f["candidates"]
                          for a in c["assumptions"])

    payload = {
        "note": ("Shadow only. Every candidate carries "
                 "promotion_status=shadow_only, nothing here is imported by "
                 "resolve_review_regions or any decision path, and no "
                 "production behaviour changed."),
        "production_behavior_changed": False,
        "real_vlm_calls": 0,
        "external_api_calls": 0,
        "human_reviewed_bbox_gold": False,
        "cannot_report": [
            "which candidate is correct",
            "localization accuracy or recall",
            "whether candidate coverage of all rows constitutes recall",
        ],
        "threshold_provenance": scfg.threshold_provenance,
        "calibrated_with_human_gold": scfg.calibrated_with_human_gold,
        "field_order_provenance": scfg.field_order_provenance,
        "evidence_terms": sorted(scfg.weights),
        "weights": scfg.weights,
        "summary": {
            "fields_examined": len(all_fields),
            "fields_with_at_least_one_candidate": len(with_candidates),
            "fields_with_no_candidate": len(all_fields) - len(with_candidates),
            "total_candidates": sum(f["candidate_count"] for f in all_fields),
            "candidates_per_field_p50": (
                statistics.median([f["candidate_count"] for f in with_candidates])
                if with_candidates else None),
            "max_candidates_for_one_field": max(
                [f["candidate_count"] for f in all_fields], default=0),
            "fields_with_ambiguity_gt_1": sum(
                1 for f in with_candidates if f["ambiguity_count"] > 1),
            "top1_minus_top2_p50": round(statistics.median(gaps), 4) if gaps else None,
            "top1_minus_top2_min": round(min(gaps), 4) if gaps else None,
            "candidates_by_strategy": dict(strategies),
            "assumptions_recorded": dict(assumptions),
        },
        "pages": pages,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    s = payload["summary"]
    lines = [
        "# 结构定位 Shadow 候选报告", "",
        "> **影子实验。** 全部候选 `promotion_status=shadow_only`，",
        "> 不被 `resolve_review_regions` 或任何决策路径引用，**生产行为未改变**。",
        "> **无人工 bbox Gold，因此本报告不说哪个候选是对的。**", "",
        f"- 阈值来源：`{payload['threshold_provenance']}`　"
        f"人工 Gold 校准：**{payload['calibrated_with_human_gold']}**",
        f"- 字段顺序来源：`{payload['field_order_provenance']}`", "",
        "## 汇总", "", "| 指标 | 值 |", "|---|---|",
        f"| 考察字段数 | {s['fields_examined']} |",
        f"| 至少一个候选 | **{s['fields_with_at_least_one_candidate']}** |",
        f"| 无候选 | {s['fields_with_no_candidate']} |",
        f"| 候选总数 | {s['total_candidates']} |",
        f"| 单字段候选数中位数 | {s['candidates_per_field_p50']} |",
        f"| 单字段最多候选 | {s['max_candidates_for_one_field']} |",
        f"| 存在歧义（>1 候选）的字段 | **{s['fields_with_ambiguity_gt_1']}** |",
        f"| top1−top2 中位数 | {s['top1_minus_top2_p50']} |",
        f"| top1−top2 最小值 | {s['top1_minus_top2_min']} |", "",
        f"- 候选策略分布：`{s['candidates_by_strategy']}`",
        f"- 记录的假设：`{s['assumptions_recorded']}`",
        f"- 参与评分的证据项：`{payload['evidence_terms']}`", "",
        "## 逐字段", "",
        "| 文档 | 字段 | 候选数 | 歧义 | top1−top2 | top1 策略 | 无候选原因 |",
        "|---|---|---|---|---|---|---|",
    ]
    for page in pages:
        for f in page["fields"]:
            top = f["candidates"][0]["strategy"] if f["candidates"] else "—"
            lines.append(
                f"| {page['document_id']} | {f['target_field']} | "
                f"{f['candidate_count']} | {f['ambiguity_count']} | "
                f"{f['top1_minus_top2'] if f['top1_minus_top2'] is not None else '—'} | "
                f"`{top}` | {f['no_candidate_reason'] or '—'} |")
    lines += ["", "> 候选覆盖了表格全部行**不等于 Recall 100%**；",
              "> 在人工 Gold 到位之前，覆盖率只说明搜索空间大小，不说明命中。"]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"fields {s['fields_examined']}  with candidates "
          f"{s['fields_with_at_least_one_candidate']}  "
          f"total candidates {s['total_candidates']}")
    print(f"ambiguous fields {s['fields_with_ambiguity_gt_1']}  "
          f"top1-top2 p50={s['top1_minus_top2_p50']} min={s['top1_minus_top2_min']}")
    print(f"strategies {s['candidates_by_strategy']}")
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
