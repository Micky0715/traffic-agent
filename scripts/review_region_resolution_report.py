"""Measure crop AVAILABILITY after graded resolution. No VLM is called.

This round evaluates whether a region can be produced at all, not whether a
field was recovered. Those are different questions and the second one cannot be
answered here: there is no human-reviewed bbox gold, so localisation accuracy,
crop recall, field recovery rate and VLM accuracy are all unreportable.

Two rates are reported separately and must never be added together:

    precise localisation rate   — a located field, answer_eligible
    any region rate             — including contextual and diagnostic regions,
                                  which are a place to look, not an answer
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.multimodal.config import load_multimodal_config  # noqa: E402
from src.tables.cells import assign_blocks_to_cells  # noqa: E402
from src.tables.config import load_tables_config  # noqa: E402
from src.tables.grid import recover_grid  # noqa: E402
from src.vision.config import load_config  # noqa: E402
from src.vision.field_completeness import evaluate_field_completeness  # noqa: E402
from src.vision.ocr_engine import get_ocr_engine  # noqa: E402
from src.vision.review_region_config import load_review_region_config  # noqa: E402
from src.vision.review_region_resolver import (  # noqa: E402
    CONTEXTUAL, DIAGNOSTIC, PRECISE, STRUCTURAL, resolve_review_regions,
)
from src.vision.table_structure import detect_table_structure  # noqa: E402
from src.vision.value_validation import (  # noqa: E402
    field_pairs_from_completeness, validate_field_values,
)

BASELINE = ROOT / "outputs" / "review_region_baseline_audit.json"
REGISTERED = ROOT / "data" / "review_region_cases_registered.json"
OUT_JSON = ROOT / "outputs" / "review_region_resolution_report.json"
OUT_MD = ROOT / "outputs" / "review_region_resolution_report.md"


def run_page(document_id: str, image: Path, cfgs) -> dict:
    vcfg, tcfg, rcfg = cfgs
    engine = get_ocr_engine("fixture")
    if not image.exists():
        return {"document_id": document_id, "image_present": False,
                "regions": [], "targets": []}

    frame = cv2.imread(str(image))
    height, width = frame.shape[:2]
    ocr = engine.recognize(image)
    struct = detect_table_structure(image, ocr_result=ocr)
    completeness = evaluate_field_completeness(
        ocr, vcfg.drawing_types, vcfg.field_completeness, table=struct,
        min_table_confidence=vcfg.min_table_confidence)

    table = recover_grid(image, tcfg.grid, document_id=document_id)
    if table is not None:
        assign_blocks_to_cells(table, ocr.blocks, tcfg.assignment)

    pairs = field_pairs_from_completeness(completeness.field_pairs)
    spec = vcfg.drawing_types.get(completeness.drawing_type)
    validation = validate_field_values(pairs, vcfg.value_validation, spec)

    # Targets exactly as the multimodal stage builds them: missing required
    # fields, invalid values, isolated labels. CELL_SPANS_COLUMNS is NOT here
    # and is not added — the baseline measured it as a false positive.
    field_reasons: dict = {}
    for name in completeness.missing_fields:
        field_reasons.setdefault(name, []).append("required_field_missing")
    for label in validation.invalid_fields:
        field_reasons.setdefault(label.rpartition(".")[2], []).append(
            "invalid_field_value")
    for name in completeness.isolated_labels or []:
        field_reasons.setdefault(name, []).append("isolated_label_without_value")

    regions = resolve_review_regions(
        document_id=document_id, page_no=1, image_size=(width, height),
        ocr_result=ocr, parsed_table=table, completeness=completeness,
        drawing_type=completeness.drawing_type,
        target_fields=sorted(field_reasons), config=rcfg,
        field_reasons=field_reasons)

    return {
        "document_id": document_id,
        "image_present": True,
        "image_size": [width, height],
        "ocr_blocks": len(ocr.blocks),
        "drawing_type": completeness.drawing_type,
        "drawing_type_known": completeness.drawing_type != "unknown",
        "grid_recovered": table is not None,
        "targets": sorted(field_reasons),
        "requests_before_dedup": len(field_reasons) or (1 if regions else 0),
        "regions": [r.to_dict() for r in regions],
    }


def main() -> None:
    cfgs = (load_config(), load_tables_config(), load_review_region_config())
    rcfg = cfgs[2]
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) \
        if BASELINE.exists() else {"targets": []}
    registered = json.loads(REGISTERED.read_text(encoding="utf-8"))

    documents = sorted({row["document_id"] for row in baseline["targets"]} |
                       {d["document_id"] for d in baseline.get("documents", [])})
    pages = [run_page(doc, ROOT / "data" / "drawings" / f"{doc}.png", cfgs)
             for doc in documents]

    regions = [r for page in pages for r in page["regions"]]
    available = [r for r in regions if r["bbox"]]
    by_trust = Counter(r["trust_level"] for r in available)
    by_strategy = Counter(r["strategy"] for r in regions)

    # Per-field coverage: which of the original targets now has a region, and
    # at which grade. A field covered only by a contextual region is NOT a
    # precise localisation and is counted separately.
    field_grade: dict = {}
    for page in pages:
        for region in page["regions"]:
            if not region["bbox"]:
                continue
            for name in region["target_fields"]:
                key = f"{page['document_id']}.{name}"
                rank = {PRECISE: 0, STRUCTURAL: 1, CONTEXTUAL: 2, DIAGNOSTIC: 3}
                if key not in field_grade or \
                        rank[region["trust_level"]] < rank[field_grade[key]]:
                    field_grade[key] = region["trust_level"]

    all_targets = [f"{p['document_id']}.{name}"
                   for p in pages for name in p["targets"]]
    total = len(all_targets) or 1
    precise = sum(1 for k in all_targets if field_grade.get(k) == PRECISE)
    structural = sum(1 for k in all_targets if field_grade.get(k) == STRUCTURAL)
    contextual = sum(1 for k in all_targets if field_grade.get(k) == CONTEXTUAL)
    any_region = sum(1 for k in all_targets if k in field_grade)
    unavailable = total - any_region

    areas = [r["area_ratio"] for r in available] or [0.0]
    before = sum(p["requests_before_dedup"] for p in pages)

    # ---- the nine baseline failures, one by one -------------------------
    baseline_failures = [r for r in baseline["targets"] if not r["crop_available"]]
    resolved_rows = []
    for row in baseline_failures:
        key = f"{row['document_id']}.{row['field_name']}"
        grade = field_grade.get(key)
        region = next((r for p in pages if p["document_id"] == row["document_id"]
                       for r in p["regions"]
                       if row["field_name"] in r["target_fields"] and r["bbox"]), None)
        resolved_rows.append({
            "document_id": row["document_id"],
            "field_name": row["field_name"],
            "baseline_failure_stage": row["failure_stage"],
            "now_strategy": region["strategy"] if region else "unavailable",
            "now_trust_level": grade or "—",
            "now_bbox": region["bbox"] if region else [],
            "still_unavailable": region is None,
            "answer_eligible": bool(region and region["answer_eligible"]),
            "why": (region["reasons"][-1] if region and region["reasons"]
                    else "no strategy produced a region"),
        })

    payload = {
        "note": ("Crop AVAILABILITY only. No VLM was called. No human-reviewed "
                 "bbox gold exists, so localisation accuracy, crop recall, "
                 "field recovery rate, VLM accuracy and generalisation are NOT "
                 "reported and must not be inferred from these counts."),
        "real_vlm_calls": 0,
        "human_reviewed_bbox_gold": False,
        "threshold_provenance": rcfg.threshold_provenance,
        "not_a_blind_test": registered["not_a_blind_test"],
        "metrics": {
            "total_review_targets": total,
            "precise_crop_available_count": precise,
            "precise_crop_available_rate": round(precise / total, 4),
            "structural_crop_available_count": structural,
            "structural_crop_available_rate": round(structural / total, 4),
            "contextual_crop_available_count": contextual,
            "contextual_crop_available_rate": round(contextual / total, 4),
            "diagnostic_region_count": by_strategy.get("full_page_diagnostic", 0),
            "diagnostic_region_rate": round(
                by_strategy.get("full_page_diagnostic", 0) / total, 4),
            "any_region_available_count": any_region,
            "any_region_available_rate": round(any_region / total, 4),
            "crop_unavailable_count": unavailable,
            "crop_unavailable_rate": round(unavailable / total, 4),
            "requests_before_dedup": before,
            "requests_after_dedup": len(regions),
            "deduplicated_request_count": max(0, before - len(regions)),
            "regions_by_strategy": dict(by_strategy),
            "regions_by_trust_level": dict(by_trust),
            "area_ratio_p50": round(statistics.median(areas), 4),
            "area_ratio_p95": round(sorted(areas)[int(len(areas) * 0.95) - 1
                                                  if len(areas) > 1 else 0], 4),
            "unknown_type_pages": sum(1 for p in pages
                                      if p.get("image_present")
                                      and not p["drawing_type_known"]),
            "unknown_type_pages_with_diagnostic": sum(
                1 for p in pages if p.get("image_present")
                and not p["drawing_type_known"]
                and any(r["strategy"] == "full_page_diagnostic" for r in p["regions"])),
            "clean_pages_with_zero_requests": sum(
                1 for p in pages if p.get("image_present")
                and not p["targets"] and not p["regions"]),
            "CELL_SPANS_COLUMNS_only_requests": 0,
        },
        "two_rates_are_not_addable": (
            "precise_crop_available_rate counts located fields; "
            "any_region_available_rate includes contextual and diagnostic "
            "regions, which are a place to look, not an answer. A full-page "
            "diagnostic is NEVER counted as a local crop."),
        "baseline_nine_failures_resolved": resolved_rows,
        "pages": pages,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    m = payload["metrics"]
    lines = [
        "# 复核区域分级定位报告", "",
        "> **本轮不调用 VLM**（`real_vlm_calls=0`）。**无人工 bbox Gold**，",
        "> 因此不报告定位准确率、裁剪 Recall、字段恢复率、VLM 准确率与泛化能力。", "",
        f"> 阈值来源：{rcfg.threshold_provenance}", "",
        "## 两个必须分开看的比率", "", "| 指标 | 分子 | 比率 |", "|---|---|---|",
        f"| **精确局部裁剪可得率** | {precise}/{total} | **{m['precise_crop_available_rate']:.1%}** |",
        f"| 结构级 | {structural}/{total} | {m['structural_crop_available_rate']:.1%} |",
        f"| 上下文级 | {contextual}/{total} | {m['contextual_crop_available_rate']:.1%} |",
        f"| **任意复核区域可得率** | {any_region}/{total} | **{m['any_region_available_rate']:.1%}** |",
        f"| 不可用 | {unavailable}/{total} | {m['crop_unavailable_rate']:.1%} |", "",
        f"> {payload['two_rates_are_not_addable']}", "",
        "## 去重", "",
        f"- 去重前请求：**{m['requests_before_dedup']}**",
        f"- 去重后请求：**{m['requests_after_dedup']}**",
        f"- 合并掉：**{m['deduplicated_request_count']}**", "",
        "## 分布", "",
        f"- strategy：`{m['regions_by_strategy']}`",
        f"- trust_level：`{m['regions_by_trust_level']}`",
        f"- 面积占比 p50=**{m['area_ratio_p50']:.1%}** p95=**{m['area_ratio_p95']:.1%}**", "",
        f"- 类型未知页：{m['unknown_type_pages']}，其中生成诊断请求："
        f"**{m['unknown_type_pages_with_diagnostic']}**",
        f"- 干净页零请求：**{m['clean_pages_with_zero_requests']}**",
        f"- 仅由 CELL_SPANS_COLUMNS 产生的复核请求：**{m['CELL_SPANS_COLUMNS_only_requests']}**", "",
        "## 基线九条 crop_unavailable 的新状态", "",
        "| 文档 | 字段 | 基线失败原因 | 现策略 | 信任级 | 仍不可用 | 可自动回答 | 原因 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in resolved_rows:
        lines.append(
            f"| {row['document_id']} | {row['field_name']} | "
            f"`{row['baseline_failure_stage']}` | `{row['now_strategy']}` | "
            f"{row['now_trust_level']} | {'**是**' if row['still_unavailable'] else '否'} | "
            f"{'是' if row['answer_eligible'] else '**否**'} | {row['why'][:54]} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"targets {total}  precise {precise} ({m['precise_crop_available_rate']:.1%})  "
          f"any_region {any_region} ({m['any_region_available_rate']:.1%})  "
          f"unavailable {unavailable}")
    print(f"strategies {dict(by_strategy)}")
    print(f"dedup {m['requests_before_dedup']} -> {m['requests_after_dedup']}")
    print(f"unknown-type pages with diagnostic: "
          f"{m['unknown_type_pages_with_diagnostic']}/{m['unknown_type_pages']}  "
          f"clean pages with zero requests: {m['clean_pages_with_zero_requests']}")
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
