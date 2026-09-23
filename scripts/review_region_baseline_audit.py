"""Why 9 of 11 review targets could not be cropped. AUDIT ONLY.

Runs the CURRENT code unchanged and records, per target, what geometry the
pipeline had available at the moment it gave up. No algorithm is called
differently here than in scripts/multimodal_review_eval.py; the difference is
that this script also asks what ELSE was on the page — a table, a grid cell, a
title block, a neighbouring label — so the next round knows whether a fallback
had anything to fall back to.

Nothing here modifies code, config, cache or any previous report.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.multimodal.config import load_multimodal_config  # noqa: E402
from src.multimodal.crop import resolve_review_crop  # noqa: E402
from src.multimodal.pipeline import locate_label_bbox  # noqa: E402
from src.multimodal.reasons import (  # noqa: E402
    INVALID_FIELD_VALUE, REQUIRED_FIELD_MISSING,
)
from src.multimodal.triggers import TriggerCollector  # noqa: E402
from src.tables.cells import assign_blocks_to_cells, detect_key_value_layout  # noqa: E402
from src.tables.config import load_tables_config  # noqa: E402
from src.tables.grid import find_table_regions, recover_grid  # noqa: E402
from src.vision.config import load_config  # noqa: E402
from src.vision.field_completeness import evaluate_field_completeness  # noqa: E402
from src.vision.ocr_engine import get_ocr_engine  # noqa: E402
from src.vision.table_structure import detect_table_structure  # noqa: E402
from src.vision.value_validation import (  # noqa: E402
    field_pairs_from_completeness, validate_field_values,
)

REGISTERED = ROOT / "data" / "multimodal_cases_registered.json"
OUT_JSON = ROOT / "outputs" / "review_region_baseline_audit.json"
OUT_MD = ROOT / "outputs" / "review_region_baseline_audit.md"

CODE_MAP = {
    "review_request_generation": "src/multimodal/triggers.py:32 TriggerCollector "
                                 "(legacy: src/tables/review.py:57 collect_review_requests)",
    "locate_label_bbox": "src/multimodal/pipeline.py:35",
    "crop_generation": "src/multimodal/crop.py:106 resolve_review_crop",
    "table_and_cell_bbox": "src/tables/grid.py:142 recover_grid; "
                           "src/tables/schemas.py TableCell.bbox",
    "drawing_type_classification": "src/vision/field_completeness.py:118 classify_drawing_type",
    "field_completeness": "src/vision/field_completeness.py:315 evaluate_field_completeness",
    "vlm_executor": "src/multimodal/executor.py:44 VisionReviewExecutor",
    "second_evidence_policy": "src/multimodal/decision.py:120 run_second_pass "
                              "-> src/rag/policy.py RequiredEvidencePolicy",
    "multimodal_report": "scripts/multimodal_review_eval.py",
    "multimodal_freeze": "scripts/multimodal_freeze.py",
}


def _iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def classify_failure(target, crop, page_present, table, ocr_block_count) -> str:
    """Which stage ran out of information. One bucket per target."""
    if not page_present:
        return "image_missing"
    if crop.available:
        return "none"
    if "coordinate space" in crop.reason:
        return "bbox_out_of_bounds"
    has_any_bbox = any([target.value_bbox, target.label_bbox,
                        target.cell_bbox, target.row_bbox])
    if not has_any_bbox:
        if ocr_block_count == 0:
            return "no_ocr_text_at_all"
        if table is None:
            return "no_label_bbox_and_no_table_structure"
        return "no_label_bbox_but_table_exists"
    return "all_candidate_regions_below_min_size"


def main() -> None:
    cases = json.loads(REGISTERED.read_text(encoding="utf-8"))["cases"]
    mm = load_multimodal_config()
    vcfg = load_config()
    tcfg = load_tables_config()
    engine = get_ocr_engine("fixture")
    scratch = ROOT / "outputs" / ".baseline_audit_crops"

    rows, documents = [], []
    for case in cases:
        image = ROOT / case["image"]
        document_id = case["document_id"]
        page_present = image.exists()

        ocr = engine.recognize(image) if page_present else None
        struct = detect_table_structure(image, ocr_result=ocr) if page_present else None
        completeness = evaluate_field_completeness(
            ocr, vcfg.drawing_types, vcfg.field_completeness, table=struct,
            min_table_confidence=vcfg.min_table_confidence) if page_present else None

        # What geometry the page actually offers, whether or not anything used it.
        table = recover_grid(image, tcfg.grid, document_id=document_id) \
            if page_present else None
        if table is not None and ocr is not None:
            assign_blocks_to_cells(table, ocr.blocks, tcfg.assignment)
        regions = find_table_regions(cv2.imread(str(image)), tcfg.grid) \
            if page_present else []
        key_value = bool(table is not None and detect_key_value_layout(table))
        cells_with_bbox = sum(1 for c in (table.cells if table else []) if c.bbox)

        pairs = field_pairs_from_completeness(completeness.field_pairs) \
            if completeness else []
        spec = vcfg.drawing_types.get(completeness.drawing_type) if completeness else None
        validation = validate_field_values(pairs, vcfg.value_validation, spec) \
            if completeness else None

        # ---- reproduce the targets exactly as the eval builds them --------
        collector = TriggerCollector(
            document_id, drawing_type=completeness.drawing_type if completeness else None)
        if completeness:
            for name in completeness.missing_fields:
                collector.add(name, REQUIRED_FIELD_MISSING,
                              label_bbox=locate_label_bbox(name, ocr.blocks))
            for label in (validation.invalid_fields if validation else []):
                field_name = label.rpartition(".")[2]
                pair = next((p for p in pairs if p.field_name == field_name), None)
                collector.add(field_name, INVALID_FIELD_VALUE,
                              ocr_value=pair.raw_value if pair else None,
                              value_bbox=pair.bbox if pair else None)
            for name in completeness.isolated_labels or []:
                collector.add(name, "isolated_label_without_value",
                              label_bbox=locate_label_bbox(name, ocr.blocks))
        targets = collector.targets(mm.review)

        case_rows = []
        for target in targets:
            crop = resolve_review_crop(target, image, mm.review, scratch) \
                if page_present else None
            failure = classify_failure(
                target, crop, page_present, table,
                len(ocr.blocks) if ocr else 0) if crop else "image_missing"

            # Could a fallback have found anything? Asked, not used.
            case_rows.append({
                "document_id": document_id,
                "case_id": case["case_id"],
                "field_name": target.field_name,
                "review_reasons": list(target.review_reasons),
                "produced_because": target.review_reasons[0],
                "label_bbox_found": bool(target.label_bbox),
                "value_bbox_found": bool(target.value_bbox),
                "cell_bbox_found": bool(target.cell_bbox),
                "row_bbox_found": bool(target.row_bbox),
                "crop_available": bool(crop and crop.available),
                "region_scope": crop.region_scope if crop else "image_missing",
                "crop_reason": crop.reason if crop else "image missing",
                "failure_stage": failure,
                "fallbacks_that_existed_on_the_page": {
                    "page_image": page_present,
                    "table_bbox": bool(table and table.bbox),
                    "any_cell_bbox": cells_with_bbox > 0,
                    "key_value_layout": key_value,
                    "title_block_region": len(regions) > 0,
                    "neighbouring_ocr_anchors": bool(ocr and len(ocr.blocks) >= 2),
                },
            })
        rows.extend(case_rows)

        documents.append({
            "case_id": case["case_id"],
            "document_id": document_id,
            "image_present": page_present,
            "ocr_blocks": len(ocr.blocks) if ocr else 0,
            "ocr_average_confidence": round(ocr.average_confidence, 4) if ocr else None,
            "drawing_type": completeness.drawing_type if completeness else None,
            "drawing_type_known": bool(
                completeness and completeness.drawing_type != "unknown"),
            "required_field_list_available": bool(spec),
            "missing_fields": list(completeness.missing_fields) if completeness else [],
            "isolated_labels": list(completeness.isolated_labels) if completeness else [],
            "grid_recovered": table is not None,
            "grid_shape": f"{table.row_count}x{table.col_count}" if table else None,
            "key_value_layout": key_value,
            "cells_with_bbox": cells_with_bbox,
            "table_regions_found": len(regions),
            "targets_generated": len(targets),
            "crops_available": sum(1 for r in case_rows if r["crop_available"]),
        })

    # ---- duplicate-region check -----------------------------------------
    available = [r for r in rows if r["crop_available"]]
    duplicates = []
    for i, a in enumerate(available):
        for b in available[i + 1:]:
            if a["document_id"] != b["document_id"]:
                continue
            duplicates.append({"a": a["field_name"], "b": b["field_name"],
                               "document_id": a["document_id"],
                               "note": "same page, both cropped separately"})

    # ---- CELL_SPANS_COLUMNS, measured on the table stage -----------------
    from src.tables.headers import expand_header_paths
    from src.tables.review import collect_review_requests
    from src.vision.ocr_engine import OCR_FIXTURE_DIR
    legacy = Counter()
    for fixture in sorted(OCR_FIXTURE_DIR.glob("*.json")):
        if ".processed" in fixture.stem:
            continue
        img = ROOT / "data" / "drawings" / f"{fixture.stem}.png"
        if not img.exists():
            continue
        tbl = recover_grid(img, tcfg.grid, document_id=fixture.stem)
        if tbl is None:
            continue
        assign_blocks_to_cells(tbl, engine.recognize(img).blocks, tcfg.assignment)
        for request in collect_review_requests(
                tbl, high_risk_fields=tcfg.high_risk_fields,
                header_paths=expand_header_paths(tbl, [])):
            legacy[request.review_reason] += 1

    failure_counts = Counter(r["failure_stage"] for r in rows)
    payload = {
        "note": ("AUDIT ONLY. Current code, unchanged. No algorithm was modified, "
                 "no VLM called, no previous report or cache touched."),
        "real_vlm_calls": 0,
        "code_locations": CODE_MAP,
        "totals": {
            "total_review_targets": len(rows),
            "crop_available": len(available),
            "crop_unavailable": len(rows) - len(available),
            "crop_unavailable_rate": round((len(rows) - len(available)) / len(rows), 4)
            if rows else None,
        },
        "q1_why_each_target_was_produced": dict(Counter(
            r["produced_because"] for r in rows)),
        "q2_targets_with_bbox": [
            {"document_id": r["document_id"], "field_name": r["field_name"],
             "region_scope": r["region_scope"]} for r in available],
        "q3_q4_failure_stage_counts": dict(failure_counts),
        "q5_fallbacks_available_on_failing_pages": [
            {"document_id": r["document_id"], "field_name": r["field_name"],
             **r["fallbacks_that_existed_on_the_page"]}
            for r in rows if not r["crop_available"]],
        "q6_cell_spans_columns_requests_in_table_stage": dict(legacy),
        "q7_pages_with_zero_targets": [
            {"document_id": d["document_id"],
             "drawing_type": d["drawing_type"],
             "drawing_type_known": d["drawing_type_known"],
             "required_field_list_available": d["required_field_list_available"],
             "ocr_blocks": d["ocr_blocks"],
             "why": ("drawing type unknown -> no required-field list -> nothing "
                     "to declare missing -> no review request")
             if not d["drawing_type_known"] else "evidence was complete"}
            for d in documents if d["targets_generated"] == 0],
        "q8_duplicate_crops_on_one_page": duplicates,
        "documents": documents,
        "targets": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    total = payload["totals"]
    lines = [
        "# 复核区域定位基线审计", "",
        "> **仅审计。** 运行当前未修改的代码；未调用 VLM，未改动任何历史报告或缓存。", "",
        f"- 复核目标总数：**{total['total_review_targets']}**",
        f"- 可裁剪：**{total['crop_available']}**　"
        f"不可裁剪：**{total['crop_unavailable']}**"
        f"（{total['crop_unavailable_rate']:.1%}）", "",
        "## 真实代码位置", "", "| 环节 | 位置 |", "|---|---|",
    ]
    for key, value in CODE_MAP.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += ["", "## Q1 每个目标因何产生", "",
              f"`{payload['q1_why_each_target_was_produced']}`", "",
              "## Q3/Q4 失败发生在哪一级", "", "| 失败阶段 | 数量 |", "|---|---|"]
    for stage, count in sorted(failure_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{stage}` | {count} |")
    lines += ["", "## Q5 失败目标所在页面还剩什么可用结构", "",
              "| document | 字段 | 页图 | 表bbox | 单元格bbox | 键值表 | 表格区域 | 邻近锚点 |",
              "|---|---|---|---|---|---|---|---|"]
    for row in payload["q5_fallbacks_available_on_failing_pages"]:
        tick = lambda v: "有" if v else "—"  # noqa: E731
        lines.append(
            f"| {row['document_id']} | {row['field_name']} | {tick(row['page_image'])} | "
            f"{tick(row['table_bbox'])} | {tick(row['any_cell_bbox'])} | "
            f"{tick(row['key_value_layout'])} | {tick(row['title_block_region'])} | "
            f"{tick(row['neighbouring_ocr_anchors'])} |")
    lines += ["", "## Q6 CELL_SPANS_COLUMNS 在表格阶段单独触发的请求数", "",
              f"`{payload['q6_cell_spans_columns_requests_in_table_stage']}`", "",
              "## Q7 零请求页面", ""]
    for row in payload["q7_pages_with_zero_targets"]:
        lines.append(f"- **{row['document_id']}**：type=`{row['drawing_type']}` "
                     f"known={row['drawing_type_known']} "
                     f"ocr_blocks={row['ocr_blocks']} — {row['why']}")
    lines += ["", "## Q8 同页重复裁剪", "",
              f"{len(duplicates)} 对：`{duplicates}`" if duplicates else "无", "",
              "## 逐文档", "",
              "| case | 文档 | OCR块 | 类型 | 类型已知 | 必需字段表 | 网格 | 键值表 | 目标数 | 可裁剪 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for d in documents:
        lines.append(
            f"| {d['case_id']} | {d['document_id']} | {d['ocr_blocks']} | "
            f"{d['drawing_type']} | {'是' if d['drawing_type_known'] else '**否**'} | "
            f"{'有' if d['required_field_list_available'] else '**无**'} | "
            f"{d['grid_shape'] or '—'} | {'是' if d['key_value_layout'] else '否'} | "
            f"{d['targets_generated']} | {d['crops_available']} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"targets {total['total_review_targets']}  "
          f"available {total['crop_available']}  "
          f"unavailable {total['crop_unavailable']} "
          f"({total['crop_unavailable_rate']:.1%})")
    print(f"failure stages: {dict(failure_counts)}")
    print(f"legacy table-stage reasons: {dict(legacy)}")
    print(f"zero-target pages: {[r['document_id'] for r in payload['q7_pages_with_zero_targets']]}")
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
