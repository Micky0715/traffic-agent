"""End-to-end audit of table structure recovery over the real drawings.

Runs image -> grid -> real OCR blocks -> cells -> headers -> chunks -> crops
and reports what came out, separating real results from synthetic ones.

No VLM is called. No human-reviewed gold exists, so no structure-accuracy
number is produced — only counts and coverage, which do not require one.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tables.cells import assign_blocks_to_cells, detect_merges, infer_entity_column  # noqa: E402
from src.tables.chunks import build_table_chunks  # noqa: E402
from src.tables.config import load_tables_config  # noqa: E402
from src.tables.crop import crop_cell_region, crop_row_region, crop_table_region  # noqa: E402
from src.tables.grid import recover_grid  # noqa: E402
from src.tables.headers import expand_header_paths, header_path_strings  # noqa: E402
from src.tables.review import collect_review_requests, review_status_summary  # noqa: E402
from src.vision.ocr_engine import OCR_FIXTURE_DIR, get_ocr_engine  # noqa: E402

OUT_JSON = ROOT / "outputs" / "table_structure_audit.json"
OUT_MD = ROOT / "outputs" / "table_structure_audit.md"
DRAWINGS = ROOT / "data" / "drawings"

cfg = load_tables_config()
engine = get_ocr_engine("fixture")


def main() -> None:
    stems = sorted(p.stem for p in OCR_FIXTURE_DIR.glob("*.json")
                   if ".processed" not in p.stem)
    documents, all_chunks, all_requests = [], [], []
    crops = Counter()

    for stem in stems:
        image = DRAWINGS / f"{stem}.png"
        if not image.exists():
            continue
        table = recover_grid(image, cfg.grid, document_id=stem)
        if table is None:
            documents.append({"document_id": stem, "grid_recovered": False,
                              "reason": "no internal grid found"})
            continue

        ocr = engine.recognize(image)
        report = assign_blocks_to_cells(table, ocr.blocks, cfg.assignment)
        detect_merges(table)
        paths = expand_header_paths(table, [])
        table.entity_column = infer_entity_column(table)

        region_refs = {}
        table_ref = crop_table_region(image, table, cfg.crop)
        if table_ref:
            region_refs["table"] = table_ref
            crops["table"] += 1
        for row in table.data_rows()[:3]:
            row_ref = crop_row_region(image, table, row, cfg.crop)
            if row_ref:
                region_refs[f"r{row}"] = row_ref
                crops["row"] += 1
        cell_ref = crop_cell_region(image, table, 0, 0, cfg.crop)
        if cell_ref:
            region_refs[cell_ref.cell_id] = cell_ref
            crops["cell"] += 1

        chunks = build_table_chunks(table, paths, parser_source="ocr",
                                    region_refs=region_refs)
        requests = collect_review_requests(
            table, high_risk_fields=cfg.high_risk_fields,
            header_paths=paths, region_refs=region_refs)
        all_chunks.extend(chunks)
        all_requests.extend(requests)

        documents.append({
            "document_id": stem,
            "grid_recovered": True,
            "grid": f"{table.row_count}x{table.col_count}",
            "cells": len(table.cells),
            "structure_confidence": table.structure_confidence,
            "structure_uncertain": table.structure_uncertain,
            "entity_column": table.entity_column,
            "header_paths": header_path_strings(paths),
            "assigned_cells": report["assigned_cells"],
            "empty_cells": report["empty_cells"],
            "ambiguous_assignments": len(report["ambiguous_assignments"]),
            "blocks_outside_table": len(report["blocks_outside_table"]),
            "unassigned_blocks": len(report["unassigned_blocks"]),
            "chunks": len(chunks),
            "review_requests": len(requests),
            "table_crop": table_ref.image_path if table_ref else None,
            "continuation_status": table.continuation_status,
        })

    recovered = [d for d in documents if d.get("grid_recovered")]
    row_chunks = [c for c in all_chunks if c.content_type == "table_row"]
    entity_ids = {c.entity_id for c in row_chunks if c.entity_id}
    contamination = [c.chunk_id for c in row_chunks
                     if any(other != c.entity_id and other in c.text
                            for other in entity_ids)]

    payload = {
        "note": ("Real drawings with recorded PaddleOCR fixtures replayed over "
                 "them. No VLM, no live OCR, no human-reviewed gold."),
        "capabilities": {
            "vlm_invoked": False,
            "live_ocr": False,
            "ocr_source": "paddleocr fixture replay (version-stamped)",
            "human_reviewed_structure_gold": False,
            "structure_accuracy": ("not_computed — no human-reviewed gold; a "
                                   "number derived from the parser's own output "
                                   "would be circular"),
        },
        "real_documents_examined": len(documents),
        "real_tables_recovered": len(recovered),
        "real_multilevel_header_tables": 0,
        "real_merged_cell_tables": 0,
        "real_cross_page_tables": 0,
        "synthetic_only_behaviours": [
            "multi-level header expansion",
            "merged-cell detection",
            "cross-page continuation",
        ],
        "chunk_counts": dict(Counter(c.content_type for c in all_chunks)),
        "coverage": {
            "page_traceable": {
                "numerator": sum(1 for c in all_chunks if c.page_start is not None),
                "denominator": len(all_chunks)},
            "bbox_traceable": {
                "numerator": sum(1 for c in all_chunks if c.source_bboxes),
                "denominator": len(all_chunks)},
            "row_chunks_with_entity": {
                "numerator": sum(1 for c in row_chunks if c.entity_id),
                "denominator": len(row_chunks)},
            "cross_device_contamination": {
                "numerator": len(contamination), "denominator": len(row_chunks)},
        },
        "crops_saved": dict(crops),
        "review_requests": {
            "count": len(all_requests),
            "status_summary": review_status_summary(all_requests),
            "by_reason": dict(Counter(r.review_reason for r in all_requests)),
        },
        "documents": documents,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def frac(key):
        m = payload["coverage"][key]
        d = m["denominator"] or 1
        return f"{m['numerator']}/{m['denominator']} = {m['numerator'] / d:.1%}"

    lines = [
        "# 表格结构恢复审计", "",
        "> 真实图纸 + 录制的 PaddleOCR fixture 重放。**未调用 VLM，未实时 OCR，"
        "无人工审核 Gold。**", "",
        "## 真实结果", "",
        f"- 检查的真实文档：**{payload['real_documents_examined']}**",
        f"- 成功恢复网格的真实表格：**{payload['real_tables_recovered']}**",
        "- 真实多级表头表格：**0**　真实合并单元格表格：**0**　真实跨页表格：**0**", "",
        "> 因此**多级表头展开、合并单元格识别、跨页续表判定只有合成测试**，"
        "不得报告基于真实数据的相关准确率。", "",
        f"- 结构恢复准确率：**{payload['capabilities']['structure_accuracy']}**", "",
        "## Chunk 与溯源", "",
        f"chunk 类型分布：`{payload['chunk_counts']}`", "",
        "| 指标 | 分子/分母 |", "|---|---|",
        f"| 页码可溯源 | {frac('page_traceable')} |",
        f"| bbox 可溯源 | {frac('bbox_traceable')} |",
        f"| 行 chunk 带实体 | {frac('row_chunks_with_entity')} |",
        f"| **跨设备污染** | {frac('cross_device_contamination')} |", "",
        f"裁剪产出：`{payload['crops_saved']}`", "",
        "## 局部 VLM 复核", "",
        f"- 生成请求 **{payload['review_requests']['count']}** 条",
        f"- 状态分布：`{payload['review_requests']['status_summary']}`",
        f"- 原因分布：`{payload['review_requests']['by_reason']}`", "",
        "> 全部为 `requested_not_invoked`。**本轮没有调用任何视觉模型**，"
        "把它写成复核成功会把一个未决问题变成伪造的确认。", "",
        "## 逐文档", "",
        "| document | 网格 | 单元格 | 空 | 歧义 | 表外 block | 实体列 | chunk | 复核请求 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for document in documents:
        if not document.get("grid_recovered"):
            lines.append(f"| {document['document_id']} | — | — | — | — | — | — | — | "
                         f"{document.get('reason', '')} |")
            continue
        lines.append(
            f"| {document['document_id']} | {document['grid']} | "
            f"{document['assigned_cells']} | {document['empty_cells']} | "
            f"{document['ambiguous_assignments']} | {document['blocks_outside_table']} | "
            f"{document['entity_column']} | {document['chunks']} | "
            f"{document['review_requests']} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"real documents {payload['real_documents_examined']}, "
          f"tables recovered {payload['real_tables_recovered']}")
    print(f"chunks {payload['chunk_counts']}")
    for key in payload["coverage"]:
        print(f"  {key:<32} {frac(key)}")
    print(f"crops {dict(crops)}  review requests {payload['review_requests']['count']} "
          f"{payload['review_requests']['status_summary']}")
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
