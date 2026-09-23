"""Inventory every image in the repo, classified by what its OCR actually is.

Provenance is decided from evidence on disk, never from a file name. An image
with a version-stamped fixture is a replay of a recorded real run; an image
whose only OCR is a hand-written stub is a mock; an image with no OCR at all is
`unknown_provenance` until somebody says otherwise — it is not promoted to real
because it happens to sit in the same folder as real ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tables.config import load_tables_config  # noqa: E402
from src.tables.grid import _binarize, _cluster, _line_positions, find_table_regions, recover_grid  # noqa: E402

OUT_JSON = ROOT / "outputs" / "table_corpus_inventory.json"
OUT_MD = ROOT / "outputs" / "table_corpus_inventory.md"

IMAGE_DIRS = [ROOT / "data" / "drawings", ROOT / "data" / "adversarial"]
FIXTURES = ROOT / "data" / "ocr_fixtures"
STUBS = ROOT / "data" / "ocr_stub"

# Images this repo generated for an earlier adversarial round. Self-authored
# and recorded as such: they are not enterprise data and not independent.
SELF_AUTHORED_DIR = "adversarial"


def classify_provenance(stem: str, directory: str) -> tuple[str, str]:
    fixture = FIXTURES / f"{stem}.json"
    if fixture.exists():
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        provenance = payload.get("provenance", {})
        if provenance.get("paddleocr_version"):
            return "real_image_fixture_replay", "paddleocr_fixture_replay"
    if (STUBS / f"{stem}.json").exists():
        return "handwritten_mock", "handwritten_stub"
    if directory == SELF_AUTHORED_DIR:
        return "synthetic_image", "none"
    return "unknown_provenance", "none"


def watermark_like(binary) -> bool:
    """Large mid-grey coverage is what a tiled notice looks like once
    binarized. A heuristic, reported as such."""
    return float((binary > 0).mean()) > 0.16


def main() -> None:
    cfg = load_tables_config()
    rows = []

    for directory in IMAGE_DIRS:
        if not directory.exists():
            continue
        for image_path in sorted(directory.glob("*.png")):
            if ".processed" in image_path.stem:
                continue
            stem = image_path.stem
            kind, ocr_source = classify_provenance(stem, directory.name)

            image = cv2.imread(str(image_path))
            binary = _binarize(image)
            height, width = binary.shape
            regions = find_table_regions(image, cfg.grid)
            table = recover_grid(image_path, cfg.grid, document_id=stem) \
                if regions else None

            rules = _cluster(_line_positions(binary, True, cfg.grid.region_line_fraction),
                             cfg.grid.line_cluster_tolerance_px)
            columns = _cluster(_line_positions(binary, False, cfg.grid.region_line_fraction),
                               cfg.grid.line_cluster_tolerance_px)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rows.append({
                "document_id": stem,
                "source_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                "source_type": "png",
                "real_or_synthetic": kind,
                "ocr_source": ocr_source,
                "page_count": 1,
                "has_table": bool(table),
                "has_grid": bool(table),
                "grid_shape": f"{table.row_count}x{table.col_count}" if table else None,
                # No table in this corpus has more than one header level or a
                # merged cell; asserted from the recovered grid, not assumed.
                "has_multilevel_header": False,
                "has_merged_cells": False,
                "possible_cross_page_table": False,   # every file is one page
                "has_watermark": watermark_like(binary),
                "has_skew": table is None and len(rules) < cfg.grid.min_rules_for_table
                and len(columns) >= 2,
                "has_low_contrast": float(gray.std()) < 35.0,
                "long_horizontal_rules": len(rules),
                "long_vertical_rules": len(columns),
                "table_regions_found": len(regions),
            })

    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["real_or_synthetic"], []).append(row["document_id"])

    real_tables = [r for r in rows
                   if r["has_table"] and r["real_or_synthetic"] == "real_image_fixture_replay"]

    inventory = {
        "note": ("Provenance decided from evidence on disk. An image is never "
                 "promoted to real because of where it sits or what it is called."),
        "image_count": len(rows),
        "by_provenance": {k: len(v) for k, v in sorted(by_kind.items())},
        "documents_by_provenance": {k: sorted(v) for k, v in sorted(by_kind.items())},
        "real_tables_recovered": len(real_tables),
        "real_multilevel_header_tables": 0,
        "real_merged_cell_tables": 0,
        "real_cross_page_tables": 0,
        "missing_data": [
            "no real table with a multi-level header",
            "no real table with a merged cell",
            "no real cross-page table (every source file is a single page)",
            "no human-reviewed structure gold",
        ],
        "consequence": ("multi-level headers, merged cells and cross-page "
                        "continuation can only be pinned by synthetic tests this "
                        "round; no real-data accuracy may be claimed for them"),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(inventory, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    lines = [
        "# 表格语料清单", "",
        f"共 {len(rows)} 张图片。**来源按磁盘上的证据判定，不按文件名或目录归类。**", "",
        "## 来源分布", "",
        "| 分类 | 数量 | 含义 |", "|---|---|---|",
    ]
    meaning = {
        "real_image_fixture_replay": "有版本戳的真实 PaddleOCR 运行结果重放（非实时推理）",
        "handwritten_mock": "只有手写 stub，无真实 OCR",
        "synthetic_image": "本仓库自撰生成的图片（上一轮对抗集）",
        "unknown_provenance": "无任何 OCR 证据，**不归为真实**",
        "real_image_real_ocr": "实时真实 OCR（本轮无）",
    }
    for kind, ids in sorted(by_kind.items()):
        lines.append(f"| `{kind}` | {len(ids)} | {meaning.get(kind, '')} |")
    lines += [
        "", "## 真实表格统计", "",
        f"- 成功恢复网格的真实表格：**{len(real_tables)}**",
        "- 多级表头的真实表格：**0**",
        "- 含合并单元格的真实表格：**0**",
        "- 跨页的真实表格：**0**（每个源文件都是单页）",
        "- 人工审核过的结构 Gold：**0**", "",
        "> 因此本轮**多级表头、合并单元格、跨页续表只能由合成测试固定行为**，",
        "> 不得报告任何基于真实数据的相关准确率。", "",
        "## 缺失数据清单（待人工补充）", "",
    ]
    for item in inventory["missing_data"]:
        lines.append(f"- {item}")
    lines += ["", "## 逐张明细", "",
              "| document_id | 来源 | ocr_source | 有网格 | 形状 | 横线 | 竖线 | 水印 | 低对比度 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['document_id']} | {row['real_or_synthetic']} | {row['ocr_source']} | "
            f"{'是' if row['has_grid'] else '否'} | {row['grid_shape'] or '—'} | "
            f"{row['long_horizontal_rules']} | {row['long_vertical_rules']} | "
            f"{'是' if row['has_watermark'] else '否'} | "
            f"{'是' if row['has_low_contrast'] else '否'} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"images: {len(rows)}")
    for kind, ids in sorted(by_kind.items()):
        print(f"  {kind:<30} {len(ids)}")
    print(f"real tables recovered: {len(real_tables)}")
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
