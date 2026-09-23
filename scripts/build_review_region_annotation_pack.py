"""Build a local pack a person can review tomorrow, page by page.

Produces overlays, crops and an HTML sheet showing EVERY candidate the resolver
generated — not only the one it chose. Showing only the winner would invite the
reviewer to agree with it, and a gold set drawn on top of the prediction is the
prediction.

Also emits data/review_region_gold_unreviewed.jsonl: one pending record per
(page, field), every conclusion field empty. Nothing here writes a gold bbox.

    python scripts/build_review_region_annotation_pack.py

No network, no model, no paid API. Existing reviewed records are never
overwritten — they are carried through untouched and reported.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.tables.cells import assign_blocks_to_cells  # noqa: E402
from src.tables.config import load_tables_config  # noqa: E402
from src.tables.grid import recover_grid  # noqa: E402
from src.vision.config import load_config  # noqa: E402
from src.vision.field_completeness import evaluate_field_completeness  # noqa: E402
from src.vision.ocr_engine import get_ocr_engine  # noqa: E402
from src.vision.region_gold import (  # noqa: E402
    HUMAN_REVIEWED, RegionGoldRecord, read_jsonl, status_summary, write_jsonl,
)
from src.vision.review_region_config import load_review_region_config  # noqa: E402
from src.vision.review_region_resolver import resolve_review_regions  # noqa: E402
from src.vision.table_structure import detect_table_structure  # noqa: E402
from src.vision.value_validation import (  # noqa: E402
    field_pairs_from_completeness, validate_field_values,
)

PACK = ROOT / "outputs" / "review_region_annotation_pack"
GOLD = ROOT / "data" / "review_region_gold_unreviewed.jsonl"
BASELINE = ROOT / "outputs" / "review_region_baseline_audit.json"

# BGR. One colour per strategy, so a reviewer can tell at a glance which box
# came from reading a label and which from framing a whole table.
STRATEGY_COLOUR = {
    "exact_label_right": (60, 170, 60),        # green
    "alias_label_right": (200, 160, 40),       # teal
    "table_value_cell": (200, 90, 200),        # magenta
    "structural_row_inference": (200, 140, 255),
    "table_or_title_block": (40, 140, 240),    # orange
    "full_page_diagnostic": (60, 60, 220),     # red
}
OCR_COLOUR = (180, 180, 180)
CELL_COLOUR = (120, 200, 230)
PREVIEW_MAX_LONG_EDGE = 1400


def draw(frame, bbox, colour, thickness=2):
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    height, width = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width - 1, x1), min(height - 1, y1)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, thickness)
    return x0, y0


def build_page(document_id: str, image: Path, cfgs) -> Dict:
    vcfg, tcfg, rcfg = cfgs
    frame = cv2.imread(str(image))
    height, width = frame.shape[:2]
    ocr = get_ocr_engine("fixture").recognize(image)
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

    field_reasons: Dict[str, List[str]] = {}
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

    # ---- overlay: original untouched, a copy is drawn on ------------------
    overlay = frame.copy()
    for block in ocr.blocks:
        bbox = list(getattr(block, "bbox", []) or [])
        if len(bbox) == 4:
            draw(overlay, bbox, OCR_COLOUR, 1)
    for cell in (table.cells if table is not None else []):
        if cell.bbox:
            draw(overlay, cell.bbox, CELL_COLOUR, 1)
    for region in regions:
        if not region.bbox:
            continue
        colour = STRATEGY_COLOUR.get(region.strategy, (0, 0, 0))
        x0, y0 = draw(overlay, region.bbox, colour, 3)
        cv2.putText(overlay, region.strategy, (x0 + 4, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    (PACK / "pages").mkdir(parents=True, exist_ok=True)
    (PACK / "overlays").mkdir(parents=True, exist_ok=True)
    (PACK / "crops").mkdir(parents=True, exist_ok=True)

    shutil.copyfile(image, PACK / "pages" / f"{document_id}.png")
    overlay_path = PACK / "overlays" / f"{document_id}.overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    # A downscaled preview for very large pages; the factor is recorded so a
    # coordinate read off the preview can be mapped back to the original.
    long_edge = max(width, height)
    preview_scale = 1.0
    if long_edge > PREVIEW_MAX_LONG_EDGE:
        preview_scale = PREVIEW_MAX_LONG_EDGE / long_edge
        preview = cv2.resize(overlay, (int(width * preview_scale),
                                       int(height * preview_scale)))
        cv2.imwrite(str(PACK / "overlays" / f"{document_id}.preview.png"), preview)

    crops = []
    for index, region in enumerate(regions):
        if not region.bbox:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in region.bbox]
        patch = frame[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
        if patch.size == 0:
            continue
        name = f"{document_id}.{index}.{region.strategy}.png"
        cv2.imwrite(str(PACK / "crops" / name), patch)
        crops.append({"file": f"crops/{name}", "region_index": index})

    return {
        "document_id": document_id,
        "image_path": str(image.relative_to(ROOT)).replace("\\", "/"),
        "image_size": [width, height],
        "preview_scale": preview_scale,
        "page_file": f"pages/{document_id}.png",
        "overlay_file": f"overlays/{document_id}.overlay.png",
        "preview_file": (f"overlays/{document_id}.preview.png"
                         if preview_scale != 1.0 else None),
        "drawing_type": completeness.drawing_type,
        "drawing_type_known": completeness.drawing_type != "unknown",
        "ocr_block_count": len(ocr.blocks),
        "ocr_blocks": [{"text": getattr(b, "text", ""),
                        "bbox": list(getattr(b, "bbox", []) or [])}
                       for b in ocr.blocks],
        "table_bbox": list(table.bbox) if table is not None and table.bbox else None,
        "table_shape": (f"{table.row_count}x{table.col_count}"
                        if table is not None else None),
        "cell_bboxes": [{"row": c.row_start, "col": c.col_start,
                         "bbox": list(c.bbox or []),
                         "text": c.text_normalized}
                        for c in (table.cells if table is not None else [])
                        if c.bbox],
        "missing_fields": list(completeness.missing_fields),
        "field_reasons": field_reasons,
        "target_fields": sorted(field_reasons),
        "candidates": [r.to_dict() for r in regions],
        "crops": crops,
    }


def render_html(pages: List[Dict]) -> str:
    rows = []
    legend = "".join(
        f'<span class="chip" style="border-color:rgb({c[2]},{c[1]},{c[0]})">{s}</span>'
        for s, c in STRATEGY_COLOUR.items())
    for page in pages:
        cands = "".join(
            f"<tr><td>{i}</td><td><code>{c['strategy']}</code></td>"
            f"<td>{c['trust_level']}</td>"
            f"<td>{', '.join(c['target_fields'])}</td>"
            f"<td>{[round(v) for v in c['bbox']]}</td>"
            f"<td>{c['area_ratio']:.1%}</td>"
            f"<td>{'是' if c['localization_uncertain'] else '否'}</td>"
            f"<td>{'是' if c['answer_eligible'] else '否'}</td>"
            f"<td>{'; '.join(c['source_anchors'])}</td></tr>"
            for i, c in enumerate(page["candidates"]))
        crops = "".join(
            f'<figure><img src="{c["file"]}"><figcaption>候选 {c["region_index"]}'
            f'</figcaption></figure>' for c in page["crops"])
        fields = "".join(
            f"<tr><td><b>{name}</b></td><td>{', '.join(reasons)}</td>"
            f"<td class=blank>gold_bbox = ____________</td>"
            f"<td class=blank>region_type = ____________</td>"
            f"<td class=blank>field_visible = ___</td>"
            f"<td class=blank>value_visible = ___</td></tr>"
            for name, reasons in sorted(page["field_reasons"].items()))
        src = page["preview_file"] or page["overlay_file"]
        scale_note = (f"（预览缩放 {page['preview_scale']:.3f}，"
                      f"原图坐标 = 预览坐标 / {page['preview_scale']:.3f}）"
                      if page["preview_file"] else "（原始尺寸）")
        rows.append(f"""
<section>
  <h2>{page['document_id']}</h2>
  <p class="meta">尺寸 {page['image_size'][0]}×{page['image_size'][1]} ·
     图纸类型 <code>{page['drawing_type']}</code>
     （已知：{'是' if page['drawing_type_known'] else '<b>否</b>'}） ·
     OCR 块 {page['ocr_block_count']} ·
     表格 {page['table_shape'] or '—'}</p>
  <p class="meta">{scale_note}</p>
  <img class="page" src="{src}">
  <h3>系统候选（全部，不只是被选中的）</h3>
  <table><thead><tr><th>#</th><th>strategy</th><th>trust</th><th>字段</th>
    <th>bbox</th><th>面积</th><th>定位不确定</th><th>可自动答</th><th>锚点</th>
  </tr></thead><tbody>{cands or '<tr><td colspan=9>无候选</td></tr>'}</tbody></table>
  <h3>候选裁剪预览</h3>
  <div class="crops">{crops or '无'}</div>
  <h3>需要你填写的 Gold（逐字段）</h3>
  <table><thead><tr><th>字段</th><th>触发原因</th><th colspan=4>人工填写</th>
  </tr></thead><tbody>{fields or '<tr><td colspan=6>本页无待审字段</td></tr>'}</tbody></table>
</section>""")

    return f"""<!doctype html><meta charset="utf-8">
<title>bbox Gold 人工审核包</title>
<style>
 body{{font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;margin:24px;max-width:1200px}}
 section{{border-top:2px solid #ddd;padding-top:16px;margin-top:32px}}
 img.page{{max-width:100%;border:1px solid #bbb}}
 table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}}
 th,td{{border:1px solid #ccc;padding:4px 6px;text-align:left;vertical-align:top}}
 th{{background:#f4f4f4}}
 td.blank{{background:#fffbe6;font-family:monospace}}
 .crops{{display:flex;flex-wrap:wrap;gap:12px}}
 .crops img{{max-height:150px;border:1px solid #bbb}}
 figcaption{{font-size:12px;color:#666}}
 .chip{{display:inline-block;padding:2px 8px;margin:2px;border:3px solid;border-radius:4px;font-size:12px}}
 .meta{{color:#555}}
 .warn{{background:#fff3f3;border-left:4px solid #c00;padding:8px 12px}}
</style>
<h1>bbox Gold 人工审核包</h1>
<p class="warn"><b>不要顺着系统候选框标。</b>先看原图判断字段值在哪里，再看候选框。
候选框是被评测的对象，不是参考答案。若某字段肉眼不可见，标 <code>field_visible=false</code>
并写 <code>unlocatable_reason</code>，不要勉强画一个框。</p>
<p>图例：{legend}
<span class="chip" style="border-color:#b4b4b4">OCR 块</span>
<span class="chip" style="border-color:#e6c878">表格单元格</span></p>
<p>填写方法见 <code>docs/review_region_annotation_guide.md</code>。
填完请编辑 <code>data/review_region_gold_unreviewed.jsonl</code>，
然后运行 <code>python scripts/validate_review_region_gold.py</code>。</p>
{''.join(rows)}
"""


def render_markdown(pages: List[Dict]) -> str:
    lines = ["# bbox Gold 人工审核表", "",
             "> **不要顺着系统候选框标。** 候选框是被评测的对象，不是参考答案。", "",
             "填写 `data/review_region_gold_unreviewed.jsonl`，"
             "再跑 `python scripts/validate_review_region_gold.py`。", ""]
    for page in pages:
        lines += [
            f"## {page['document_id']}", "",
            f"- 原图：`{page['page_file']}`　叠加图：`{page['overlay_file']}`",
            f"- 尺寸 {page['image_size'][0]}×{page['image_size'][1]}，"
            f"类型 `{page['drawing_type']}`"
            f"（已知：{'是' if page['drawing_type_known'] else '**否**'}），"
            f"OCR 块 {page['ocr_block_count']}，表格 {page['table_shape'] or '—'}", "",
            "### 系统候选（全部）", "",
            "| # | strategy | trust | 字段 | bbox | 面积 | 可自动答 |",
            "|---|---|---|---|---|---|---|"]
        for index, cand in enumerate(page["candidates"]):
            lines.append(
                f"| {index} | `{cand['strategy']}` | {cand['trust_level']} | "
                f"{', '.join(cand['target_fields'])} | "
                f"{[round(v) for v in cand['bbox']]} | {cand['area_ratio']:.1%} | "
                f"{'是' if cand['answer_eligible'] else '否'} |")
        lines += ["", "### 待填 Gold", "",
                  "| 字段 | 触发原因 | gold_bbox | gold_region_type | field_visible | value_visible |",
                  "|---|---|---|---|---|---|"]
        for name, reasons in sorted(page["field_reasons"].items()):
            lines.append(f"| **{name}** | {', '.join(reasons)} | ______ | ______ | __ | __ |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    cfgs = (load_config(), load_tables_config(), load_review_region_config())
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) \
        if BASELINE.exists() else {"documents": []}
    documents = sorted({d["document_id"] for d in baseline.get("documents", [])})

    pages = []
    for document_id in documents:
        image = ROOT / "data" / "drawings" / f"{document_id}.png"
        if image.exists():
            pages.append(build_page(document_id, image, cfgs))

    # ---- pending gold records -------------------------------------------
    existing = read_jsonl(GOLD)
    by_id = {r.record_id: r for r in existing}
    reviewed_kept = sum(1 for r in existing if r.label_status == HUMAN_REVIEWED)

    records = []
    created = 0
    for page in pages:
        for name in page["target_fields"]:
            record = RegionGoldRecord.pending(
                document_id=page["document_id"], page_no=1,
                image_path=ROOT / page["image_path"], target_field=name,
                notes="pending human review; "
                      f"triggered by {','.join(page['field_reasons'][name])}")
            record.image_path = page["image_path"]
            previous = by_id.get(record.record_id)
            if previous is not None:
                # Never overwrite a person's work.
                records.append(previous)
                continue
            records.append(record)
            created += 1
    for record_id, record in by_id.items():
        if record_id not in {r.record_id for r in records}:
            records.append(record)

    write_jsonl(records, GOLD)

    manifest = {
        "note": ("Every candidate the resolver produced is shown, not only the "
                 "one it selected. A gold drawn on top of the prediction is the "
                 "prediction."),
        "real_vlm_calls": 0,
        "external_api_calls": 0,
        "pages": len(pages),
        "pending_records_created": created,
        "human_reviewed_records_preserved": reviewed_kept,
        "total_records": len(records),
        "status_summary": status_summary(records),
        "candidates_by_strategy": dict(Counter(
            c["strategy"] for p in pages for c in p["candidates"])),
        "gold_file": "data/review_region_gold_unreviewed.jsonl",
        "page_details": pages,
    }
    (PACK / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (PACK / "index.html").write_text(render_html(pages), encoding="utf-8")
    (PACK / "review_sheet.md").write_text(render_markdown(pages), encoding="utf-8")

    print(f"pages {len(pages)}  pending records created {created}  "
          f"human-reviewed preserved {reviewed_kept}  total {len(records)}")
    print(f"status: {manifest['status_summary']}")
    print(f"candidates: {manifest['candidates_by_strategy']}")
    print(f"-> {PACK / 'index.html'}")
    print(f"-> {PACK / 'review_sheet.md'}")
    print(f"-> {GOLD}")


if __name__ == "__main__":
    main()
