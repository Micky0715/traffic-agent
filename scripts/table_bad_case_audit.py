"""Audit the ten failure modes named for this stage.

Each case records what went in, what the parser made of it, which layer the
failure belongs to, and whether a local vision pass could plausibly recover it.
Cases that pass are recorded too, but the point of the file is the ones that do
not — an audit showing only successes has not audited anything.

REAL cases run against data/drawings/*.png with their recorded PaddleOCR
fixtures. SYNTHETIC cases are constructed inline, and say so, because the repo
has no real multi-level header, merged cell or cross-page table to exercise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tables.cells import assign_blocks_to_cells, detect_merges, infer_entity_column  # noqa: E402
from src.tables.chunks import build_table_chunks  # noqa: E402
from src.tables.config import load_tables_config  # noqa: E402
from src.tables.continuation import assess_continuation  # noqa: E402
from src.tables.crop import crop_region  # noqa: E402
from src.tables.grid import recover_grid  # noqa: E402
from src.tables.headers import expand_header_paths, header_path_strings, looks_like_repeated_header  # noqa: E402
from src.tables.schemas import ParsedTable, TableCell  # noqa: E402
from src.vision.field_completeness import evaluate_field_completeness  # noqa: E402
from src.vision.config import load_config as load_vision_config  # noqa: E402
from src.vision.ocr_engine import get_ocr_engine  # noqa: E402
from src.vision.table_structure import detect_table_structure  # noqa: E402

OUT_JSON = ROOT / "outputs" / "table_bad_case_audit.json"
OUT_MD = ROOT / "outputs" / "table_bad_case_audit.md"
DRAWINGS = ROOT / "data" / "drawings"

cfg = load_tables_config()
vision_cfg = load_vision_config()
engine = get_ocr_engine("fixture")


def parsed(stem: str, title: str = "", header_rows=()):
    image = DRAWINGS / f"{stem}.png"
    table = recover_grid(image, cfg.grid, document_id=stem, title=title)
    if table is None:
        return image, None, None, None
    ocr = engine.recognize(image)
    report = assign_blocks_to_cells(table, ocr.blocks, cfg.assignment)
    paths = expand_header_paths(table, list(header_rows))
    table.entity_column = infer_entity_column(table)
    return image, table, report, paths


def synth(header_rows_text, data_rows_text, *, width=100, height=20) -> ParsedTable:
    rows = list(header_rows_text) + list(data_rows_text)
    cols = max(len(r) for r in rows)
    cells = [TableCell(cell_id=f"S:r{r}c{c}", row_start=r, row_end=r,
                       col_start=c, col_end=c,
                       bbox=[c * width, r * height, (c + 1) * width, (r + 1) * height],
                       text_raw=(rows[r][c] if c < len(rows[r]) else ""),
                       text_normalized=(rows[r][c] if c < len(rows[r]) else ""))
             for r in range(len(rows)) for c in range(cols)]
    return ParsedTable(document_id="synthetic", table_id="S", page_start=1, page_end=1,
                       bbox=[0, 0, cols * width, len(rows) * height], cells=cells,
                       row_count=len(rows), col_count=cols)


cases = []


def record(**kwargs):
    cases.append(kwargs)


# 1. OCR read the value, the LABEL was destroyed -> no field pair -----------
image, _, _, _ = parsed("FAN-A24-01")
ocr = engine.recognize(image)
completeness = evaluate_field_completeness(
    ocr, vision_cfg.drawing_types, vision_cfg.field_completeness,
    table=detect_table_structure(image, ocr_result=ocr),
    min_table_confidence=vision_cfg.min_table_confidence)
pair_fields = {p.field_name for p in completeness.field_pairs}
record(
    case="1_label_destroyed_value_readable", data="REAL", document="FAN-A24-01",
    input_ocr_text=ocr.text[:150],
    observation=("OCR returned FAN-CAB-24 correctly, but its label was blurred to "
                 "'制编' rather than '控制柜编号', so label->value pairing never fired"),
    field_pairs=sorted(pair_fields),
    succeeded="控制柜编号" in pair_fields,
    failing_layer="label recognition -> field pairing",
    vlm_recoverable=True,
    vlm_reason="LABEL_VALUE_PAIRING_FAILED")

# 2. value present in the body, not in the title block ----------------------
image, table, report, _ = parsed("FAN-A23-01")
ocr = engine.recognize(image)
body_hit = any("电机 M-23" in b.text for b in ocr.blocks)
in_table = bool(table) and any("M-23" in (c.text_normalized or "") for c in table.cells)
record(
    case="2_value_in_body_not_in_title_block", data="REAL", document="FAN-A23-01",
    observation=("the drawing prints 电机 M-23 as a body annotation above the table; "
                 "the title block row for it is covered by the watermark"),
    body_annotation_present=body_hit, value_inside_table=in_table,
    succeeded=in_table,
    failing_layer="table scope — a body annotation is outside the table region",
    vlm_recoverable=True, vlm_reason="WATERMARK_OVER_FIELD")

# 3. watermark over the value column ----------------------------------------
image, table, report, _ = parsed("FAN-A23-01")
record(
    case="3_watermark_covers_values", data="REAL", document="FAN-A23-01",
    grid=(f"{table.row_count}x{table.col_count}" if table else None),
    recovered_cells=[c.text_normalized for c in (table.cells if table else [])][:8],
    observation="watermark text occupies the value column; cells recover the notice",
    succeeded=False,
    failing_layer="OCR recognition — the covering text is what was on the pixels",
    vlm_recoverable=True, vlm_reason="WATERMARK_OVER_FIELD")

# 4. a whole device missing from a multi-device table -----------------------
missing = synth([["设备", "功率"]], [["C21", "45kW"], ["C23", "37kW"]])
missing.entity_column = 0
paths = expand_header_paths(missing, [0])
chunks = build_table_chunks(missing, paths)
entities = {c.entity_id for c in chunks if c.content_type == "table_row"}
record(
    case="4_whole_device_missing_from_table", data="SYNTHETIC",
    observation=("C22's row is absent from the page entirely; nothing in the grid "
                 "indicates a row was ever there"),
    entities_recovered=sorted(e for e in entities if e),
    succeeded=False,
    failing_layer="undetectable at this layer — the grid has no gap to see",
    vlm_recoverable=False,
    vlm_reason="a vision pass on a region that contains nothing cannot recover it")

# 5. 备注 inheriting the neighbouring header --------------------------------
remarks = synth([["设备", "参数", "", "备注"], ["", "功率", "风量", ""]],
                [["A16", "45kW", "28000m3/h", "常用"]])
remarks_paths = expand_header_paths(remarks, [0, 1])
record(
    case="5_remarks_column_inherits_neighbour_header", data="SYNTHETIC",
    header_paths=header_path_strings(remarks_paths),
    observation=("regression: an earlier version carried the blank sub-header "
                 "leftward across a top-level boundary and produced 备注.风量"),
    succeeded=remarks_paths[3] == ["备注"],
    failing_layer="header expansion (fixed)",
    vlm_recoverable=False, vlm_reason="a logic defect, not a recognition one")

# 6. an OCR block straddling two cells --------------------------------------
straddle = synth([["A", "B"]], [["", ""]])


class _Block:
    def __init__(self, text, bbox, confidence=0.9):
        self.text, self.bbox, self.confidence = text, bbox, confidence


straddle_report = assign_blocks_to_cells(straddle, [_Block("45kW", [80, 22, 120, 38])],
                                         cfg.assignment)
record(
    case="6_block_straddles_two_cells", data="SYNTHETIC",
    ambiguous=straddle_report["ambiguous_assignments"],
    observation="centre-point assignment would place this silently in one cell",
    succeeded=bool(straddle_report["ambiguous_assignments"]),
    failing_layer="handled — flagged with both candidates retained",
    vlm_recoverable=True, vlm_reason="CELL_SPANS_COLUMNS")

# 7. skew pushing text out of its cell --------------------------------------
image, table, report, _ = parsed("FAN-A27-01-skewed")
record(
    case="7_skew_breaks_cell_assignment", data="REAL", document="FAN-A27-01-skewed",
    grid_recovered=table is not None,
    observation=("the page is rotated, so the morphological line pass finds no "
                 "axis-aligned rules and no grid is produced"),
    succeeded=False,
    failing_layer="grid detection — lines are not axis-aligned",
    vlm_recoverable=True,
    vlm_reason="a deskew pass or a vision read of the region could recover it")

# 8. a repeated header treated as a data row --------------------------------
repeated = synth([["设备", "功率"]], [["设备", "功率"], ["A16", "45kW"]])
repeated.entity_column = 0
repeated_paths = expand_header_paths(repeated, [0])
repeated_chunks = build_table_chunks(repeated, repeated_paths)
repeated_entities = {c.entity_id for c in repeated_chunks
                     if c.content_type == "table_row"}
record(
    case="8_repeated_header_as_data_row", data="SYNTHETIC",
    row_entities=sorted(e for e in repeated_entities if e),
    observation="a header reprinted at the top of a continued page is not a device",
    succeeded=("设备" not in repeated_entities and
               looks_like_repeated_header(repeated, 1, [0])),
    failing_layer="handled — repeated header rows are dropped",
    vlm_recoverable=False, vlm_reason="structural, not a recognition problem")

# 9. two different tables merged because the column counts match ------------
left = synth([["设备", "功率"]], [["A16", "45kW"]])
left.document_id = "doc"
left.bbox = [60.0, 200.0, 940.0, 260.0]
right = synth([["站点", "编号"]], [["天河", "S1"]])
right.document_id = "doc"
right.page_start = right.page_end = 2
right.bbox = [60.0, 200.0, 940.0, 260.0]
merge_result = assess_continuation(
    left, right, cfg.continuation,
    previous_header_paths=["设备", "功率"], current_header_paths=["站点", "编号"])
record(
    case="9_different_tables_merged_on_column_count", data="SYNTHETIC",
    continuation_status=merge_result["status"], reasons=merge_result["reasons"],
    observation="same column count, same extent, adjacent pages, different headers",
    succeeded=merge_result["status"] == "separate_table",
    failing_layer="handled — header compatibility is a required signal",
    vlm_recoverable=False, vlm_reason="structural")

# 10. crop running off the page / degenerate -------------------------------
image = DRAWINGS / "FAN-MULTI-01.png"
clipped = crop_region(image, [-40.0, -40.0, 100.0, 100.0], cfg.crop,
                      document_id="FAN-MULTI-01", page=1, region_type="table",
                      output_dir=ROOT / "outputs" / "table_crops")
tiny_cfg = cfg.crop.model_copy(update={"padding_px": 0, "min_crop_size_px": 10})
tiny = crop_region(image, [10.0, 10.0, 12.0, 12.0], tiny_cfg,
                   document_id="FAN-MULTI-01", page=1, region_type="cell",
                   output_dir=ROOT / "outputs" / "table_crops")
record(
    case="10_crop_out_of_bounds_or_degenerate", data="REAL", document="FAN-MULTI-01",
    clipped_crop_saved=clipped is not None,
    clipped_flag=(clipped.clipped if clipped else None),
    degenerate_crop_returned_none=tiny is None,
    observation="a negative-origin box is clipped; a 2px box returns None",
    succeeded=(clipped is not None and clipped.clipped and tiny is None),
    failing_layer="handled",
    vlm_recoverable=False, vlm_reason="not a recognition problem")

# 11. false uncertainty on a clean real table ------------------------------
image, table, report, _ = parsed("FAN-MULTI-01", title="A16/A17/A18风机组接线图")
correct = {(0, 0): "图号", (4, 0): "A16功率", (4, 1): "45kW", (7, 1): "55kW"}
all_correct = all(table.cell_at(r, c).text_normalized == v
                  for (r, c), v in correct.items())
record(
    case="11_false_uncertainty_from_tall_ocr_boxes", data="REAL",
    document="FAN-MULTI-01",
    ambiguous_count=len(report["ambiguous_assignments"]),
    total_cells=len(table.cells),
    sampled_cells_all_correct=all_correct,
    observation=("every cell's text is correct, yet several are flagged uncertain "
                 "because PaddleOCR's boxes are taller than the printed rows and "
                 "coverage falls below min_coverage_ratio"),
    succeeded=False,
    failing_layer="threshold calibration, not assignment",
    vlm_recoverable=False,
    vlm_reason=("min_coverage_ratio needs calibrating against human-confirmed "
                "cells; it was NOT tuned on this sample"))

report_payload = {
    "note": ("Failure-mode audit for table structure recovery. REAL cases use "
             "recorded PaddleOCR fixtures replayed over the original images; "
             "SYNTHETIC cases are constructed inline because the repo has no "
             "real multi-level header, merged cell or cross-page table."),
    "vlm_invoked": False,
    "case_count": len(cases),
    "passed": sum(1 for c in cases if c.get("succeeded")),
    "failed": sum(1 for c in cases if not c.get("succeeded")),
    "real_cases": sum(1 for c in cases if c["data"] == "REAL"),
    "synthetic_cases": sum(1 for c in cases if c["data"] == "SYNTHETIC"),
    "cases": cases,
}
OUT_JSON.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2,
                               default=str), encoding="utf-8")

lines = ["# 表格 Bad Case 审计", "",
         f"共 {len(cases)} 个场景，通过 {report_payload['passed']}，"
         f"未通过 {report_payload['failed']}。",
         f"真实数据场景 {report_payload['real_cases']}，合成场景 "
         f"{report_payload['synthetic_cases']}。**本轮未调用 VLM。**", "",
         "| 场景 | 数据 | 结果 | 失败层 | 可否局部 VLM 补救 |", "|---|---|---|---|---|"]
for case in cases:
    lines.append(f"| {case['case']} | {case['data']} | "
                 f"{'✅ 通过' if case.get('succeeded') else '❌ 未通过'} | "
                 f"{case['failing_layer']} | "
                 f"{'可' if case.get('vlm_recoverable') else '不可'} |")
lines.append("")
for case in cases:
    lines += [f"## {case['case']}", "",
              f"- 数据：**{case['data']}**"
              + (f"　文档：`{case['document']}`" if case.get("document") else ""),
              f"- 观察：{case['observation']}",
              f"- 结果：{'通过' if case.get('succeeded') else '**未通过**'}",
              f"- 失败层：{case['failing_layer']}",
              f"- 局部 VLM：{'可补救' if case.get('vlm_recoverable') else '不可补救'}"
              f"（{case.get('vlm_reason', '')}）", ""]
OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print(f"cases {len(cases)}  passed {report_payload['passed']}  "
      f"failed {report_payload['failed']}")
for case in cases:
    print(f"  {'PASS' if case.get('succeeded') else 'FAIL'}  {case['data']:<10} "
          f"{case['case']}")
print(f"\n-> {OUT_JSON}\n-> {OUT_MD}")
