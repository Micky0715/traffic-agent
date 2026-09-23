# 表格 Bad Case 审计

共 11 个场景，通过 5，未通过 6。
真实数据场景 6，合成场景 5。**本轮未调用 VLM。**

| 场景 | 数据 | 结果 | 失败层 | 可否局部 VLM 补救 |
|---|---|---|---|---|
| 1_label_destroyed_value_readable | REAL | ❌ 未通过 | label recognition -> field pairing | 可 |
| 2_value_in_body_not_in_title_block | REAL | ❌ 未通过 | table scope — a body annotation is outside the table region | 可 |
| 3_watermark_covers_values | REAL | ❌ 未通过 | OCR recognition — the covering text is what was on the pixels | 可 |
| 4_whole_device_missing_from_table | SYNTHETIC | ❌ 未通过 | undetectable at this layer — the grid has no gap to see | 不可 |
| 5_remarks_column_inherits_neighbour_header | SYNTHETIC | ✅ 通过 | header expansion (fixed) | 不可 |
| 6_block_straddles_two_cells | SYNTHETIC | ✅ 通过 | handled — flagged with both candidates retained | 可 |
| 7_skew_breaks_cell_assignment | REAL | ❌ 未通过 | grid detection — lines are not axis-aligned | 可 |
| 8_repeated_header_as_data_row | SYNTHETIC | ✅ 通过 | handled — repeated header rows are dropped | 不可 |
| 9_different_tables_merged_on_column_count | SYNTHETIC | ✅ 通过 | handled — header compatibility is a required signal | 不可 |
| 10_crop_out_of_bounds_or_degenerate | REAL | ✅ 通过 | handled | 不可 |
| 11_false_uncertainty_from_tall_ocr_boxes | REAL | ❌ 未通过 | threshold calibration, not assignment | 不可 |

## 1_label_destroyed_value_readable

- 数据：**REAL**　文档：`FAN-A24-01`
- 观察：OCR returned FAN-CAB-24 correctly, but its label was blurred to '制编' rather than '控制柜编号', so label->value pairing never fired
- 结果：**未通过**
- 失败层：label recognition -> field pairing
- 局部 VLM：可补救（LABEL_VALUE_PAIRING_FAILED）

## 2_value_in_body_not_in_title_block

- 数据：**REAL**　文档：`FAN-A23-01`
- 观察：the drawing prints 电机 M-23 as a body annotation above the table; the title block row for it is covered by the watermark
- 结果：**未通过**
- 失败层：table scope — a body annotation is outside the table region
- 局部 VLM：可补救（WATERMARK_OVER_FIELD）

## 3_watermark_covers_values

- 数据：**REAL**　文档：`FAN-A23-01`
- 观察：watermark text occupies the value column; cells recover the notice
- 结果：**未通过**
- 失败层：OCR recognition — the covering text is what was on the pixels
- 局部 VLM：可补救（WATERMARK_OVER_FIELD）

## 4_whole_device_missing_from_table

- 数据：**SYNTHETIC**
- 观察：C22's row is absent from the page entirely; nothing in the grid indicates a row was ever there
- 结果：**未通过**
- 失败层：undetectable at this layer — the grid has no gap to see
- 局部 VLM：不可补救（a vision pass on a region that contains nothing cannot recover it）

## 5_remarks_column_inherits_neighbour_header

- 数据：**SYNTHETIC**
- 观察：regression: an earlier version carried the blank sub-header leftward across a top-level boundary and produced 备注.风量
- 结果：通过
- 失败层：header expansion (fixed)
- 局部 VLM：不可补救（a logic defect, not a recognition one）

## 6_block_straddles_two_cells

- 数据：**SYNTHETIC**
- 观察：centre-point assignment would place this silently in one cell
- 结果：通过
- 失败层：handled — flagged with both candidates retained
- 局部 VLM：可补救（CELL_SPANS_COLUMNS）

## 7_skew_breaks_cell_assignment

- 数据：**REAL**　文档：`FAN-A27-01-skewed`
- 观察：the page is rotated, so the morphological line pass finds no axis-aligned rules and no grid is produced
- 结果：**未通过**
- 失败层：grid detection — lines are not axis-aligned
- 局部 VLM：可补救（a deskew pass or a vision read of the region could recover it）

## 8_repeated_header_as_data_row

- 数据：**SYNTHETIC**
- 观察：a header reprinted at the top of a continued page is not a device
- 结果：通过
- 失败层：handled — repeated header rows are dropped
- 局部 VLM：不可补救（structural, not a recognition problem）

## 9_different_tables_merged_on_column_count

- 数据：**SYNTHETIC**
- 观察：same column count, same extent, adjacent pages, different headers
- 结果：通过
- 失败层：handled — header compatibility is a required signal
- 局部 VLM：不可补救（structural）

## 10_crop_out_of_bounds_or_degenerate

- 数据：**REAL**　文档：`FAN-MULTI-01`
- 观察：a negative-origin box is clipped; a 2px box returns None
- 结果：通过
- 失败层：handled
- 局部 VLM：不可补救（not a recognition problem）

## 11_false_uncertainty_from_tall_ocr_boxes

- 数据：**REAL**　文档：`FAN-MULTI-01`
- 观察：every cell's text is correct, yet several are flagged uncertain because PaddleOCR's boxes are taller than the printed rows and coverage falls below min_coverage_ratio
- 结果：**未通过**
- 失败层：threshold calibration, not assignment
- 局部 VLM：不可补救（min_coverage_ratio needs calibrating against human-confirmed cells; it was NOT tuned on this sample）
