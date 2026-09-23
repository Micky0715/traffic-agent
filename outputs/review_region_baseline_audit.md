# 复核区域定位基线审计

> **仅审计。** 运行当前未修改的代码；未调用 VLM，未改动任何历史报告或缓存。

- 复核目标总数：**11**
- 可裁剪：**2**　不可裁剪：**9**（81.8%）

## 真实代码位置

| 环节 | 位置 |
|---|---|
| `review_request_generation` | `src/multimodal/triggers.py:32 TriggerCollector (legacy: src/tables/review.py:57 collect_review_requests)` |
| `locate_label_bbox` | `src/multimodal/pipeline.py:35` |
| `crop_generation` | `src/multimodal/crop.py:106 resolve_review_crop` |
| `table_and_cell_bbox` | `src/tables/grid.py:142 recover_grid; src/tables/schemas.py TableCell.bbox` |
| `drawing_type_classification` | `src/vision/field_completeness.py:118 classify_drawing_type` |
| `field_completeness` | `src/vision/field_completeness.py:315 evaluate_field_completeness` |
| `vlm_executor` | `src/multimodal/executor.py:44 VisionReviewExecutor` |
| `second_evidence_policy` | `src/multimodal/decision.py:120 run_second_pass -> src/rag/policy.py RequiredEvidencePolicy` |
| `multimodal_report` | `scripts/multimodal_review_eval.py` |
| `multimodal_freeze` | `scripts/multimodal_freeze.py` |

## Q1 每个目标因何产生

`{'required_field_missing': 11}`

## Q3/Q4 失败发生在哪一级

| 失败阶段 | 数量 |
|---|---|
| `no_label_bbox_but_table_exists` | 9 |
| `none` | 2 |

## Q5 失败目标所在页面还剩什么可用结构

| document | 字段 | 页图 | 表bbox | 单元格bbox | 键值表 | 表格区域 | 邻近锚点 |
|---|---|---|---|---|---|---|---|
| FAN-A23-01 | 名称 | 有 | 有 | 有 | — | 有 | 有 |
| FAN-A23-01 | 控制柜编号 | 有 | 有 | 有 | — | 有 | 有 |
| FAN-A23-01 | 电机编号 | 有 | 有 | 有 | — | 有 | 有 |
| FAN-A24-01 | 名称 | 有 | 有 | 有 | 有 | 有 | 有 |
| FAN-A24-01 | 图号 | 有 | 有 | 有 | 有 | 有 | 有 |
| FAN-A24-01 | 控制柜编号 | 有 | 有 | 有 | 有 | 有 | 有 |
| FAN-A24-01 | 断路器编号 | 有 | 有 | 有 | 有 | 有 | 有 |
| FAN-A24-01 | 电机编号 | 有 | 有 | 有 | 有 | 有 | 有 |
| FAN-A24-01 | 页码|版本 | 有 | 有 | 有 | 有 | 有 | 有 |

## Q6 CELL_SPANS_COLUMNS 在表格阶段单独触发的请求数

`{'CELL_SPANS_COLUMNS': 57}`

## Q7 零请求页面

- **FAN-A22-01-heavyblur**：type=`unknown` known=False ocr_blocks=0 — drawing type unknown -> no required-field list -> nothing to declare missing -> no review request
- **FAN-MULTI-01**：type=`fan_group` known=True ocr_blocks=28 — evidence was complete
- **FAN-A13-02**：type=`fan_wiring` known=True ocr_blocks=16 — evidence was complete

## Q8 同页重复裁剪

1 对：`[{'a': '图号', 'b': '断路器编号', 'document_id': 'FAN-A23-01', 'note': 'same page, both cropped separately'}]`

## 逐文档

| case | 文档 | OCR块 | 类型 | 类型已知 | 必需字段表 | 网格 | 键值表 | 目标数 | 可裁剪 |
|---|---|---|---|---|---|---|---|---|---|
| MM1_watermark_silent_miss | FAN-A23-01 | 11 | fan_wiring | 是 | 有 | 6x2 | 否 | 5 | 2 |
| MM2_stamp_contaminated_value | FAN-A24-01 | 16 | fan_wiring | 是 | 有 | 4x2 | 是 | 6 | 0 |
| MM3_heavy_blur | FAN-A22-01-heavyblur | 0 | unknown | **否** | **无** | — | 否 | 0 | 0 |
| MM4_ocr_right_vlm_wrong | FAN-MULTI-01 | 28 | fan_group | 是 | 有 | 12x2 | 是 | 0 | 0 |
| MM5_clean_page | FAN-A13-02 | 16 | fan_wiring | 是 | 有 | 6x2 | 否 | 0 | 0 |