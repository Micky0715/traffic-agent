# 多模态复核报告

- 推理模式：**cache_replay**
- 实时调用次数：**0** / 上限 5
- 缓存出处：`{'entries': 2, 'recorded_from_live_inference': 2, 'models': ['qwen-vl-max'], 'prompt_versions': ['vision_field_review_v1']}`

> cache_replay 的耗时是文件读取，**不是模型延迟**，本报告不以模型延迟名义报告它。

## 汇总

| 指标 | 值 |
|---|---|
| budget_exhausted | 0 |
| cache_replay | 2 |
| call_failed | 0 |
| crop_unavailable | 9 |
| disabled | 0 |
| not_invoked | 0 |
| real_inference | 0 |
| schema_failure | 0 |
| success | 2 |
| total_review_requests | 11 |
| unreadable | 0 |

## 逐案例

| case | 文档 | OCR置信 | 完整度 | 触发复核 | 裁剪 | 状态 | 恢复字段 | 冲突 |
|---|---|---|---|---|---|---|---|---|
| MM1_watermark_silent_miss | FAN-A23-01 | 0.9957 | 0.1667 | 是 | label_right,unavailable | crop_unavailable,success,crop_unavailable,success,crop_unavailable | 图号,断路器编号 | 0 |
| MM2_stamp_contaminated_value | FAN-A24-01 | 0.6141 | 0.0 | 是 | unavailable | crop_unavailable,crop_unavailable,crop_unavailable,crop_unavailable,crop_unavailable,crop_unavailable | 无 | 0 |
| MM3_heavy_blur | FAN-A22-01-heavyblur | 0.0 | 0.0 | 否 | — | — | 无 | 0 |
| MM4_ocr_right_vlm_wrong | FAN-MULTI-01 | 0.9998 | 1.0 | 否 | — | — | 无 | 1 |
| MM5_clean_page | FAN-A13-02 | 0.9976 | 1.0 | 否 | — | — | 无 | 0 |

## 预注册期望校验

通过 11 / 11

| case | 检查 | 期望 | 实测 | 结果 |
|---|---|---|---|---|
| MM1_watermark_silent_miss | review_triggered | `True` | `True` | PASS |
| MM1_watermark_silent_miss | no_whole_page_fallback | `no page-scope crop` | `['label_right', 'unavailable']` | PASS |
| MM2_stamp_contaminated_value | review_triggered | `True` | `True` | PASS |
| MM2_stamp_contaminated_value | vlm_not_auto_adopted | `all unresolved` | `0 resolved` | PASS |
| MM3_heavy_blur | no_whole_page_fallback | `no page-scope crop` | `['(no crop)']` | PASS |
| MM3_heavy_blur | degraded_page_does_not_guess | `['crop_unavailable', 'human_review', 'abstain']` | `['(no review)']` | PASS |
| MM4_ocr_right_vlm_wrong | conflict_recorded | `True` | `True` | PASS |
| MM4_ocr_right_vlm_wrong | conflict_blocks_execute | `not execute` | `human_review` | PASS |
| MM4_ocr_right_vlm_wrong | vlm_not_auto_adopted | `all unresolved` | `0 resolved` | PASS |
| MM5_clean_page | review_triggered | `False` | `False` | PASS |
| MM5_clean_page | no_vlm_call | `0` | `0` | PASS |