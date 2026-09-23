# 复核区域定位评测

## 状态：`not_evaluated`

- 原因：`no_human_reviewed_bbox_gold`
- 人工审核记录数：**0**
- Gold 状态：`{'unreviewed': 11, 'human_reviewed': 0, 'excluded': 0}`

> No metric is emitted. A 0% or 100% here would be read as a measurement; there is nothing to measure against. The system's own output is not an answer key.

## 下一步

annotate data/review_region_gold_unreviewed.jsonl using docs/review_region_annotation_guide.md, then run scripts/validate_review_region_gold.py

## 人工审核后才能计算的指标

- `IoU`
- `IoU@0.3`
- `IoU@0.5`
- `IoU@0.75`
- `field_localization_recall`
- `precise_localization_recall`
- `candidate_oracle_recall`
- `top1_localization_recall`
- `unlocatable_detection_accuracy`
- `average_candidate_count`
- `area_ratio_p50`
- `area_ratio_p95`