# 复核区域定位评测

> 6 locatable reviewed record(s). Every rate below moves by 16.7% per record; these are counts, not performance estimates, and generalisation must not be inferred.

- 人工审核记录：**11**（可定位 6，不可定位 5）

## 总体

| 指标 | IoU@0.3 | IoU@0.5 | IoU@0.75 |
|---|---|---|---|
| candidate_oracle_recall | 33.3% | 33.3% | 0.0% |
| top1_localization_recall | 33.3% | 33.3% | 0.0% |
| precise_localization_recall | 33.3% | 33.3% | 0.0% |

- unlocatable 判定准确率：0.0
- 平均候选数：1
- 面积占比 p50/p95：0.2927 / 0.2927

> full-page diagnostic 不计入任何定位成功；unreviewed 与 excluded 不进入分母。