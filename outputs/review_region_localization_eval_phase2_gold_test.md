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

---

# Phase 1 补充（evaluator_version 2）

> **IoU 衡量框是否紧；Coverage 衡量是否把目标包含进去。** 两者同时报告，Coverage 不替代 IoU。只看 Coverage 时，整页裁剪能拿满分。

## Gold Coverage

| 口径 | Coverage@0.5 | Coverage@0.8 | Coverage@0.95 |
|---|---|---|---|
| top1 | 5/6 | 5/6 | 5/6 |
| candidate oracle | 5/6 | 5/6 | 5/6 |

## 按定位等级拆分（top1）

| 等级 | n | Cov@0.8 | Cov@0.95 | IoU@0.5 | IoU@0.75 | 平均 precision |
|---|---|---|---|---|---|---|
| precise | 2 | 2/2 | 2/2 | 2/2 | 0/2 | 0.6451 |
| structural | 0 | — | — | — | — | — |
| contextual | 4 | 3/4 | 3/4 | 0/4 | 0/4 | 0.079 |
| diagnostic | 0 | — | — | — | — | — |
| unavailable | 0 | — | — | — | — | — |

> contextual 区域即使 Coverage=1 也**不计入** precise localization；full-page diagnostic 不计入局部定位成功。

## 不可定位字段：安全与成本分开

| 指标 | 类型 | 计数 | 含义 |
|---|---|---|---|
| unlocatable_auto_answer_false_positive_rate | **安全** | **0/5** | a human could not locate the field, yet the region the system selected was answer_eligible |
| unlocatable_review_trigger_rate | 成本 | **5/5** | a human could not locate the field, yet a review region was produced for it |

- 选中区域等级分布：`{'precise': 0, 'structural': 1, 'contextual': 4, 'diagnostic': 0, 'unavailable': 0}`
- answer_eligible measures the REGION GATE only. It is not proof that an answer was emitted: value validation and the evidence policy still run after it.

> 旧指标 `unlocatable_detection_accuracy`（0.0）保留为 **legacy**：它把安全失败和成本浪费混成一个数，0/5 看起来像五次安全错误。

## 逐条

| record | 可定位 | strategy | trust | IoU | Coverage | Precision | answer_eligible | 结论 |
|---|---|---|---|---|---|---|---|---|
| FAN-A23-01:p1:名称 | 是 | `table_or_title_block` | contextual | 0.105 | 1.00 | 0.11 | 否 | **context_only** |
| FAN-A23-01:p1:图号 | 是 | `exact_label_right` | precise | 0.652 | 1.00 | 0.65 | 是 | **covered_but_loose** |
| FAN-A23-01:p1:控制柜编号 | 是 | `table_or_title_block` | contextual | 0.105 | 1.00 | 0.11 | 否 | **context_only** |
| FAN-A23-01:p1:断路器编号 | 是 | `exact_label_right` | precise | 0.638 | 1.00 | 0.64 | 是 | **covered_but_loose** |
| FAN-A23-01:p1:电机编号 | 是 | `table_or_title_block` | contextual | 0.105 | 1.00 | 0.11 | 否 | **context_only** |
| FAN-A24-01:p1:名称 | 是 | `table_or_title_block` | contextual | 0.000 | 0.00 | 0.00 | 否 | **missed** |
| FAN-A24-01:p1:图号 | 否 | `table_or_title_block` | contextual | — | — | — | 否 | **unnecessary_review** |
| FAN-A24-01:p1:控制柜编号 | 否 | `table_or_title_block` | contextual | — | — | — | 否 | **unnecessary_review** |
| FAN-A24-01:p1:断路器编号 | 否 | `table_or_title_block` | contextual | — | — | — | 否 | **unnecessary_review** |
| FAN-A24-01:p1:电机编号 | 否 | `alias_label_right` | structural | — | — | — | 否 | **unnecessary_review** |
| FAN-A24-01:p1:页码|版本 | 否 | `table_or_title_block` | contextual | — | — | — | 否 | **unnecessary_review** |

结论计数：`{'precise_match': 0, 'covered_but_loose': 2, 'context_only': 3, 'missed': 1, 'correctly_abstained': 0, 'unnecessary_review': 5, 'unsafe_auto_answer': 0}`

---

# Phase 2 字段归属（evaluator_version 3）

> 只有 `association_status=confirmed` 的记录进入正式定位评测。v1 记录一律为 `unknown`：v1 从未记录值是否属于目标字段。

| 归属状态 | 数量 |
|---|---|
| confirmed | 0 |
| ambiguous | 0 |
| unassigned | 0 |
| unknown | 11 |

- Schema 版本：`{'region_gold/1': 11}`
- 正式定位评测：**`not_evaluated`**（`no_confirmed_attribution_gold`）
- ambiguous 自动回答误放行：**—**
- unassigned 自动回答误放行：**—**
- unknown：11（来自 v1：11）

## 归属感知的决策分布

| 归属状态 | 可自动回答 | 仅复核 | 无区域 |
|---|---|---|---|
| confirmed | 0 | 0 | 0 |
| ambiguous | 0 | 0 | 0 |
| unassigned | 0 | 0 | 0 |
| unknown | 2 | 9 | 0 |

## 口径变更

- 仍然有效：legacy IoU metrics and phase-1 coverage as statements about BOXES vs gold_region_type boxes；phase-1 unlocatable safety/cost split (region gate behaviour)
- 被取代：reading legacy/phase-1 localisation recall as FIELD localisation: it is only that once records are association_status=confirmed