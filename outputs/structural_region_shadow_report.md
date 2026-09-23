# 结构定位 Shadow 候选报告

> **影子实验。** 全部候选 `promotion_status=shadow_only`，
> 不被 `resolve_review_regions` 或任何决策路径引用，**生产行为未改变**。
> **无人工 bbox Gold，因此本报告不说哪个候选是对的。**

- 阈值来源：`engineering_initial_value`　人工 Gold 校准：**False**
- 字段顺序来源：`dataset_or_project_convention`

## 汇总

| 指标 | 值 |
|---|---|
| 考察字段数 | 11 |
| 至少一个候选 | **6** |
| 无候选 | 5 |
| 候选总数 | 19 |
| 单字段候选数中位数 | 3.0 |
| 单字段最多候选 | 4 |
| 存在歧义（>1 候选）的字段 | **6** |
| top1−top2 中位数 | 0.0187 |
| top1−top2 最小值 | 0.0 |

- 候选策略分布：`{'single_sided_anchor': 6, 'field_order_only': 12, 'kv_row_candidate': 1}`
- 记录的假设：`{'bounded on one side only; the row could lie further in the unbounded direction': 6, 'no identified neighbour places this row': 10, 'placed by layout convention alone; no text and no anchor': 12, "field is not in this drawing type's order": 3}`
- 参与评分的证据项：`['cell_content', 'field_order', 'grid_consistency', 'neighbor_anchor', 'text_similarity']`

## 逐字段

| 文档 | 字段 | 候选数 | 歧义 | top1−top2 | top1 策略 | 无候选原因 |
|---|---|---|---|---|---|---|
| FAN-A23-01 | 名称 | 0 | 0 | — | `—` | page has no confirmed key/value table |
| FAN-A23-01 | 图号 | 0 | 0 | — | `—` | page has no confirmed key/value table |
| FAN-A23-01 | 控制柜编号 | 0 | 0 | — | `—` | page has no confirmed key/value table |
| FAN-A23-01 | 断路器编号 | 0 | 0 | — | `—` | page has no confirmed key/value table |
| FAN-A23-01 | 电机编号 | 0 | 0 | — | `—` | page has no confirmed key/value table |
| FAN-A24-01 | 名称 | 3 | 3 | 0.0 | `single_sided_anchor` | — |
| FAN-A24-01 | 图号 | 3 | 3 | 0.0183 | `single_sided_anchor` | — |
| FAN-A24-01 | 控制柜编号 | 3 | 3 | 0.125 | `single_sided_anchor` | — |
| FAN-A24-01 | 断路器编号 | 3 | 3 | 0.0192 | `single_sided_anchor` | — |
| FAN-A24-01 | 电机编号 | 4 | 4 | 0.2667 | `kv_row_candidate` | — |
| FAN-A24-01 | 页码|版本 | 3 | 3 | 0.0 | `field_order_only` | — |

> 候选覆盖了表格全部行**不等于 Recall 100%**；
> 在人工 Gold 到位之前，覆盖率只说明搜索空间大小，不说明命中。