# OCR Pipeline A/B/C/D/E 实验报告

> 样本数：11，耗时：19.0s，OCR 引擎：`fixture`。
> A/B/E 的 OCR 来自 FixtureOCREngine——重放 data/ocr_fixtures/ 中**已保存的真实 PaddleOCR 运行结果**（每份带图像 sha256、库/模型版本、设备、生成时间）。内容是真实的，但**不是本轮实时推理**。
> C/D 是真实 Qwen-VL 调用；D 额外套了真实 cv2 预处理。
> E 的路由决策、VLM Fallback 调用和 Validator 校验部分始终是真实的。

## 实验 A：Original -> OCR(fixture) -> TableParser(real)
- 整体平均得分：79.5%（n=11）
- 开发集平均得分：71.4%（n=7）
- 回归验证集平均得分：93.8%（n=4）
  - stamp: 87.5%（n=2）
  - watermark: 25.0%（n=1）
  - normal: 100.0%（n=2）
  - complex_table: 100.0%（n=3）
  - skew: 100.0%（n=1）
  - blur: 37.5%（n=2）

## 实验 B：Preprocess(real) -> OCR(fixture) -> TableParser(real)
- 整体平均得分：72.7%（n=11）
- 开发集平均得分：71.4%（n=7）
- 回归验证集平均得分：75.0%（n=4）
  - stamp: 87.5%（n=2）
  - watermark: 25.0%（n=1）
  - normal: 100.0%（n=2）
  - complex_table: 100.0%（n=3）
  - skew: 100.0%（n=1）
  - blur: 0.0%（n=2）

## 实验 C：Original -> VLM(real) -> JSON
- 整体平均得分：77.7%（n=11）
- 开发集平均得分：75.0%（n=7）
- 回归验证集平均得分：82.5%（n=4）
  - stamp: 75.0%（n=2）
  - watermark: 75.0%（n=1）
  - normal: 90.0%（n=2）
  - complex_table: 100.0%（n=3）
  - skew: 75.0%（n=1）
  - blur: 37.5%（n=2）

## 实验 D：Preprocess(real) -> VLM(real) -> JSON
- 整体平均得分：70.9%（n=11）
- 开发集平均得分：75.0%（n=7）
- 回归验证集平均得分：63.7%（n=4）
  - stamp: 75.0%（n=2）
  - watermark: 75.0%（n=1）
  - normal: 90.0%（n=2）
  - complex_table: 100.0%（n=3）
  - skew: 75.0%（n=1）
  - blur: 0.0%（n=2）

## 实验 E：OCR(fixture)+TableParser(real) -> QualityJudge routing(real) -> VLM Fallback(real when triggered) -> Validator(real)
- 整体平均得分：84.1%（n=11）
- 开发集平均得分：78.6%（n=7）
- 回归验证集平均得分：93.8%（n=4）
  - stamp: 87.5%（n=2）
  - watermark: 75.0%（n=1）
  - normal: 100.0%（n=2）
  - complex_table: 100.0%（n=3）
  - skew: 100.0%（n=1）
  - blur: 37.5%（n=2）

### 路由指标（实验 E）

> `fallback_proxy_positive` 是**派生代理标签**，不是金标：它由「纯 OCR 路径未取全任务所需字段」推导而来，和最终字段命中率**共用同一份答案**，因此召回率与命中率不独立，不能互相印证。真正的标签需要人在看到任何分数之前判断该页是否需要兜底。

> 四个命中口径由松到紧：`legacy_containment_hit_rate`（gold 串出现在两源拼接文本中，最乐观，既不分来源也不管矛盾）→ `structured_candidate_recall`（存在结构化候选包含该值）→ `conflict_free_exact_rate`（存在无冲突候选精确等于该值）→ **`decision_ready_exact_rate`**（在此之上还要求实体归属已确定、且未违反字段值规则——即系统可在无人复核下直接作答）。对外引用请用最后一个，`legacy` 仅用于与历史报告对比。

| 指标 | 开发集(7) | 回归验证集(4) | 合计(11) |
|---|---|---|---|
| 兜底触发率 | 57.1% | 50.0% | 54.5% |
| 代理兜底召回率 | 100.0% | 100.0% | 100.0% |
| 代理误触发率 | 25.0% | 33.3% | 28.6% |
| VLM 字段恢复率 | 25.0% | 0.0% | 22.2% |
| 纯 OCR 字段命中率（字符串包含） | 71.4% | 93.8% | 79.5% |
| legacy_containment_hit_rate（乐观口径） | 78.6% | 93.8% | 84.1% |
| structured_candidate_recall | 78.6% | 87.5% | 81.8% |
| conflict_free_exact_rate（保守口径） | 78.6% | 87.5% | 81.8% |
| decision_ready_exact_rate（可直接作答口径） | 71.4% | 81.2% | 75.0% |
| 字段冲突数 | 0 | 0 | 0 |
| 非法字段值数 | 1 | 0 | 1 |
| 设备归属不确定数 | 5 | 2 | 7 |
| 实体已对齐数 | 5 | 3 | 8 |
| 实体单边存在数(无对象可配) | 13 | 7 | 20 |
| 实体对齐 unresolved 数(两侧有候选但配不上) | 0 | 0 | 0 |

实体对齐状态分布：`aligned_exact_identifier` × 6；`aligned_singleton` × 2；`single_source` × 20

单边存在(single_source)原因分布：`ocr_side_absent` × 4；`vlm_not_invoked` × 15；`vlm_side_absent` × 1

实体映射方式分布：`exact->breaker` × 2；`exact->fan` × 5；`exact->motor` × 2；`field_rule->breaker` × 6；`field_rule->cabinet` × 6；`field_rule->fan` × 3；`field_rule->motor` × 6；`field_rule->pump` × 3；`regex->pump` × 3

触发原因分布（合计）：`drawing_type_unknown` × 1；`field_completeness_below_threshold` × 3；`group_cardinality_uncertain` × 2；`invalid_field_value` × 1；`isolated_labels_without_values` × 1；`ocr_confidence_below_threshold` × 2；`repeated_boilerplate_text` × 1
