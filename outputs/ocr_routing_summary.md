# OCR Pipeline A/B/C/D/E 实验报告

> 样本数：11，耗时：48.7s，OCR 引擎：`fixture`。
> A/B/E 的 OCR 来自 FixtureOCREngine——重放 data/ocr_fixtures/ 中**已保存的真实 PaddleOCR 运行结果**（每份带图像 sha256、库/模型版本、设备、生成时间）。内容是真实的，但**不是本轮实时推理**。
> C/D 是真实 Qwen-VL 调用；D 额外套了真实 cv2 预处理。
> E 的路由决策、VLM Fallback 调用和 Validator 校验部分始终是真实的。

## 实验 A：Original -> OCR(fixture) -> TableParser(real)
- 整体平均得分：79.5%（n=11）
- 开发集平均得分：71.4%（n=7）
- 回归验证集平均得分：93.8%（n=4）
  - stamp: 87.5%（n=2）
  - normal: 100.0%（n=2）
  - watermark: 25.0%（n=1）
  - complex_table: 100.0%（n=3）
  - skew: 100.0%（n=1）
  - blur: 37.5%（n=2）

## 实验 B：Preprocess(real) -> OCR(fixture) -> TableParser(real)
- 整体平均得分：72.7%（n=11）
- 开发集平均得分：71.4%（n=7）
- 回归验证集平均得分：75.0%（n=4）
  - stamp: 87.5%（n=2）
  - normal: 100.0%（n=2）
  - watermark: 25.0%（n=1）
  - complex_table: 100.0%（n=3）
  - skew: 100.0%（n=1）
  - blur: 0.0%（n=2）

## 实验 C：Original -> VLM(real) -> JSON
- 整体平均得分：77.7%（n=11）
- 开发集平均得分：75.0%（n=7）
- 回归验证集平均得分：82.5%（n=4）
  - stamp: 75.0%（n=2）
  - normal: 90.0%（n=2）
  - watermark: 75.0%（n=1）
  - complex_table: 100.0%（n=3）
  - skew: 75.0%（n=1）
  - blur: 37.5%（n=2）

## 实验 D：Preprocess(real) -> VLM(real) -> JSON
- 整体平均得分：70.9%（n=11）
- 开发集平均得分：75.0%（n=7）
- 回归验证集平均得分：63.7%（n=4）
  - stamp: 75.0%（n=2）
  - normal: 90.0%（n=2）
  - watermark: 75.0%（n=1）
  - complex_table: 100.0%（n=3）
  - skew: 75.0%（n=1）
  - blur: 0.0%（n=2）

## 实验 E：OCR(fixture)+TableParser(real) -> QualityJudge routing(real) -> VLM Fallback(real when triggered) -> Validator(real)
- 整体平均得分：84.1%（n=11）
- 开发集平均得分：78.6%（n=7）
- 回归验证集平均得分：93.8%（n=4）
  - stamp: 87.5%（n=2）
  - normal: 100.0%（n=2）
  - watermark: 75.0%（n=1）
  - complex_table: 100.0%（n=3）
  - skew: 100.0%（n=1）
  - blur: 37.5%（n=2）

### 路由指标（实验 E）

> `fallback_proxy_positive` 是**派生代理标签**，不是金标：它由「纯 OCR 路径未取全任务所需字段」推导而来，和最终字段命中率**共用同一份答案**，因此召回率与命中率不独立，不能互相印证。真正的标签需要人在看到任何分数之前判断该页是否需要兜底。

| 指标 | 开发集(7) | 回归验证集(4) | 合计(11) |
|---|---|---|---|
| 兜底触发率 | 42.9% | 50.0% | 45.5% |
| 代理兜底召回率 | 66.7% | 100.0% | 75.0% |
| 代理误触发率 | 25.0% | 33.3% | 28.6% |
| VLM 字段恢复率 | 28.6% | 0.0% | 25.0% |
| 纯 OCR 字段命中率 | 71.4% | 93.8% | 79.5% |
| OCR+VLM 最终字段命中率 | 78.6% | 93.8% | 84.1% |

触发原因分布（合计）：`drawing_type_unknown` × 1；`field_completeness_below_threshold` × 3；`group_cardinality_uncertain` × 2；`isolated_labels_without_values` × 1；`ocr_confidence_below_threshold` × 2；`repeated_boilerplate_text` × 1
