# OCR Pipeline A/B/C/D/E 真实实验报告

> 样本数：11，耗时：41.3s。
> A/B 使用 MockOCREngine（无真实像素感知能力，见 data/ocr_stub/README.md），数字不代表真实 OCR 准确率。
> C/D 是真实 Qwen-VL 调用；D 额外套了真实 cv2 预处理，是本轮唯一完整意义上的真实 A/B 对照。
> E 的 OCR/表格结构部分是 Mock，路由决策、VLM Fallback 调用和 Validator 校验部分是真实的。

## 实验 A：Original -> OCR(mock) -> TableParser(real)
- 整体平均得分：64.5%（n=11）
- dev 平均得分：71.4%（n=7）
- holdout 平均得分：52.5%（n=4）
  - normal: 67.5%（n=2）
  - complex_table: 100.0%（n=3）
  - stamp: 50.0%（n=2）
  - watermark: 75.0%（n=1）
  - skew: 100.0%（n=1）
  - blur: 0.0%（n=2）

## 实验 B：Preprocess(real) -> OCR(mock) -> TableParser(real)
- 整体平均得分：64.5%（n=11）
- dev 平均得分：71.4%（n=7）
- holdout 平均得分：52.5%（n=4）
  - normal: 67.5%（n=2）
  - complex_table: 100.0%（n=3）
  - stamp: 50.0%（n=2）
  - watermark: 75.0%（n=1）
  - skew: 100.0%（n=1）
  - blur: 0.0%（n=2）

## 实验 C：Original -> VLM(real) -> JSON
- 整体平均得分：77.7%（n=11）
- dev 平均得分：75.0%（n=7）
- holdout 平均得分：82.5%（n=4）
  - normal: 90.0%（n=2）
  - complex_table: 100.0%（n=3）
  - stamp: 75.0%（n=2）
  - watermark: 75.0%（n=1）
  - skew: 75.0%（n=1）
  - blur: 37.5%（n=2）

## 实验 D：Preprocess(real) -> VLM(real) -> JSON
- 整体平均得分：70.9%（n=11）
- dev 平均得分：75.0%（n=7）
- holdout 平均得分：63.7%（n=4）
  - normal: 90.0%（n=2）
  - complex_table: 100.0%（n=3）
  - stamp: 75.0%（n=2）
  - watermark: 75.0%（n=1）
  - skew: 75.0%（n=1）
  - blur: 0.0%（n=2）

## 实验 E：OCR(mock)+TableParser(real) -> QualityJudge routing(real) -> VLM Fallback(real when triggered) -> Validator(real)
- 整体平均得分：77.7%（n=11）
- dev 平均得分：75.0%（n=7）
- holdout 平均得分：82.5%（n=4）
  - normal: 90.0%（n=2）
  - complex_table: 100.0%（n=3）
  - stamp: 75.0%（n=2）
  - watermark: 75.0%（n=1）
  - skew: 75.0%（n=1）
  - blur: 37.5%（n=2）
