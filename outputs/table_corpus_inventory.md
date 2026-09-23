# 表格语料清单

共 46 张图片。**来源按磁盘上的证据判定，不按文件名或目录归类。**

## 来源分布

| 分类 | 数量 | 含义 |
|---|---|---|
| `real_image_fixture_replay` | 11 | 有版本戳的真实 PaddleOCR 运行结果重放（非实时推理） |
| `synthetic_image` | 21 | 本仓库自撰生成的图片（上一轮对抗集） |
| `unknown_provenance` | 14 | 无任何 OCR 证据，**不归为真实** |

## 真实表格统计

- 成功恢复网格的真实表格：**8**
- 多级表头的真实表格：**0**
- 含合并单元格的真实表格：**0**
- 跨页的真实表格：**0**（每个源文件都是单页）
- 人工审核过的结构 Gold：**0**

> 因此本轮**多级表头、合并单元格、跨页续表只能由合成测试固定行为**，
> 不得报告任何基于真实数据的相关准确率。

## 缺失数据清单（待人工补充）

- no real table with a multi-level header
- no real table with a merged cell
- no real cross-page table (every source file is a single page)
- no human-reviewed structure gold

## 逐张明细

| document_id | 来源 | ocr_source | 有网格 | 形状 | 横线 | 竖线 | 水印 | 低对比度 |
|---|---|---|---|---|---|---|---|---|
| FAN-A13-02 | real_image_fixture_replay | paddleocr_fixture_replay | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A14-01 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A15-01 | unknown_provenance | none | 是 | 5x2 | 8 | 2 | 否 | 否 |
| FAN-A16-02 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A17-01-lowres | unknown_provenance | none | 是 | 6x2 | 7 | 2 | 否 | 是 |
| FAN-A17-01 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A18-01 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A19-01 | real_image_fixture_replay | paddleocr_fixture_replay | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A20-01 | unknown_provenance | none | 否 | — | 4 | 2 | 否 | 是 |
| FAN-A21-01 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A21-02 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A22-01-heavyblur | real_image_fixture_replay | paddleocr_fixture_replay | 否 | — | 4 | 2 | 否 | 是 |
| FAN-A23-01 | real_image_fixture_replay | paddleocr_fixture_replay | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A24-01 | real_image_fixture_replay | paddleocr_fixture_replay | 是 | 4x2 | 7 | 2 | 否 | 是 |
| FAN-A25-01 | real_image_fixture_replay | paddleocr_fixture_replay | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A26-01 | real_image_fixture_replay | paddleocr_fixture_replay | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-A27-01-skewed | real_image_fixture_replay | paddleocr_fixture_replay | 否 | — | 0 | 0 | 否 | 否 |
| FAN-A28-01-borderless | real_image_fixture_replay | paddleocr_fixture_replay | 否 | — | 2 | 2 | 否 | 否 |
| FAN-MULTI-01 | real_image_fixture_replay | paddleocr_fixture_replay | 是 | 12x2 | 15 | 2 | 否 | 否 |
| FAN-RELATION-01 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| FAN-RELATION-02 | unknown_provenance | none | 是 | 2x2 | 5 | 2 | 否 | 否 |
| PSD-B08-01 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| PSD-MULTI-01 | unknown_provenance | none | 是 | 11x2 | 14 | 4 | 否 | 否 |
| PUMP-02-01 | unknown_provenance | none | 是 | 6x2 | 9 | 2 | 否 | 否 |
| PUMP-MULTI-02 | real_image_fixture_replay | paddleocr_fixture_replay | 是 | 11x2 | 14 | 2 | 否 | 否 |
| ADV01 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV02 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV03 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV04 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV05 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV06 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV07 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV08 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV09 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV10 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV11 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV12 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV13 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV14 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV15 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV16 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 否 |
| ADV17 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 是 |
| ADV18 | synthetic_image | none | 否 | — | 2 | 2 | 否 | 是 |
| ADV19 | synthetic_image | none | 否 | — | 0 | 0 | 否 | 是 |
| ADV20 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 是 |
| ADV21 | synthetic_image | none | 否 | — | 3 | 2 | 否 | 是 |