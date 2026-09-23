# 复核区域分级定位报告

> **本轮不调用 VLM**（`real_vlm_calls=0`）。**无人工 bbox Gold**，
> 因此不报告定位准确率、裁剪 Recall、字段恢复率、VLM 准确率与泛化能力。

> 阈值来源：engineering starting values; NOT calibrated against human-reviewed bbox gold, which does not exist in this repo

## 两个必须分开看的比率

| 指标 | 分子 | 比率 |
|---|---|---|
| **精确局部裁剪可得率** | 2/11 | **18.2%** |
| 结构级 | 1/11 | 9.1% |
| 上下文级 | 8/11 | 72.7% |
| **任意复核区域可得率** | 11/11 | **100.0%** |
| 不可用 | 0/11 | 0.0% |

> precise_crop_available_rate counts located fields; any_region_available_rate includes contextual and diagnostic regions, which are a place to look, not an answer. A full-page diagnostic is NEVER counted as a local crop.

## 去重

- 去重前请求：**12**
- 去重后请求：**6**
- 合并掉：**6**

## 分布

- strategy：`{'full_page_diagnostic': 1, 'exact_label_right': 2, 'table_or_title_block': 2, 'alias_label_right': 1}`
- trust_level：`{'diagnostic': 1, 'precise': 2, 'contextual': 2, 'structural': 1}`
- 面积占比 p50=**18.6%** p95=**29.6%**

- 类型未知页：1，其中生成诊断请求：**1**
- 干净页零请求：**2**
- 仅由 CELL_SPANS_COLUMNS 产生的复核请求：**0**

## 基线九条 crop_unavailable 的新状态

| 文档 | 字段 | 基线失败原因 | 现策略 | 信任级 | 仍不可用 | 可自动回答 | 原因 |
|---|---|---|---|---|---|---|---|
| FAN-A23-01 | 名称 | `no_label_bbox_but_table_exists` | `table_or_title_block` | contextual | 否 | **否** | field could not be located; the table containing it wa |
| FAN-A23-01 | 控制柜编号 | `no_label_bbox_but_table_exists` | `table_or_title_block` | contextual | 否 | **否** | field could not be located; the table containing it wa |
| FAN-A23-01 | 电机编号 | `no_label_bbox_but_table_exists` | `table_or_title_block` | contextual | 否 | **否** | field could not be located; the table containing it wa |
| FAN-A24-01 | 名称 | `no_label_bbox_but_table_exists` | `table_or_title_block` | contextual | 否 | **否** | field is not in the configured title-block order |
| FAN-A24-01 | 图号 | `no_label_bbox_but_table_exists` | `table_or_title_block` | contextual | 否 | **否** | field is not in the configured title-block order |
| FAN-A24-01 | 控制柜编号 | `no_label_bbox_but_table_exists` | `table_or_title_block` | contextual | 否 | **否** | field is not in the configured title-block order |
| FAN-A24-01 | 断路器编号 | `no_label_bbox_but_table_exists` | `table_or_title_block` | contextual | 否 | **否** | field is not in the configured title-block order |
| FAN-A24-01 | 电机编号 | `no_label_bbox_but_table_exists` | `alias_label_right` | structural | 否 | **否** | alias match '电机编号' score=0.75 |
| FAN-A24-01 | 页码|版本 | `no_label_bbox_but_table_exists` | `table_or_title_block` | contextual | 否 | **否** | field is not in the configured title-block order |