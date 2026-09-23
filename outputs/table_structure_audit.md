# 表格结构恢复审计

> 真实图纸 + 录制的 PaddleOCR fixture 重放。**未调用 VLM，未实时 OCR，无人工审核 Gold。**

## 真实结果

- 检查的真实文档：**11**
- 成功恢复网格的真实表格：**8**
- 真实多级表头表格：**0**　真实合并单元格表格：**0**　真实跨页表格：**0**

> 因此**多级表头展开、合并单元格识别、跨页续表判定只有合成测试**，不得报告基于真实数据的相关准确率。

- 结构恢复准确率：**not_computed — no human-reviewed gold; a number derived from the parser's own output would be circular**

## Chunk 与溯源

chunk 类型分布：`{'table_parent': 8, 'table_row': 48}`

| 指标 | 分子/分母 |
|---|---|
| 页码可溯源 | 56/56 = 100.0% |
| bbox 可溯源 | 56/56 = 100.0% |
| 行 chunk 带实体 | 0/48 = 0.0% |
| **跨设备污染** | 0/48 = 0.0% |

裁剪产出：`{'table': 8, 'row': 24, 'cell': 8}`

## 局部 VLM 复核

- 生成请求 **60** 条
- 状态分布：`{'not_required': 0, 'requested_not_invoked': 60, 'cache_replay': 0, 'real_inference': 0}`
- 原因分布：`{'CELL_SPANS_COLUMNS': 57, 'HEADER_ATTRIBUTION_UNCERTAIN': 3}`

> 全部为 `requested_not_invoked`。**本轮没有调用任何视觉模型**，把它写成复核成功会把一个未决问题变成伪造的确认。

## 逐文档

| document | 网格 | 单元格 | 空 | 歧义 | 表外 block | 实体列 | chunk | 复核请求 |
|---|---|---|---|---|---|---|---|---|
| FAN-A13-02 | 6x2 | 10 | 2 | 7 | 4 | None | 6 | 7 |
| FAN-A19-01 | 6x2 | 10 | 2 | 7 | 3 | None | 6 | 7 |
| FAN-A22-01-heavyblur | — | — | — | — | — | — | — | no internal grid found |
| FAN-A23-01 | 6x2 | 6 | 6 | 6 | 4 | None | 5 | 6 |
| FAN-A24-01 | 4x2 | 8 | 0 | 7 | 4 | None | 5 | 6 |
| FAN-A25-01 | 6x2 | 10 | 2 | 4 | 4 | None | 6 | 4 |
| FAN-A26-01 | 6x2 | 10 | 2 | 7 | 3 | None | 6 | 7 |
| FAN-A27-01-skewed | — | — | — | — | — | — | — | no internal grid found |
| FAN-A28-01-borderless | — | — | — | — | — | — | — | no internal grid found |
| FAN-MULTI-01 | 12x2 | 24 | 0 | 6 | 4 | None | 13 | 6 |
| PUMP-MULTI-02 | 11x2 | 19 | 3 | 15 | 4 | None | 9 | 17 |