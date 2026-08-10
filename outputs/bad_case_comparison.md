# V2 到 V3 的 Bad Case 对比

> 数据来自 `data/challenge_cases.jsonl` 的本地回归。

| Case | V2 预测意图 | 正确意图 | 修复 | V3 |
|---|---|---|---|---|
| B01 | equipment_status, regulation_lookup | equipment_status | 否定范围：先移除“不要查规范”对应的候选意图。 | 通过 |
| B02 | regulation_lookup | general_qa | 引号语义：分类时将引号内容替换为占位符。 | 通过 |
| B03 | metric_query | metric_query | 时间冲突：同时出现相对与绝对时间时进入澄清。 | 通过 |
| B04 | fault_diagnosis, metric_query | metric_query | 否定下游意图：识别“不需要处理建议”。 | 通过 |
| B06 | metric_query | unsupported | 工具禁用：用户禁止必要数据源时返回 unsupported。 | 通过 |
| B08 | regulation_lookup, work_order_query | regulation_lookup | 关键词碰撞：“规范中‘工单’一词”整体按文档检索。 | 通过 |
| B10 | drawing_lookup, equipment_status | equipment_status | 显式排除：用户说“别查图纸”时不因图号触发工具。 | 通过 |

## 指标变化

- V2 对抗集端到端成功率：30.0%
- V3 对抗集端到端成功率：100.0%
- 注意：对抗集只有 10 条定向用例，只能用于回归验证，不能代表真实线上泛化。