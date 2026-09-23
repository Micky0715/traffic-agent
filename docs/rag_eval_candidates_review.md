# RAG 评测候选集 · 待人工审核

共 42 条，全部 `label_status=unreviewed`。

> **这些不是 Gold。** 每一条的 `candidate_expected_decision` 与 `candidate_answer`
> 都是机器提出的**建议**，由脚本按语料结构生成，**不是运行系统得到的输出**。
> 人工确认之前，不得用它计算任何正式 Recall@5、拒答率或准确率指标。

审核方式：逐条确认或修改 `candidate_expected_decision`，
然后把 `label_status` 改成 `reviewed`。**只有改过的行才算 Gold。**

文件：`data/rag_eval_candidates_unreviewed.jsonl`

## 分布

| 生成方式 | 条数 | 关注点 |
|---|---|---|
| `absent_device` | 8 | 设备不存在，应拒答而非用邻近设备顶替 |
| `exact_field_from_corpus` | 6 | 语料确有该字段，应可作答 |
| `field_absent_for_present_device` | 7 | 设备在但字段缺，应拒答 |
| `irrelevant_question` | 4 | 完全无关，refusal 形式待定 |
| `multi_device_interference` | 6 | 同表兄弟设备不得串入答案 |
| `one_present_one_absent` | 5 | 应 partial，不能因缺一个否决另一个 |
| `synonym_surface_form` | 6 | 同义表达能否命中同一字段 |

## absent_device

| case_id | query | 建议决策 | 建议答案 | 审核备注 |
|---|---|---|---|---|
| RAGC-012 | 查询A99风机的电机编号 | abstain | — | confirm this device really is absent |
| RAGC-013 | 查询B77屏蔽门的电机编号 | abstain | — | confirm this device really is absent |
| RAGC-014 | 查询9号水泵的电机编号 | abstain | — | confirm this device really is absent |
| RAGC-015 | 查询C41风机的电机编号 | abstain | — | confirm this device really is absent |
| RAGC-016 | 查询TR-05变压器的电机编号 | abstain | — | confirm this device really is absent |
| RAGC-017 | 查询ESC-12扶梯的电机编号 | abstain | — | confirm this device really is absent |
| RAGC-018 | 查询A00风机的电机编号 | abstain | — | confirm this device really is absent |
| RAGC-019 | 查询PUMP-99的电机编号 | abstain | — | confirm this device really is absent |

## exact_field_from_corpus

| case_id | query | 建议决策 | 建议答案 | 审核备注 |
|---|---|---|---|---|
| RAGC-000 | 查询A13风机接线图的电机编号 | execute | M-13 |  |
| RAGC-001 | 查询A19风机接线图的电机编号 | execute | M-19 |  |
| RAGC-002 | 查询A25风机接线图的电机编号 | execute | M-25 |  |
| RAGC-003 | 查询A26风机接线图的电机编号 | execute | M-26 |  |
| RAGC-004 | 查询A27风机接线图的电机编号 | execute | M-27 |  |
| RAGC-005 | 查询A28风机参数表的电机编号 | execute | M-28 |  |

## field_absent_for_present_device

| case_id | query | 建议决策 | 建议答案 | 审核备注 |
|---|---|---|---|---|
| RAGC-020 | 查询A13风机接线图的版本 | abstain | — |  |
| RAGC-021 | 查询A19风机接线图的版本 | abstain | — |  |
| RAGC-022 | 查询FAN-A23-01的版本 | abstain | — |  |
| RAGC-023 | 查询A25风机接线图的版本 | abstain | — |  |
| RAGC-024 | 查询A26风机接线图的页码 | abstain | — |  |
| RAGC-025 | 查询A27风机接线图的版本 | abstain | — |  |
| RAGC-026 | 查询A28风机参数表的页码 | abstain | — |  |

## irrelevant_question

| case_id | query | 建议决策 | 建议答案 | 审核备注 |
|---|---|---|---|---|
| RAGC-038 | 今天天气怎么样 | clarify | — | decide whether clarify or abstain is the right refusal here |
| RAGC-039 | 帮我写一首诗 | clarify | — | decide whether clarify or abstain is the right refusal here |
| RAGC-040 | 解释一下量子纠缠 | clarify | — | decide whether clarify or abstain is the right refusal here |
| RAGC-041 | 附近有什么餐厅 | clarify | — | decide whether clarify or abstain is the right refusal here |

## multi_device_interference

| case_id | query | 建议决策 | 建议答案 | 审核备注 |
|---|---|---|---|---|
| RAGC-032 | 查询A16的功率 | execute | 45kW | sibling devices must not leak into the answer |
| RAGC-033 | 查询A17的功率 | execute | 55kW | sibling devices must not leak into the answer |
| RAGC-034 | 查询A18的功率 | execute | 37kW | sibling devices must not leak into the answer |
| RAGC-035 | 查询20的流量 | execute | 150m³/h | sibling devices must not leak into the answer |
| RAGC-036 | 查询2A的流量 | execute | 80m³/h | sibling devices must not leak into the answer |
| RAGC-037 | 查询2B的流量 | execute | 80m³/h | sibling devices must not leak into the answer |

## one_present_one_absent

| case_id | query | 建议决策 | 建议答案 | 审核备注 |
|---|---|---|---|---|
| RAGC-027 | 查询A13风机接线图的控制柜编号和版本 | partial | FAN-CAB-3 |  |
| RAGC-028 | 查询A19风机接线图的控制柜编号和版本 | partial | 审核专用毫AB9 |  |
| RAGC-029 | 查询A25风机接线图的控制柜编号和版本 | partial | FAN-CAB-25 |  |
| RAGC-030 | 查询A26风机接线图的控制柜编号和页码 | partial | FAN-CAB-26 |  |
| RAGC-031 | 查询A27风机接线图的控制柜编号和版本 | partial | FAN-CAB-27 |  |

## synonym_surface_form

| case_id | query | 建议决策 | 建议答案 | 审核备注 |
|---|---|---|---|---|
| RAGC-006 | A13风机接线图的控制柜号是多少 | execute | FAN-CAB-3 | confirm the synonym belongs in configs/rag.yaml |
| RAGC-007 | A19风机接线图的控制柜号是多少 | execute | 审核专用毫AB9 | confirm the synonym belongs in configs/rag.yaml |
| RAGC-008 | A25风机接线图的控制柜号是多少 | execute | FAN-CAB-25 | confirm the synonym belongs in configs/rag.yaml |
| RAGC-009 | A26风机接线图的控制柜号是多少 | execute | FAN-CAB-26 | confirm the synonym belongs in configs/rag.yaml |
| RAGC-010 | A27风机接线图的控制柜号是多少 | execute | FAN-CAB-27 | confirm the synonym belongs in configs/rag.yaml |
| RAGC-011 | A28风机参数表的控制柜号是多少 | execute | FAN-CAB-28 | confirm the synonym belongs in configs/rag.yaml |