# 夜间垂直链路交付报告

> 一次性运行。零 API 调用，未连接 ES / Milvus，未加载 embedding 或 reranker 模型，未调用 VLM。
> 未提交、未推送。

## 0. 基线与冻结

| 项 | 开工前 | 完成后 |
|---|---|---|
| vision freeze | FREEZE OK | FREEZE OK（重设基线，旧清单归档） |
| routing freeze | `CHANGED extractors.py`（已归因于上一轮授权的 Phase 2A-1） | FREEZE OK（重设基线，旧清单归档） |
| pytest | 278 passed | **364 passed, 5 deselected** |

开工前 routing 冻结失败的唯一差异是 `src/extractors.py`，其 sha256 与 `docs/phase2a1_result.md`
记录一致；数据集/历史预测/评测器/配置四组全部与清单一致。属已归因的授权改动，非未解释破损。
旧清单归档为 `outputs/routing_freeze_manifest_pre_phaseA.json` 与 `_pre_phaseB.json`、
`outputs/vision_freeze_manifest_pre_overnight.json`，未覆盖。

## 1. Phase A：中文边界与 span 排除

| 路由 | challenge75 端到端 | 槽位覆盖 |
|---|---|---|
| V1 | 32.0% → **36.0%** | 93.3% → 98.7% |
| V2 | 38.7% → **44.0%** | 89.3% → 94.7% |
| V3 | 52.0% → **57.3%** | 89.3% → 94.7% |

intent exact / tool exact / clarify accuracy / plan-mode accuracy **四项全部不变**，
**新增失败 0 条**，base46 三个路由全部零变化——本次改动只影响槽位抽取。

累计（含上一轮 2A-1）：V3 端到端 **49.3% → 57.3%（+8.0pp）**，
expected slot coverage **86.67% → 94.7%**。

7 条目标用例修复 6 条（B11 B44 来自 2A-1；B12 B13 B15 B16 来自本轮）。
剩余 B71 的原因是阈值词「达到」不在关键词表内，属另一类问题，未在本轮范围内处理。

**challenge75 是已见回归集，用于定向修复；不是盲测，不能证明泛化。**

## 2. Phase B/C/D：结构化切分 → 混合召回 → 证据决策

### 结构指标（分子/分母）

| 指标 | 值 | 数据来源 |
|---|---|---|
| chunk 总数 | 16 | 见下 |
| header_path 保留率 | 7/7 = 100.0% | 表格类 chunk |
| entity_id 保留率 | 10/10 = 100.0% | 设备级 chunk |
| page 溯源覆盖率 | 16/16 = 100.0% | 全部 chunk |
| bbox 溯源覆盖率 | 14/16 = 87.5% | 全部 chunk（缺的 2 条是合成条款 fixture，未提供 bbox） |
| chunk_id 稳定覆盖率 | 16/16 = 100.0% | 全部 chunk |
| **跨设备污染** | 0/10 = 0.0% | 设备级 chunk |

chunk 类型分布：`{'clause': 2, 'drawing_field': 5, 'section': 2, 'table_parent': 2, 'table_row': 5}`

数据来源：

- `repo_mock_drawings` 5 条 —— 来自 `data/mock_drawings.json`，**仓库 MOCK 数据，非企业数据**
- `synthetic_test_fixture` 11 条 —— 本轮构造的结构形状（多级表头、合并单元格），**只用于结构计数，不用于任何效果指标**

### Recall@5

**NOT COMPUTED。** 本仓库没有独立的检索 gold；从系统自身输出反推 gold 会使召回率成为循环论证。

### 服务真实性

| 组件 | 状态 |
|---|---|
| BM25 | 真实 Okapi BM25 算法，**进程内**，`real_service=False` —— 不是 Elasticsearch |
| Dense | `dense_offline_deterministic`，`real_model=False` —— 声明式确定性替身，非 embedding 模型 |
| Reranker | `not_invoked` —— no reranker model installed in this environment |
| Elasticsearch | `not_connected` |
| Milvus | `not_connected` |

检索延迟 p50 0.13ms / p95 0.13ms —— in-process retrieval only; no network, no model inference。

候选来源分布：sparse_only 0、
dense_only 0、both 16。

### 决策分布（7 个场景）

`{'execute': 1, 'clarify': 1, 'partial': 1, 'abstain': 2, 'human_review': 1, 'fallback': 1}`

| 场景 | 决策 | 映射到路由契约 | reason_codes |
|---|---|---|---|
| complete_evidence | **execute** | execute | [] |
| missing_user_parameter | **clarify** | clarify | ['USER_PARAMETER_MISSING'] |
| partial_coverage | **partial** | partial | ['PARTIAL_FIELD_COVERAGE'] |
| non_high_risk_conflict | **abstain** | fallback | ['UNRESOLVED_FIELD_CONFLICT'] |
| high_risk_conflict | **human_review** | fallback | ['HIGH_RISK_FIELD_CONFLICT'] |
| untraceable_value | **fallback** | fallback | ['REQUIRED_EVIDENCE_MISSING'] |
| remediation_exhausted | **abstain** | fallback | ['REMEDIATION_EXHAUSTED'] |

`abstain` 与 `human_review` 在路由契约中没有对应值，统一映射为 `fallback`，
细粒度值保留在本层，映射表见 `src/rag/policy.py::ROUTING_DECISION_MAP`。

## 3. 本轮运行真实链路抓到的两个问题

**① 表头继承越界（已修）**。`备注` 列的 header_path 变成 `[备注, 风量]`——空白单元格
向左继承跨过了不同父级。修复后为 `[备注]`，行文本从 `备注.风量=常用` 变为 `备注=常用`。
已补回归测试 `test_merged_header_does_not_leak_into_the_next_top_level_column`。

**② 无匹配查询仍返回结果（未修，如实记录）**。`完全不存在的查询词组合` 返回 3 条，
因为中文按字切分后与条款共享「的」「组」「合」等常用字。**没有加分数下限掩盖它**——
那是看到输出后调参。列为已知限制。

## 4. 未验证能力

- ❌ 真实 Elasticsearch / Milvus 连接
- ❌ 真实 BGE-Reranker 运行
- ❌ 实时 VLM 调用（决策链路走到该步并记录 `local_vlm_reidentify:not_invoked`）
- ❌ 真实跨页表（仅合成 fixture 验证判定逻辑）
- ❌ 独立 holdout（challenge75 已见，合成 fixture 由本人构造）
- ❌ LangGraph（依赖未安装，见 `docs/routing_freeze_manifest.md` 运行环境节）