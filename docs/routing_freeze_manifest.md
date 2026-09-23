# 路由实验冻结清单（Phase 2A-0）

冻结时间：2026-09-15T10:55:43+00:00

> **冻结生效后，在 Phase 2A-1 验收完成之前，不得修改 `extractors.py`、
> `router_v2/v3.py`、`validator.py` 或任何路由行为。**

工具：`scripts/routing_freeze.py --write` / `--verify`　清单：`outputs/routing_freeze_manifest.json`

## 1. 分组与哈希

按**输入种类**分组而不是合成一个摘要——数据集变了和评测器变了会让不同的结论失效，
只报「有东西变了」等于让读的人自己猜。

| 组 | 文件数 | sha256 | 来源 |
|---|---|---|---|
| `routing_runtime_code` | 8 | `f89fa82f933ed913c368246e94ef1273…` | 静态 import 闭包 |
| `routing_config_and_prompts` | 3 | `b8ba7b0a1ce29a4bdbed50eb7d3cc4b8…` | 显式清单 |
| `routing_evaluator` | 4 | `f85c291da97a07cb01c60264c9577926…` | 显式清单 |
| `routing_datasets` | 2 | `9bb8c25001c2afce847397697e7631e5…` | 显式清单 |
| `historical_llm_predictions` | 6 | `3e477bd4e2f773fc5c5e585e9e33b2fa…` | 显式清单 |

## 2. import 闭包入口（逐个记录）

| 入口 | 文件 | 闭包文件数 | 闭包内容 |
|---|---|---|---|
| `rule_v1` | `src/router_v1.py` | 4 | `extractors.py`、`models.py`、`normalize.py`、`router_v1.py` |
| `rule_v2` | `src/router_v2.py` | 4 | `extractors.py`、`models.py`、`normalize.py`、`router_v2.py` |
| `rule_v3` | `src/router_v3.py` | 5 | `extractors.py`、`models.py`、`normalize.py`、`router_v2.py`、`router_v3.py` |
| `llm` | `src/llm_router.py` | 2 | `llm_router.py`、`models.py` |
| `validator` | `src/validator.py` | 2 | `models.py`、`validator.py` |

**并集（8 个）**：`src/extractors.py`、`src/llm_router.py`、`src/models.py`、`src/normalize.py`、`src/router_v1.py`、`src/router_v2.py`、`src/router_v3.py`、`src/validator.py`

注意 `rule_v3` 的闭包里含 `router_v2.py`——V3 委托给 V2，改 V2 会同时改变 V3。
`rule_v1` 的闭包不含它，所以 V1 不受影响。逐入口记录就是为了能看出这种差别。

## 3. 静态分析的限制

> 代码依赖范围来自静态 import 分析，动态导入、配置加载、Prompt、模型文件和 subprocess 依赖无法仅靠 import 闭包发现，因此非代码输入采用显式清单补充。

本仓库有两处具体实例，都靠显式清单补上：

**① Prompt 由运行时字符串加载。** `llm_router._call_json(prompt_file)` 执行
`(ROOT / "prompts" / prompt_file).read_text()`，文件名是函数参数。
任何 import 闭包都看不见它。因此 prompt 显式声明，并**反向交叉核对**：
扫描路由代码中的 `.md` 字符串字面量，若有指向真实 prompt 却未被声明的，
报 `UNRESOLVED_DEPENDENCY`。

**② `validator.py` 不被任何 router import。** 它是 `run_eval` 在路由外面套的一层，
却会重写每个 plan 的工具映射、并能把整盘计划强制转成 clarify。
闭包永远找不到它，所以把它作为**独立入口**显式声明。

**当前 unresolved_dependencies：0 条**（无）

## 4. 数据集

| 文件 | 条数 | 说明 |
|---|---|---|
| `data/eval_cases.jsonl` | 46 | base46。**V2/V3 已 100% 饱和，不得再用于任何改进论证** |
| `data/challenge_cases.jsonl` | 75 | challenge75。**只称回归验证集，不称 holdout**——设计规则时已看过 |

原始 10 条对抗集**数据文件已不存在**，仅预测留存，无法重新评分。

## 5. 历史 LLM 预测

6 个**原始**文件。
`outputs/unified_routing_predictions/` 下的转换副本**不在冻结范围内**——
把派生产物和它的源一起冻结，会让一次正常的重新生成看起来像篡改。

## 6. 运行环境

```
Python        3.12.10
平台          Windows-11-10.0.26200-SP0
CPU           Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
pydantic      2.12.4
openai        2.53.0
PyYAML        6.0.2
langgraph     None   ← 未安装；LangGraph 支链本轮无法运行
git HEAD      86c3cf820d493f0f78ee2ad87c894ad6a0b16086
工作区未提交   True
```

⚠️ **git HEAD 不能作冻结凭据**：仓库只有一个初始提交，全部改动都在未提交工作区。
凭据是上面五组 sha256。

## 7. verify 的告警分类

| 标签 | 含义 |
|---|---|
| `CHANGED` | 路由运行时代码被改 |
| `CONFIG_CHANGED` | 路由配置或 Prompt 被改 |
| `EVALUATOR_CHANGED` | 评测器或指标计算被改 |
| `DATASET_CHANGED` | 评测集被改 |
| `PREDICTION_CHANGED` | 历史真实 LLM 预测被改 |
| `ADDED` / `REMOVED` | 某组的文件集合本身变了（闭包新增/丢失依赖） |
| `UNRESOLVED_DEPENDENCY` | 存在无法静态解析的依赖，未被显式补充 |

## 8. 反向验证结果

`tests/test_routing_freeze.py`，17 项全部通过。**全部使用备份/还原，不污染真实数据与历史预测**，
并有一项专门检查「所有篡改是否都已还原」，防止某次还原失败后污染数据悄悄变成新基线。

| # | 场景 | 期望 | 实测 |
|---|---|---|---|
| 1 | 新增无关模块 `src/_freeze_probe_pkg/` | 不报警 | ✅ FREEZE OK |
| 2 | 修改 `src/extractors.py` | `CHANGED` | ✅ |
| 3 | 修改 `configs/routing.yaml` | `CONFIG_CHANGED` | ✅ |
| 4 | 修改 `prompts/01_intent_router_v2.md` | `CONFIG_CHANGED` | ✅ |
| 5 | 修改 `data/challenge_cases.jsonl` | `DATASET_CHANGED` | ✅ |
| 6 | 修改 `src/evaluator.py` | `EVALUATOR_CHANGED` | ✅ |
| 7 | 修改历史 prediction | `PREDICTION_CHANGED` | ✅ |
| 8 | 修改 `src/validator.py` | `CHANGED` | ✅ |
| 9 | 全部还原后 | FREEZE OK | ✅ |
| 10 | 从显式清单移除一个被代码引用的 prompt | `UNRESOLVED_DEPENDENCY` | ✅ |

## 9. Phase 2A-1 的基线（冻结时）

| 路由 | base46 端到端 | challenge75 端到端 |
|---|---|---|
| V1 | 34.78% | 29.33% |
| V2 | 100.00%（饱和） | 36.00% |
| V3 | 100.00%（饱和） | **49.33%** |

challenge75 / V3 细项：意图集合完全匹配 65.33%、工具集合完全匹配 81.33%、
澄清精确率 44.44%、澄清召回 100%、over-routing 14.67%、under-routing 4.00%、
期望槽位覆盖（代理）86.67%。