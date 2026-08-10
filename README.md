# 轨交故障处置 RAG / Agent：意图识别与工具调用复现包

这个仓库用于复现简历中的核心链路：

`用户问题 -> 多意图拆解 -> 参数抽取 -> 澄清判断 -> 工具规划 -> 并行/条件执行 -> 结果充分度检查 -> 汇总回答`

重点不是把完整企业系统伪造出来，而是把面试最容易被追问的部分做成可以运行、可以展示 Bad Case、可以继续扩展的最小工程。

## 1. 已包含内容

- 单标签 V1：复现“多意图只识别一个标签”的早期问题；
- 多意图 V2：支持一问多查、工具映射、参数抽取、并行和条件执行；
- Guardrail V3：处理否定范围、引号关键词、时间冲突和禁止访问数据源；
- 3 个模拟工具：`hybrid_search`、`nl2api_query`、`drawing_search`；
- LangGraph 版本工作流；
- 46 条常规回归集与 10 条对抗 Bad Case；
- 完整 Router、Validator、充分度判断、答案汇总和评测 Prompt；
- 自动输出 JSON、CSV 和 Markdown 评测报告；
- 面试讲稿和 Bad Case 复盘材料。

## 2. 目录

```text
metro_agent_repro/
├─ prompts/                 完整 Prompt
├─ schemas/                 意图规划 JSON Schema
├─ data/                    评测集与模拟业务数据
├─ src/                     路由、工具、执行器、评测代码
├─ outputs/                 已生成的评测与演示结果
└─ interview/               面试讲稿和追问答案
```

## 3. 快速运行

```bash
cd metro_agent_repro
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3.1 跑常规回归集

```bash
python -m src.run_eval --router all
```

### 3.2 跑对抗 Bad Case

```bash
python -m src.run_eval \
  --router all \
  --dataset data/challenge_cases.jsonl \
  --prefix challenge_
```

### 3.3 查看一个多意图规划

```bash
python -m src.run_case \
  "查 JTG D81-2017 的应急照明要求，再统计 A12风机最近3天告警次数，并给处理建议。"
```

### 3.4 执行模拟工具

```bash
python -m src.run_case \
  "查 JTG D81-2017 的应急照明要求，再统计 A12风机最近3天告警次数，并给处理建议。" \
  --execute
```

### 3.5 使用真实大模型验证 Prompt

复制 `.env.example` 为 `.env`，配置 OpenAI 或兼容接口：

```bash
python -m src.run_llm_case \
  "查规范、查告警次数，并给处理建议"
```

`src/llm_router.py` 使用 JSON mode，再通过 Pydantic 做二次校验。实际接入 Qwen、DeepSeek 或公司网关时，可修改 `OPENAI_BASE_URL` 与 `MODEL_NAME`。

### 3.6 运行 LangGraph 工作流

```bash
python -m src.langgraph_app \
  "先查 A12风机最近3天振动值，如果超过4.5mm/s再查检修规范。"
```

## 4. 当前离线结果

### 常规回归集：46 条

| 版本 | 意图集合完全匹配 | 工具集合完全匹配 | 端到端用例成功率 |
|---|---:|---:|---:|
| V1 单标签 | 47.8% | 65.2% | 34.8% |
| V2 多意图 | 100.0% | 100.0% | 100.0% |
| V3 Guardrail | 100.0% | 100.0% | 100.0% |

### 对抗集：10 条

| 版本 | 意图集合完全匹配 | 工具集合完全匹配 | 端到端用例成功率 |
|---|---:|---:|---:|
| V1 单标签 | 40.0% | 50.0% | 30.0% |
| V2 多意图 | 40.0% | 50.0% | 30.0% |
| V3 Guardrail | 100.0% | 100.0% | 100.0% |

这些结果只反映本仓库合成数据上的回归表现，不能替代真实项目的 200 条人工标注评测。简历中的 61%→78% 必须用真实测试集、真实模型和固定评测口径复测，不能直接拿这里的数字替换。

## 5. 面试最值得展示的三个 Case

### Case A：真实多意图

```text
查 JTG D81-2017 的应急照明要求，
再统计 A12风机最近3天告警次数，
并给处理建议。
```

正确规划：

- T1：`regulation_lookup -> hybrid_search`
- T2：`metric_query -> nl2api_query`
- T1、T2 并行；
- T3：`fault_diagnosis -> none`，依赖 T1、T2；
- 执行模式：`parallel_then_synthesize`。

### Case B：信息不完整

```text
查一下它最近报警几次，顺便看看规范要求。
```

正确行为：识别统计与规范两个意图，但“它”无法确定设备。先返回澄清问题，不执行工具，不猜测设备编号。

### Case C：条件调用

```text
先查 A12风机最近3天振动值，
如果超过4.5mm/s再查检修规范。
```

正确行为：先执行 `nl2api_query`，读取上游结果并判断阈值，满足条件才执行 `hybrid_search`。

## 6. 工业化扩展点

当前模拟工具可替换为：

- `hybrid_search`：Elasticsearch BM25 + Milvus + RRF + BGE-Reranker；
- `nl2api_query`：Schema 检索、只读 SQL/API 生成、参数校验、权限校验；
- `drawing_search`：Qwen2.5-VL 结构化字段、图号索引和页码定位；
- trace：记录标准化问题、规划 JSON、工具参数、耗时、错误、重试和最终证据；
- 评测：接入真实 200 条人工标注集，并按单意图、多意图、缺参数、否定、条件执行分层统计。
