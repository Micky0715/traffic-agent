# 结构化 BM25 最小 RAG 闭环 · 交付报告

> 零 API 调用。未连接 Elasticsearch / Milvus，未加载 BGE Embedding，
> 未运行 Reranker，未调用 VLM。未提交、未推送。

## 0. 服务真实性声明

| 项 | 状态 |
|---|---|
| Elasticsearch | **not_connected** |
| Milvus | **not_connected** |
| BGE Embedding | **not_loaded** |
| Reranker | **not_invoked** |
| VLM | **not_invoked** |
| 人工审核 Gold | **False** |
| Recall@5 | **not_computed_no_human_reviewed_gold** |
| 检索器 | bm25_local，算法 okapi_bm25，后端 in_process |
| 分词器 | `latin_code_runs_plus_cjk_unigram`，停用词启用 True |
| 缓存 | ocr fixtures are a replay of a recorded real PaddleOCR run; no live OCR in this path |
| 真实主入口 | 是 —— `python -m src.run_structured_rag_query`，复用 `StructuredRagService`，与测试同一对象 |

## 1. 语料：真实 / Mock / synthetic 分开计数

- **真实解析产物**：11 个文档、**63 个 chunk**
  - 来源：data/ocr_fixtures/*.json — replay of a recorded real PaddleOCR 3.7.0 run, stamped with image sha256, library and model versions, device and generation time
  - 实时推理：**False**（是录制后重放）　企业数据：**False**
  - field_pairs 54 条，其中带 bbox **54** 条
- synthetic chunk：**0**（本链路未使用，仅单测中出现）
- 手写 mock chunk：**0**（本链路未使用）

chunk 类型分布：`{'drawing_metadata': 9, 'drawing_field': 54}`

## 2. 溯源覆盖率

| 指标 | 分子/分母 |
|---|---|
| 页码覆盖 | 63/63 = 100.0% |
| 字段 chunk 的 bbox 覆盖 | 54/54 = 100.0% |

## 3. 五类端到端查询

| 场景 | 查询 | retrieved | eligible | 决策 |
|---|---|---|---|---|
| execute | 查询A13风机接线图的电机编号 | 20 | 1 | **execute** |
| clarify_missing_device | 查询电机编号 | 20 | 6 | **clarify** |
| partial | 查询A28风机参数表的控制柜编号和页码 | 20 | 1 | **partial** |
| abstain_unknown_device | 查询A99风机的电机编号 | 20 | 0 | **abstain** |
| abstain_invalid_value | 查询A19风机接线图的控制柜编号 | 20 | 0 | **abstain** |
| irrelevant_query | 完全不存在的查询词组合 | 0 | 0 | **clarify** |

**retrieved 与 eligible 的差**就是本轮的核心：BM25 永远返回 TopK，
返回 20 条不等于有 20 条证据。只有 eligible 才能支撑回答。

### execute 的证据引用

```
answerable_fields : {'电机编号': 'M-13'}
evidence          : FAN-A13-02  page=1  bbox=[206.0, 597.0, 261.0, 622.0]
                    chunk=drawing_field:FAN-A13-02:09006c464cd6f04c  source=ocr
```

## 4. 派生回归集（30 条）

- `label_source=derived_from_existing_gold`　`purpose=regression_only`
- questions were written FROM the answers; this catches regressions and cannot show generalization

决策分布：`{'execute': 22, 'abstain': 5, 'clarify': 3}`

| 指标 | 分子/分母 |
|---|---|
| execute 中取值正确 | 22/22 = 100.0% |
| execute 中文档正确 | 22/22 = 100.0% |

平均 retrieved 18.27 → 平均 eligible 1.13
（延迟 p50 0.737ms / p95 0.907ms，进程内、无网络）

### ⚠️ 8 条非 execute 中，7 条是派生集自己的期望值错了

这恰恰证明了「从已有输出反推 gold」的危险：

| case | 系统决策 | 派生集的期望 | 谁对 |
|---|---|---|---|
| DRV-005 | abstain | `控制柜编号='审核专用毫AB9'` | **系统对**——该值被印章污染，已判 invalid |
| DRV-024/025/026 | abstain | FAN-MULTI-01 的单一功率/风量/设备编号 | **系统对**——该图含 3 台设备，问「这张图的功率」本就有歧义 |
| DRV-027/028/029 | clarify | PUMP-MULTI-02 的单一扬程/流量/编号 | **系统对**——同样 3 台设备，系统要求澄清 |
| DRV-020 | abstain | `图号='FAN-A28-01'` | **系统错**——真实召回丢失，见下 |

### DRV-020：唯一一条真实缺陷

```
查询 '查询A28风机参数表的图号'
  6 个 图号 chunk 分数完全并列 1.226，按索引顺序排
  目标 chunk 实际排名 21，而 sparse_top_k = 20 —— 差一位掉出
```
**未把 top_k 改成 21**：那是拿评测结果调参。记为已知缺陷，
top_k 与并列打破规则应在人工审核过的 dev 集上校准。

## 5. 无关召回问题：根因与处理

```
query  '完全不存在的查询词组合'
  无停用词 token: ['完', '全', '不', '存', '在', '的', '查', '询', '词', '组', '合']
  有停用词 token: ['完', '全', '存', '询', '词', '组', '合']
```
根因：CJK tokenizes per character; 的/不 are shared with almost any Chinese text, so BM25 scored above zero against everything and returned its top K

- **主要防线**：eligibility gate — a chunk must be about the right entity and cover a requested field
- 次要防线：configured stopwords in the tokenizer
- 分数阈值：`None`，状态 **uncalibrated**
  - a floor needs calibration on a dev set; choosing one because a single query misbehaved is guesswork

## 6. 评测候选集（待人工审核）

- 文件：`data/rag_eval_candidates_unreviewed.jsonl`，共 **42 条**
- `label_status=unreviewed`，`human_reviewed_gold=False`
- machine-proposed expectations. No Recall@5, refusal rate or accuracy may be computed from these until a person has reviewed them. They are NOT gold.

审核表格：`docs/rag_eval_candidates_review.md`

## 7. 未实现能力

- ❌ Elasticsearch / Milvus 服务联调
- ❌ BGE Embedding / Dense 向量检索（本模式不使用）
- ❌ BGE-Reranker
- ❌ 实时 VLM 调用
- ❌ 人工审核 Gold → 因此**无 Recall@5**
- ❌ 跨页表（真实语料均为单页图纸）
- ❌ 真实字段冲突（本语料冲突数为 0，冲突路径仅单测覆盖）