# 历史真实 LLM 路由预测 · 冻结清单

冻结时间：2026-09-15T09:12:22+00:00
源码树 sha256：`4fee535f3d1b20da43fae3e169ad92ca06d52ffa4f644a71515a9443efea73dc`
git HEAD：`86c3cf820d493f0f78ee2ad87c894ad6a0b16086`　工作区有未提交改动：True

> 这 6 个文件是本仓库路由链路**唯一的真实 LLM 证据**。第二阶段会改变计划的表示方式，
> 因此在动手之前先把它们的字节固定下来。**原文件不被修改，转换结果写入新目录。**

## 文件

| 文件 | sha256 | 字节 | 条数 | 数据集 | 可对上 case | 有 error 的行 |
|---|---|---|---|---|---|---|
| `outputs/llm_challenge_llm_v1_naive_predictions.jsonl` | `ed4149b93401db5d…` | 6755 | 10 | challenge10_original | — | 0 |
| `outputs/llm_challenge_llm_v2_structured_predictions.jsonl` | `b063477b547eff8d…` | 10122 | 10 | challenge10_original | — | 0 |
| `outputs/llm_challenge_v2_llm_v1_naive_predictions.jsonl` | `8cbc54bb3f3e1dda…` | 50452 | 75 | challenge75 | 75 | 0 |
| `outputs/llm_challenge_v2_llm_v2_structured_predictions.jsonl` | `69d8684d87ef73bf…` | 77557 | 75 | challenge75 | 75 | 0 |
| `outputs/llm_llm_v1_naive_predictions.jsonl` | `e058e21f53744776…` | 29770 | 46 | base46 | 46 | 0 |
| `outputs/llm_llm_v2_structured_predictions.jsonl` | `218fda26c2905249…` | 55135 | 46 | base46 | 46 | 0 |

## 未知字段（一律 null，不猜测）

每个文件的行字段只有 `query` / `plan` / `error`，**不包含模型名、prompt 版本或生成时间**。
因此清单中这三项全部记为 `null`，并带上对应 reason_code：

| reason_code | 含义 |
|---|---|
| `MODEL_INFO_UNAVAILABLE_LEGACY` | prediction file records no model name; not inferred |
| `PROMPT_VERSION_UNAVAILABLE_LEGACY` | prediction file records no prompt version; not inferred |
| `GENERATION_TIME_UNAVAILABLE_LEGACY` | prediction file records no generation timestamp |
| `SOURCE_DATASET_FILE_MISSING` | the dataset these predictions were produced against is no longer present in the repo |

## 需要特别说明的一处

`llm_challenge_llm_v1_naive_predictions.jsonl` 与 `llm_challenge_llm_v2_structured_predictions.jsonl` 各 10 条，对应的是**原始 10 条对抗集**。
该数据集文件已不在仓库中——`data/challenge_cases.jsonl` 现在是扩充后的 75 条版本，
文件名前缀 `llm_challenge_v2_` 才对应它。

因此这两个文件标记 `SOURCE_DATASET_FILE_MISSING`：**预测留存，gold 已失，无法重新评分**，
转换时 case_id 只能用行号占位。这不是可以补救的，补一份 gold 等于重新发明答案。

## 使用约束

1. 不修改原 predictions 文件；
2. 转换结果写入 `outputs/unified_routing_predictions/`，不覆盖原文件；
3. 未知字段保持 null，不得事后补写推测值；
4. 任何引用这些数字的场合，必须同时说明模型名与 prompt 版本未知。