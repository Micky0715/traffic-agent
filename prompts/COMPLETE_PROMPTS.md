# 完整 Prompt 清单

本目录包含从早期单标签版本到可执行多意图版本的完整 Prompt：

1. `00_intent_router_v1_bad.md`：故意保留单标签缺陷，用于复现早期 Bad Case。
2. `01_intent_router_v2.md`：核心多意图拆解、槽位抽取、依赖规划与工具选择 Prompt。
3. `02_plan_validator.md`：工具计划二次校验与修复 Prompt。
4. `03_result_sufficiency.md`：结果充分度和重试/拒答判断 Prompt。
5. `04_answer_synthesizer.md`：带证据的最终答案汇总 Prompt。
6. `05_eval_judge.md`：离线评测裁判 Prompt。

建议线上链路不要让一个 Prompt 同时负责所有事情，而是采用：

`Router -> Validator -> Executor -> Sufficiency Checker -> Synthesizer`

这样可以分别定位漏意图、工具选错、参数错误、结果不足和答案幻觉。
