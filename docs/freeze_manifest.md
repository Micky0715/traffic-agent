# 冻结清单（视觉解析模块）

本文件记录规则与阈值冻结时的状态。**冻结之后，在盲测运行完成并如实记录结果之前，不得修改任何规则、阈值、字段映射或实体映射。**

## 1. 冻结标识

| 项 | 值 |
|---|---|
| 冻结时间 | 2026-09-15 |
| `configs/visual_parser.yaml` sha256 | `b6f5f1c4bf17cfe697d63eb868f2a72f56aa1d9fccbf4b92c8b60c67b3be6bf4` |
| 配置文件字节数 | 13155 |
| `src/` 全树 sha256（46 个 .py，按路径排序） | `4fee535f3d1b20da43fae3e169ad92ca06d52ffa4f644a71515a9443efea73dc` |
| git HEAD | `86c3cf820d493f0f78ee2ad87c894ad6a0b16086` |
| 工作区是否有未提交改动 | **是** |

⚠️ **git HEAD 不能作为冻结凭据**：本仓库只有一个初始提交，第 1–5 轮全部改动都在未提交的工作区里。**凭据是上面两个 sha256**，不是 commit id。要让 commit 具备凭据效力，必须先提交——本轮按约束未提交。

## 2. 环境版本

```
paddleocr     3.7.0
paddlepaddle  3.3.1   (CPU build, commit 7688495538f4d6c1893f084dd238a402e8f68ab6)
模型          PP-OCRv6_medium_det / PP-OCRv6_medium_rec /
             PP-LCNet_x1_0_doc_ori / PP-LCNet_x1_0_textline_ori / UVDoc
Python        3.12.10
平台          Windows-11-10.0.26200-SP0
CPU           Intel64 Family 6 Model 186，无 CUDA
VLM           VLM_MODEL_NAME（当前 qwen-vl-max），prompt 版本 v1
```

## 3. 冻结时的测试与指标

```
pytest -q            206 passed, 5 deselected
pytest -m real_ocr   5 passed（上次运行 172.97s）
```

回归集（11 张，fixture OCR 引擎，VLM 全缓存命中）：

| 指标 | 开发集(7) | 回归集(4) | 合计(11) |
|---|---|---|---|
| 兜底触发率 | 57.1% | 50.0% | 54.5% |
| 代理兜底召回率 | 100.0% | 100.0% | 100.0% |
| 代理误触发率 | 25.0% | 33.3% | 28.6% |
| VLM 字段恢复率 | 25.0% | 0.0% | 22.2% |
| legacy_containment_hit_rate | 78.6% | 93.8% | 84.1% |
| structured_candidate_recall | 78.6% | 87.5% | 81.8% |
| conflict_free_exact_rate | 78.6% | 87.5% | 81.8% |
| **decision_ready_exact_rate** | 71.4% | 81.2% | **75.0%** |
| 字段冲突数 | 0 | 0 | 0 |
| 非法字段值数 | 1 | 0 | 1 |
| 实体已对齐数 | 5 | 3 | 8 |
| 实体 unresolved 数 | 0 | 0 | 0 |

## 4. 冻结范围（这些不得再改）

- `configs/visual_parser.yaml` 全部内容：路由阈值、字段完整度阈值、水印参数、字段值规则与 origin、实体受控映射与正则、`min_context_similarity`
- `src/vision/` 下的判定逻辑：`page_router.route_ocr_page`、`field_completeness`、`value_validation`、`entity_ref`、`evidence`
- 指标口径：四个命中率的定义、代理标签定义

**允许改动**：盲测运行脚本、报告生成、文档。不得因为盲测结果不好而回头改上面任何一项并冒充盲测。

## 5. 冻结时已知的、盲测应当检验的薄弱点

按预期失败可能性排序：

1. **`drawing_no_format` 规则 origin 是 `dataset_induced`** —— 由本仓库 11 张图归纳，最可能在新编号方案上误判。
2. **`equipment_code_charset` 假设编码不含中文** —— 在某些企业图纸上可能不成立。
3. **真实冲突数为 0** —— 冲突检测与 `unresolved` 分支在真实数据上从未触发，只有单测覆盖。
4. **`aligned_spatially` 结构上不可达** —— VLM 适配器不返回 bbox，该方法缺少一侧数据，永远为 0。
5. **多设备图恒定 `group_cardinality_uncertain`** —— 必然触发兜底，会拉高误触发率。
6. **`decision_ready_exact_rate` 继承路由准确率** —— 该口径把「路由判定无需兜底」视为归属可信。若路由放行了本该升级的页，该口径会高估。
7. **实体受控映射只覆盖 6 个类型词 + 2 条正则** —— 新图纸出现屏蔽门、变压器等类型时会落到 `unknown`。

## 6. 台账与局部重识别：本轮及后续暂不实现

按约定，**在新数据中出现真实字段冲突之前**，不实现：

- 冲突后的局部区域重识别
- 设备台账权威裁决

理由（已实测）：`mock_assets.json` 5 条 + `mock_drawings.json` 2 条，46 个 gold 值仅命中 2 个（A16/A17，且 OCR 本就读对）。当前真实冲突数为 0。两者都会是真实数据从未执行的生产路径。


---

# 勘误（补记）：本清单第 1 节的冻结范围定义有缺陷

**上面记录的哈希值一律保持原样，不做任何事后修改。** 这一节说明它们为什么不再适合做冻结凭据。

## 缺陷

第 1 节用 **整个 `src/` 树** 的 sha256 作为代码侧凭据。这个范围同时犯了两个方向的错：

**太宽。** 视觉链路从不 import 的模块（后来新增的 `src/routing/`）一旦落盘，
全树哈希立刻失配。冻结校验于是报出一个并非变更的"变更"。
**一个会因无关改动而误报的哨兵，很快就不会有人再认真看它** —— 这比没有哨兵更糟。

**也太窄（如果按直觉收窄）。** 最自然的修法是改成只哈希 `src/vision/`。
实测这样会漏掉视觉链路真正依赖的 **4 个文件**：

- `src/drawing_extractor.py`
- `src/llm_router.py`
- `src/models.py`
- `src/tools.py`

改动其中任何一个都会改变解析行为，而目录式冻结**仍然报绿**。

## 修正

正确范围既不是一个目录，也不是一份手工维护的清单，而是
**从声明的入口算出的传递 import 闭包 + 解析配置**，每次运行重新计算。

- 工具：`scripts/vision_freeze.py`（`--write` / `--verify`）
- 清单：`outputs/vision_freeze_manifest.json`
- 当前范围：**21 个代码文件 + 1 个配置**
- combined sha256：`1231065b015c935b267b8519c1c5419a420d4ae2c58f5c26cf3b16a3b949ee9a`

清单会记录**解析出的文件列表本身**，因此"范围变了"（管线新增了一个依赖）
会作为 `ADDED` 显示出来，而不是被静默吸收——这正是全树哈希无法区分的两件事。

已验证的三种行为：

| 场景 | 期望 | 实测 |
|---|---|---|
| 新增无关模块（`src/dummy_unrelated/`） | 不报警 | ✅ FREEZE OK |
| 修改范围内文件（`src/llm_router.py`） | 报警 | ✅ CHANGED |
| 还原后 | 恢复 | ✅ FREEZE OK |

## 对已完成工作的影响

**对抗压力测试未被污染。** 依据：

1. 压测运行在 `src/routing/` 新增**之前**，当时的运行前校验通过；
2. `src/vision` 全树哈希 `a8b90fcc0c12b977…` **至今逐位一致**；
3. `configs/visual_parser.yaml` 哈希至今一致。

但严格地说，那次运行使用的是**范围过宽的凭据**，它当时恰好没有失效，
不代表凭据本身是可靠的。**压测结论有效，凭据方法无效**，两者分开记。

`scripts/run_adversarial_stress.py` 中硬编码的旧期望值**不修改**——
它是那次运行实际用了什么的记录，改掉就等于让脚本与它产出的报告对不上。
后续运行改用 `scripts/vision_freeze.py --verify`。
