# 指标缺口审计：legacy_containment vs structured_candidate_recall

> 纯诊断。本次审计未修改任何代码、配置、阈值、字段规则或数据。
> 范围：回归集 11 张，OCR 引擎 `fixture`（真实 PaddleOCR 结果重放），VLM 全部命中缓存。

## 1. 缺口是否完全由具体字段解释

**是。3 个 gold 字段值解释了全部缺口，无残差。**

| 聚合方式 | legacy | structured | 缺口 |
|---|---|---|---|
| 按用例平均（报告口径，n=11） | 84.1% | 77.3% | **6.8pp** |
| 按字段值（n=46） | 84.8% | 78.3% | **6.5pp** |

两种聚合的差值来自权重不同：OCR11 有 5 个 gold 值，其余各 4 个，按用例平均时每条用例等权。

**校验**：按用例平均的缺口 = (OCR04 的 0.75−0.25) + (OCR10 的 0.75−0.50) = 0.75，0.75 / 11 = **6.82pp**，与报告的 6.8pp 一致。这 0.75 正好由下列 3 行构成，**其余 9 张用例缺口为 0**。

## 2. 全部「legacy 命中但 structured 未命中」明细

### 行 1 — OCR04 / M-23

| 项 | 值 |
|---|---|
| image_id | OCR04（FAN-A23-01.png，水印图，fan_wiring） |
| gold_value | `M-23` |
| gold_field_name | VLM 侧仅以 `device_id` 形式出现 |
| 出现在哪个 blob | **OCR ✓ / VLM ✓** |
| 是否生成 FieldPair | **否** |
| 是否生成 FieldEvidence | **否** |
| entity_id / field_name / raw_value / normalized_value | 均为 `null`（无结构化候选） |
| 未命中原因 | `label_value_pairing_failed` + `vlm_device_id_suppressed_by_flattening` |

**OCR 侧**：OCR 读到的是正文标注块 `电机 M-23`，而**不是标题栏的「电机编号」标签 + 值**。该页的「电机编号」标签被水印覆盖，从未被识别出来，因此标签→值配对无从发生。

**VLM 侧**：VLM 把 M-23 作为 `device_id` 返回，`parameters` 为空数组。而第三轮为避免把电机和断路器两个部件的 id 都塞进页级 `设备编号`（会制造假冲突），在单设备图纸上**关闭了 device_id 的映射**。于是该值在结构化侧完全不可达。

### 行 2 — OCR04 / QF-23

| 项 | 值 |
|---|---|
| image_id | OCR04（同上） |
| gold_value | `QF-23` |
| gold_field_name | VLM 侧仅以 `device_id` 形式出现 |
| 出现在哪个 blob | **仅 VLM** |
| 是否生成 FieldPair | **否** |
| 是否生成 FieldEvidence | **否** |
| entity_id / field_name / raw_value / normalized_value | 均为 `null` |
| 未命中原因 | `vlm_device_id_suppressed_by_flattening` |

OCR 完全没读到该值（水印遮挡，「断路器编号」标签右侧只剩水印文字）。VLM 读到了，但同样只作为 `device_id`，被扁平化规则抑制。

### 行 3 — OCR10 / FAN-CAB-24

| 项 | 值 |
|---|---|
| image_id | OCR10（FAN-A24-01.png，模糊图，fan_wiring） |
| gold_value | `FAN-CAB-24` |
| gold_field_name | VLM 侧标签为 **`控制柜型号`** |
| 出现在哪个 blob | **OCR ✓ / VLM ✓** |
| 是否生成 FieldPair | **否** |
| 是否生成 FieldEvidence | **否** |
| entity_id / field_name / raw_value / normalized_value | 均为 `null` |
| 未命中原因 | `label_value_pairing_failed` + `vlm_field_mapping_missing` |

**OCR 侧**：OCR 文本为
```
…\n电r编号\nM-24\nE银行\nOF-24\n制编\nFAN-CAB-24
```
值 `FAN-CAB-24` 被正确识别，但它的**标签被模糊成了 `制编`**，不等于 `控制柜编号`，标签查表失败，配对不成立。

**VLM 侧**：VLM 返回 `raw_name="控制柜型号"`，而已知字段名集合里是 `控制柜编号`。**差一个字，映射失败。**
（附带观察：同一页 VLM 还返回了 `控制器编号 = CF-24`，与 gold 值不同，属另一潜在问题，本次不处理。）

## 3. 原因归类汇总

| 原因 | 次数 | 说明 |
|---|---|---|
| `label_value_pairing_failed` | 2 | 值被读出，但其标签被水印/模糊破坏，标签→值几何配对失败 |
| `vlm_device_id_suppressed_by_flattening` | 2 | 第三轮为消除多部件假冲突而抑制 device_id 映射的**直接代价** |
| `vlm_field_mapping_missing` | 1 | VLM 用了 `控制柜型号`，映射表只有 `控制柜编号` |
| `value_normalization_mismatch` | **0** | 归一化未造成任何漏配 |
| `evidence_not_generated` | **0** | 没有出现「有 FieldPair 却无 FieldEvidence」 |
| `other` | **0** | 无法归类的残差为零 |

**三个值全部是「FieldPair 未生成 ⇒ FieldEvidence 未生成」**，没有一例是在证据已生成之后才丢失的。

## 4. OCR/VLM entity_id 不一致但未记录 uncertain 的情况

**本次检出 0 条。**

检查方法：对每个 field_name，若 OCR 侧与 VLM 侧各自产生了证据，且两侧的 entity_id 集合**无交集**，则记为不一致，并核对是否已被记为 conflict 或 entity_assignment_uncertain。

**但这个 0 必须打折看**，有两条限制：

1. 该检查只在**同一 field_name 两侧都产生了证据**时才会触发。单设备图纸上 VLM 的 device_id 被抑制（见行 1、2），两侧共同覆盖的字段本就很少，检查可施展的空间有限。
2. 单设备图纸走 `flatten_entities=True`，VLM 侧 entity_id 被强制置为 `None`，与 OCR 侧一致——**不一致在结构上被消除了，而不是被验证不存在**。多设备图纸（OCR02/OCR09）两侧 entity_id 天然对齐（A16/A17/A18），也不产生不一致。

**结论：回归集上没有观察到未记录的 entity_id 不一致，但当前数据无法证明该机制有效。**

## 5. 审计结论

1. **6.8pp 缺口可被 3 个字段值完全解释，残差为 0。**
2. 缺口**不是**归一化造成的（`value_normalization_mismatch` = 0）。
3. 缺口的主导成因有两个，且性质不同：
   - **一半是能力问题**：标签被遮挡/模糊破坏后，标签→值几何配对失效（2 次）。这是 OCR 质量问题在结构化层的必然投影。
   - **一半是第三轮设计取舍的代价**：抑制 device_id 映射消除了假冲突，同时也让单设备图纸上的部件编号在结构化侧不可达（2 次）。
4. 还有 1 次是**词表问题**：`控制柜型号` vs `控制柜编号` 一字之差导致映射失败。
5. `legacy` 之所以更高，正是因为它只要在拼接文本里翻到子串就算命中——**上述三种失败它一个都看不见**。这三条明细就是「字符串里找得到」与「系统能指着一个字段说出答案」之间差距的全部内容。

## 6. 本次未做的事（按要求）

- 未修改任何代码、配置、阈值或字段规则
- 未修改 `vlm_field_mapping`（`控制柜型号` 仍未映射）
- 未调整归一化
- 未改动上一轮及本轮任何既有报告
- 未提交、未推送
