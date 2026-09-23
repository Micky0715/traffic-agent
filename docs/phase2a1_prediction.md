# Phase 2A-1 预注册：`` 中文边界 Bug 修复

**本文件在任何代码改动和任何评测运行之前写入。** run_id：`phase2a1-run1-20260915-1947`

## 1. 要修什么

`src/extractors.py` 中两个正则以 `` 开头：

```python
REG_PATTERN     = re.compile(r"(?:JTG|TB|GB|CJJ)\s*[A-Z]?\d+[A-Z]?-\d{4}", re.I)
DRAWING_PATTERN = re.compile(r"(?:DWG|TUN|ELEC|MEP|FAN)-?[A-Z0-9-]{2,}", re.I)
```

Python `re` 在 Unicode 模式下**把中文算作 `\w`**。`查JTG` 中 `查` 与 `J` 同为 `\w`，
两者之间**没有词边界**，`` 匹配失败 → 整个 pattern 不匹配。

实测（修复前）：

| 输入 | REG 匹配 |
|---|---|
| `JTG D81-2017` | ✅ |
| `（空格）JTG D81-2017` | ✅ |
| `，JTG D81-2017` | ✅ |
| `查JTG D81-2017` | ❌ **None** |

而中文查询里编号前面**几乎总是紧跟中文字**。

## 2. 连带的第二处：span 排除

`extract_slots` 中有一层保护——抽到 `regulation_code` 后，把它包含的字符串从 `assets` 中剔除，
避免 `JTG D81-2017` 里的 `D81` 被当成设备号。但因为 `regulation` 恒为 `None`，
**这层保护从未生效**。

同类问题图号侧**根本没有保护**：`FAN-A13-02` 里的 `A13` 会被 `ASSET_PATTERN` 抽成设备号，
且没有任何剔除逻辑。

## 3. 改法

1. 把两个正则开头的 `` 换成 `(?<![A-Za-z0-9])`，结尾 `` 换成 `(?![A-Za-z0-9])`。
   **边界按「编码字符集」定义，而不是按 Unicode `\w` 定义** —— 这才是这两个 pattern 的本意。
2. 把「按压缩字符串包含」的剔除改成**按匹配 span 重叠**剔除，并同时覆盖规范号与图号。

## 4. 明确不碰

否定作用域、关键词表（`REG_WORDS`/`METRIC_WORDS`/…）、澄清逻辑、
`validator` 的连坐短路、规则与 LLM 的融合。

## 5. 预期影响（写在运行之前）

**预期好转**
- 7 条「意图工具对但参数错」：`B11 B12 B13 B15 B16 B44 B71`
- `expected_slot_coverage`（代理）从 **86.67%** 上升
- `regulation_code` / `drawing_no` 首次能被正确抽出

**预期可能变差 —— 这是我主动预告的风险**
- `D81` 这类伪 asset 消失后，部分用例的 `slots` 构成改变，
  进而改变 `missing_slots` → 改变 clarify 判定 → 改变 `plan_mode`
- **意图集合匹配、工具匹配、澄清精确率都可能下降**
- **端到端 49.33% 可能不升反降**

**不预期改变**
- base46（V2/V3 已 100% 饱和）
- V1 的意图/工具选择逻辑（V1 不读 regulation_code 做分类）

## 6. 运行纪律

1. challenge75 **只运行一次**，输出前缀 `phase2a1-run1-20260915-1947`；
2. 第一次结果**永久保存**，不覆盖任何既有报告；
3. 指标**变好变坏都如实报告**；
4. **不根据 challenge75 结果继续调正则**；
5. 若发生纯实现故障（脚本崩溃等），保留故障运行结果、换新 run_id、说明原因、不覆盖旧报告；
6. 不提交、不推送。

## 7. 冻结状态

改动前 `scripts/routing_freeze.py --verify` → **FREEZE OK**。
改动后预期出现且**仅**出现：`CHANGED routing_runtime_code: src/extractors.py`。
