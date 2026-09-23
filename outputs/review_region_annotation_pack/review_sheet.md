# bbox Gold 人工审核表

> **不要顺着系统候选框标。** 候选框是被评测的对象，不是参考答案。

填写 `data/review_region_gold_unreviewed.jsonl`，再跑 `python scripts/validate_review_region_gold.py`。

## FAN-A13-02

- 原图：`pages/FAN-A13-02.png`　叠加图：`overlays/FAN-A13-02.overlay.png`
- 尺寸 1000×700，类型 `fan_wiring`（已知：是），OCR 块 16，表格 6x2

### 系统候选（全部）

| # | strategy | trust | 字段 | bbox | 面积 | 可自动答 |
|---|---|---|---|---|---|---|

### 待填 Gold

| 字段 | 触发原因 | gold_bbox | gold_region_type | field_visible | value_visible |
|---|---|---|---|---|---|

## FAN-A22-01-heavyblur

- 原图：`pages/FAN-A22-01-heavyblur.png`　叠加图：`overlays/FAN-A22-01-heavyblur.overlay.png`
- 尺寸 1000×700，类型 `unknown`（已知：**否**），OCR 块 0，表格 —

### 系统候选（全部）

| # | strategy | trust | 字段 | bbox | 面积 | 可自动答 |
|---|---|---|---|---|---|---|
| 0 | `full_page_diagnostic` | diagnostic | * | [0, 0, 1000, 700] | 100.0% | 否 |

### 待填 Gold

| 字段 | 触发原因 | gold_bbox | gold_region_type | field_visible | value_visible |
|---|---|---|---|---|---|

## FAN-A23-01

- 原图：`pages/FAN-A23-01.png`　叠加图：`overlays/FAN-A23-01.overlay.png`
- 尺寸 1000×700，类型 `fan_wiring`（已知：是），OCR 块 11，表格 6x2

### 系统候选（全部）

| # | strategy | trust | 字段 | bbox | 面积 | 可自动答 |
|---|---|---|---|---|---|---|
| 0 | `exact_label_right` | precise | 图号 | [212, 473, 948, 518] | 4.7% | 是 |
| 1 | `exact_label_right` | precise | 断路器编号 | [212, 592, 948, 638] | 4.8% | 是 |
| 2 | `table_or_title_block` | contextual | 名称, 控制柜编号, 电机编号 | [40, 460, 962, 681] | 29.3% | 否 |

### 待填 Gold

| 字段 | 触发原因 | gold_bbox | gold_region_type | field_visible | value_visible |
|---|---|---|---|---|---|
| **名称** | required_field_missing | ______ | ______ | __ | __ |
| **图号** | required_field_missing, isolated_label_without_value | ______ | ______ | __ | __ |
| **控制柜编号** | required_field_missing | ______ | ______ | __ | __ |
| **断路器编号** | required_field_missing, isolated_label_without_value | ______ | ______ | __ | __ |
| **电机编号** | required_field_missing | ______ | ______ | __ | __ |

## FAN-A24-01

- 原图：`pages/FAN-A24-01.png`　叠加图：`overlays/FAN-A24-01.overlay.png`
- 尺寸 1000×700，类型 `fan_wiring`（已知：是），OCR 块 16，表格 4x2

### 系统候选（全部）

| # | strategy | trust | 字段 | bbox | 面积 | 可自动答 |
|---|---|---|---|---|---|---|
| 0 | `alias_label_right` | structural | 电机编号 | [212, 562, 947, 638] | 8.0% | 否 |
| 1 | `table_or_title_block` | contextual | 名称, 图号, 控制柜编号, 断路器编号, 页码|版本 | [37, 458, 965, 682] | 29.6% | 否 |

### 待填 Gold

| 字段 | 触发原因 | gold_bbox | gold_region_type | field_visible | value_visible |
|---|---|---|---|---|---|
| **名称** | required_field_missing | ______ | ______ | __ | __ |
| **图号** | required_field_missing | ______ | ______ | __ | __ |
| **控制柜编号** | required_field_missing | ______ | ______ | __ | __ |
| **断路器编号** | required_field_missing | ______ | ______ | __ | __ |
| **电机编号** | required_field_missing | ______ | ______ | __ | __ |
| **页码|版本** | required_field_missing | ______ | ______ | __ | __ |

## FAN-MULTI-01

- 原图：`pages/FAN-MULTI-01.png`　叠加图：`overlays/FAN-MULTI-01.overlay.png`
- 尺寸 1000×800，类型 `fan_group`（已知：是），OCR 块 28，表格 12x2

### 系统候选（全部）

| # | strategy | trust | 字段 | bbox | 面积 | 可自动答 |
|---|---|---|---|---|---|---|

### 待填 Gold

| 字段 | 触发原因 | gold_bbox | gold_region_type | field_visible | value_visible |
|---|---|---|---|---|---|
