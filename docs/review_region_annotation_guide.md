# bbox Gold 人工标注说明

给标注人看的，不需要懂代码。**预计 11 条，第一次做大概 30–45 分钟。**

---

## 0. 一句话：你要做什么

对每一个「某张图纸上的某个字段」，用眼睛在原图上找到**这个字段的值印在哪里**，
把那个位置框出来，写成四个数字。就这样。

---

## 1. 最重要的一条规矩

> **不要顺着系统画好的框标。**

审核包里那些彩色框是**机器猜的**，它们正是你要评判的对象。
如果你看一眼机器的框然后照着描一遍，那这份 Gold 就等于机器自己的答案，
拿它算出来的准确率一定很高，而且**完全没有意义**。

正确顺序是：

1. 先看 `pages/` 里的**原图**，自己找到字段值；
2. 心里（或纸上）定好框；
3. **然后**才去看 `overlays/` 里的彩色框，看机器差多少。

---

## 2. 框什么：三种情况

### 情况 A：能看到值本身 → 只框值

比如图上写着 `控制柜编号  FAN-CAB-24`，你要框的是 **`FAN-CAB-24` 这几个字**，
不要把 `控制柜编号` 这个标签也框进去。

```
gold_region_type = value_cell
field_visible    = true
value_visible    = true
```

### 情况 B：标签和值挨在一起分不开 → 框整个格子

如果标签和值挤在同一个表格单元格里，或者中间没有明显分界，框**整个标签+值**。

```
gold_region_type = label_value_pair
```

### 情况 C：看得出字段属于某块区域，但值本身找不到 → 框那块区域

比如整个标题栏能看见，但这一行被水印糊住了，你知道它应该在标题栏里但认不出来：

```
gold_region_type = table_region   （或 title_block）
field_visible    = true
value_visible    = false
```

**这种情况请务必如实标 `value_visible=false`。**
它不是"标不好"，它恰恰是最有价值的一条——它告诉我们机器给出的大区域到底对不对。

---

## 3. 各种看不清的情况怎么办

| 你看到的 | 怎么标 |
|---|---|
| 值清清楚楚 | 框值，`value_visible=true` |
| 被水印盖着，但**眯眼还能认出来** | **照常框值**，`value_visible=true`，在 `notes` 写 `watermark` |
| 被水印/印章盖住，**认不出来** | 框它所在的区域，`value_visible=false` |
| OCR 没读到，但你**肉眼看得见** | **照常框值**。OCR 读没读到跟你无关，这正是我们要测的 |
| 整页糊成一片，什么都认不出 | `gold_region_type=unlocatable`，写 `unlocatable_reason` |
| 这张图**根本没有这个字段** | `gold_region_type=unlocatable`，`field_visible=false`，原因写 `field_not_present_on_this_drawing` |

**关键：不要因为"机器没找到"就跟着标成找不到。** 你的眼睛是标准，机器不是。

---

## 4. 什么时候用 `excluded`

拿不准的时候用它，不要硬标。

```
label_status = excluded
notes        = 写清楚为什么拿不准
```

比如：
- 这一页有两个地方都像是控制柜编号，你分不清哪个是；
- 图纸本身画错了或者印重了；
- 你觉得需要懂这个专业的人来看。

`excluded` 的记录**不进入任何指标的分母**，所以它不会拉低也不会抬高任何数字。
**一条诚实的 `excluded` 比一条勉强的 Gold 有用得多。**

---

## 5. 怎么写那四个数字

格式是 `[x0, y0, x1, y1]`，单位是**原图像素**：

- `x0, y0` = 框的**左上角**（x 向右，y 向下）
- `x1, y1` = 框的**右下角**
- 必须 `x1 > x0` 且 `y1 > y0`，不能有负数，不能超出图片边界

**如果你看的是 `preview.png`（缩小版）**，HTML 页面上会写明缩放比例，
换算方法是：`原图坐标 = 预览上量到的坐标 ÷ 缩放比例`。
用原图（`pages/` 里那张）量最省事，就不用换算。

用任何看图软件都行（Windows 照片、画图、IrfanView…），
把鼠标放到框的左上角和右下角，读出坐标即可。

---

## 6. 具体操作步骤

1. 用浏览器打开
   `outputs/review_region_annotation_pack/index.html`
2. 从上往下，一页一页看。每页会列出**这一页需要你判断的字段**。
3. **先看原图**（`pages/` 目录里那张干净的），自己找字段值。
4. 打开 `data/review_region_gold_unreviewed.jsonl`，**一行就是一条记录**，
   找到 `record_id` 对应的那一行，把这几项填好：

```json
{
  "record_id": "FAN-A24-01:p1:控制柜编号",
  "label_status": "human_reviewed",
  "gold_bbox": [220, 630, 520, 659],
  "gold_region_type": "value_cell",
  "field_visible": true,
  "value_visible": true,
  "annotator": "你的名字",
  "annotated_at": "2026-09-17",
  "notes": "水印覆盖，眯眼可辨"
}
```

> 其他字段（`document_id`、`image_sha256` 等）**不要改**。
> `image_sha256` 是用来确认你标的是这张图、不是后来换过的图。

5. 全部填完后运行：

```bash
python scripts/validate_review_region_gold.py
```

它会检查格式、坐标是否越界、`human_reviewed` 有没有写名字和日期等。
**有问题会报错并列出行号**，改完再跑一次，直到通过。

6. 通过后运行：

```bash
python scripts/evaluate_review_region_localization.py
```

**这一步之前，系统拒绝输出任何定位准确率**——因为没有答案就没有准确率。

---

## 7. 不用担心的事

- **标错了可以改**，改完重跑校验就行。
- **不用一次标完**，标几条就存，没标的还是 `unreviewed`，不影响。
- **不用凑数**，11 条里如果只有 6 条你有把握，就标 6 条，其余留 `unreviewed` 或 `excluded`。
  样本少会在报告里如实写明，不会被当成准确率高或低。

---

## 8. 一句话总结

> 用你的眼睛在原图上找到字段值，框住它；
> 看不清就如实说看不清；拿不准就 `excluded`。
> **千万别照着彩色框描。**


---

## 9. Schema v2：值看得清，但不知道是哪个字段的

> 第 1–8 节写的是 v1。v1 有一个说不出口的情况，这一节补上。**你之前按 v1 标的记录不用改，也不会被自动改。**

### 9.1 v1 为什么不够用

设想图上印着 `FAN-CAB-24`，字你认得清清楚楚，可它左边的标签被印章糊掉了，
你**没法确定**它是不是"控制柜编号"——它也可能是别的编号。

v1 只能二选一：

- 标"值可见"，那就等于你确认了它是控制柜编号——**你并没有确认**；
- 标"值不可见"，可你明明看清了——**这也不对**。

v2 把这件事拆成三个问题，**每个问题单独回答**：

| 问题 | 字段 | 怎么答 |
|---|---|---|
| 图上**有没有**一个值印在那儿？ | `value_visible` | 有墨迹就 `true`，哪怕被水印压着 |
| 这个值的字你**认不认得出**？ | `value_legible` | 能可靠读出每个字符才 `true` |
| 你能不能**确认它属于这个字段**？ | `association_status` | 见下表 |

三者互不替代：**看得见 ≠ 认得出 ≠ 知道是谁的**。

### 9.2 `association_status` 四个取值

| 取值 | 什么时候用 | 要填什么 |
|---|---|---|
| `confirmed` | 值认得出，**并且**你能确认它就是这个字段的值 | `gold_bbox` 框值；`value_legible=true` |
| `ambiguous` | 值认得出，但它**可能是好几个字段**中的一个 | `candidate_bbox` 框值；`candidate_fields` 至少列 2 个字段（含目标字段）；`gold_bbox` 留空 |
| `unassigned` | 值认得出，但你**没法把它分给**这个字段 | `candidate_bbox` 框值；`gold_bbox` 留空 |
| `unknown` | 值**认不出来**，或者这条记录没有回答这个问题 | 不填框 |

`ambiguous` 和 `unassigned` 时，`gold_region_type` 要标 `unlocatable`——
因为**字段**确实定位不了；你看到的那个**候选值**的位置放进 `candidate_bbox`，
不要放进 `gold_bbox`。`gold_bbox` 只留给"我确认这个字段就在这里"。

### 9.3 三个例子

**例 1：清清楚楚**

```json
"schema_version": "region_gold/2",
"value_visible": true, "value_legible": true,
"association_status": "confirmed",
"gold_region_type": "value_cell", "gold_bbox": [220, 480, 940, 510]
```

**例 2：值认得出，标签被毁**

```json
"schema_version": "region_gold/2",
"value_visible": true, "value_legible": true,
"association_status": "unassigned",
"gold_region_type": "unlocatable",
"unlocatable_reason": "label_destroyed_value_legible",
"gold_bbox": null, "candidate_bbox": [220, 630, 940, 660]
```

**例 3：糊得认不出**

```json
"schema_version": "region_gold/2",
"value_visible": true, "value_legible": false,
"association_status": "unknown",
"gold_region_type": "unlocatable",
"unlocatable_reason": "severe_blur"
```

### 9.4 千万别做的事

- **别因为 OCR 读出了字就标"认得出"。** OCR 读出来的不是 Gold。
  你自己看原图认不出，就是 `value_legible=false`。
- **别按行的顺序推断归属。** "控制柜编号通常在最后一行"不是你看到的证据。
- **别把旧记录"顺手升级"成 `confirmed`。** 旧记录要改，请整条重新看原图，
  另存为新文件，写上新的 `annotator` 和 `annotated_at`。
- **别把系统的候选框抄成 `candidate_bbox`。** 校验器会对完全一致的框发出提醒。

### 9.5 旧记录怎么办

v1 记录读进来时，`association_status` 一律当 `unknown`，**不会被当成 `confirmed`**。
这意味着按 v2 的口径，旧记录**不进入正式的定位准确率**——不是因为你标错了，
而是 v1 从来没问过"归属"这个问题。

想让它们进入正式评测，只能**人工重新看图**并补上 `value_legible` 和
`association_status`，存为新文件。可以先用下面的命令看看迁移会改什么（它什么也不写）：

```bash
python scripts/migrate_review_region_gold.py --gold data/review_region_gold_test.jsonl
```

迁移工具只会改 `schema_version` 并记下 `migrated_from`，**不会替你填任何判断**。
