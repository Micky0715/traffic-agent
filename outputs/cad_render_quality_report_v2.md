# CAD 渲染质量报告 v2

- 状态：`audited`
- pdftoppm：`C:\Users\jdh07\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin\pdftoppm.EXE` — pdftoppm version 25.07.0
- pdfinfo：`C:\Users\jdh07\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin\pdfinfo.EXE` — pdfinfo version 25.07.0
- 渲染命令：`pdftoppm -r 300 -png -f N -l N -singlefile <pdf> <out>`，DPI 300

## 渲染

| 项 | 值 |
|---|---|
| PDF 数 | 10 |
| pdfinfo 总页数 | 131 |
| PyMuPDF 临时页数 | 131 |
| 两者一致 | True |
| 本批渲染 | 131 |
| 跳过（已存在且哈希一致） | 0 |
| 失败页 | 0 |
| 文件冲突 | 0 |
| 文档级失败 | 0 |
| PNG 总大小 | 240.7 MB |
| 像素尺寸分布 | {'10200x6600': 89, '5100x3300': 42} |
| 图幅分布 | {'17.0x11.0in': 42, '34.0x22.0in': 89} |
| 批次墙钟耗时 | 1555 s |
| 逐页渲染耗时合计 | 1448.4 s |

## 逐文档

| 文档 | pdfinfo | PyMuPDF | 渲染 | 完整 | failed | warning | pass | 耗时 s | PNG MB |
|---|---|---|---|---|---|---|---|---|---|
| 70101-70183.pdf | 18 | 18 | 18 | True | 0 | 0 | 18 | 224.2 | 37.5 |
| 70301-70311.pdf | 4 | 4 | 4 | True | 0 | 0 | 4 | 53.2 | 7.5 |
| 71001-71070.pdf | 27 | 27 | 27 | True | 0 | 0 | 27 | 425.7 | 63.7 |
| 72901-72933.pdf | 9 | 9 | 9 | True | 0 | 0 | 9 | 106.2 | 17.4 |
| 73001-73087.pdf | 31 | 31 | 31 | True | 0 | 0 | 31 | 407.4 | 69.3 |
| D&OM-25.pdf | 8 | 8 | 8 | True | 0 | 8 | 0 | 51.3 | 15.0 |
| DET4700_All.pdf | 21 | 21 | 21 | True | 0 | 0 | 21 | 95.7 | 15.8 |
| RCD-25.pdf | 2 | 2 | 2 | True | 0 | 2 | 0 | 15.2 | 2.9 |
| SMD.pdf | 6 | 6 | 6 | True | 0 | 6 | 0 | 43.0 | 7.1 |
| TSR.pdf | 5 | 5 | 5 | True | 0 | 5 | 0 | 26.4 | 4.3 |

## 自动 QA（automated_integrity_check）

| 项 | 值 |
|---|---|
| pass / warning / failed | {'failed': 0, 'pass': 110, 'warning': 21} |
| blank suspected | 0 |
| crop risk suspected | 21 |
| severe blur suspected | 0 |
| document size outlier | 0 |
| reason code 分布 | {'crop_risk_suspected': 21} |
| QA 前后 PNG 哈希一致 | True |
| 原始 PDF 哈希全部一致 | True |

阈值：uncalibrated (configs/cad_render_qa.yaml)。所有标记均为 suspected，不是确认。

## 重复

- 文件级精确重复组：0
- 页面近似重复（dHash ≤ 6，`uncalibrated`）：状态 `computed`，候选组 7，候选对 13（跨文档 0）。仅为人工审核候选。

## 人工抽查（human_visual_review）

- status：**pending**　human_verified：**False**
- 抽查页数：6

| # | 文档 | 页 | 自动 QA | reason codes | 选入原因 |
|---|---|---|---|---|---|
| 1 | D&OM-25.pdf | 1 | warning | crop_risk_suspected | reason code: crop_risk_suspected；sheet size: 17.0x11.0in |
| 2 | 70101-70183.pdf | 1 | pass | — | sheet size: 34.0x22.0in；fill to minimum: document first page |
| 3 | 73001-73087.pdf | 1 | pass | — | largest document (73001-73087.pdf): first page |
| 4 | 73001-73087.pdf | 16 | pass | — | largest document (73001-73087.pdf): middle page |
| 5 | 73001-73087.pdf | 31 | pass | — | largest document (73001-73087.pdf): last page |
| 6 | 70301-70311.pdf | 1 | pass | — | fill to minimum: document first page |

OCR 调用 0 · VLM 调用 0 · API 费用 $0

## 人工审核（v2 新增）

- status：**sample_reviewed_supplement_pending**
- 人工看过：6 / 131 页；通过 6，不通过 0
- 裁切报警：insufficient evidence: only ['D&OM-25.pdf'] had a crop-warning page reviewed. The result cannot be generalised to ['RCD-25.pdf', 'SMD.pdf', 'TSR.pdf']; their warnings stay unreviewed.
- human_verified（语料级）：**False**。Only these 6 page(s) were looked at by a person. The other 125 page(s) are covered by automated integrity checks only.

> 上方 human_visual_review 段落是 v1 当时的状态，保留原文未改；以本节为准。
