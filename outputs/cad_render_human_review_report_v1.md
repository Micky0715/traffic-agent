# CAD 渲染人工审核报告 v1

- 人工审核文件：`data/cad_public_corpus/manifests/cad_render_human_review_v1.jsonl`（sha256 `b23f20ad7474b829…`，未修改）
- 抽样页：6　已人工审核：**6**　待审：0
- 人工通过：**6**　人工不通过：**0**

## 人工确认的问题

| 问题 | 页数 |
|---|---|
| text_blur_confirmed | 0 |
| thin_line_loss_confirmed | 0 |
| orientation_error_confirmed | 0 |
| crop_confirmed | 0 |
| pdf_mismatch_confirmed | 0 |

## 裁切报警（crop_risk_suspected）

- 自动报警总数：**21**
- 人工确认真实裁切：**0**
- 人工确认误报：**1**
- 尚未人工验证：**20**

> **insufficient evidence: only ['D&OM-25.pdf'] had a crop-warning page reviewed. The result cannot be generalised to ['RCD-25.pdf', 'SMD.pdf', 'TSR.pdf']; their warnings stay unreviewed.**

| 文档 | 报警页 | 人工看过 |
|---|---|---|
| D&OM-25.pdf | 8 | 1 |
| RCD-25.pdf | 2 | 0 |
| SMD.pdf | 6 | 0 |
| TSR.pdf | 5 | 0 |

## 每页人工结论与自动 QA 对照

| 文档 | 页 | 人工结论 | 自动 QA | reason codes | 对照 | 采集方式 |
|---|---|---|---|---|---|---|
| 70101-70183.pdf | 1 | **pass** | pass | — | agree | chat_statement |
| 70301-70311.pdf | 1 | **pass** | pass | — | agree | chat_statement |
| 73001-73087.pdf | 1 | **pass** | pass | — | agree | chat_statement |
| 73001-73087.pdf | 16 | **pass** | pass | — | agree | chat_statement |
| 73001-73087.pdf | 31 | **pass** | pass | — | agree | chat_statement |
| D&OM-25.pdf | 1 | **pass** | warning | crop_risk_suspected | auto_warning_human_pass | chat_statement |

## human_verified_scope

> Only these 6 page(s) were looked at by a person. The other 125 page(s) are covered by automated integrity checks only.

阈值未调整。本报告不包含 OCR、VLM 或任何模型调用。
