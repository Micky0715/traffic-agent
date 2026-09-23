# CAD 渲染质量报告

- 状态：**`blocked_poppler_unavailable`**
- 阻断原因：Poppler (pdftoppm/pdfinfo) is not installed; no page was rendered and no other renderer was substituted.

## automated_integrity_check（机器测的）

| 项 | 值 |
|---|---|
| PDF 数 | 10 |
| pdfinfo 总页数 | 未取得（pdfinfo 不可用） |
| 临时页数（PyMuPDF 元数据） | 131 |
| 成功渲染且可解码 | 0 |
| failed 页 | 0 |
| 完整文档 | 0 / 10 |
| pass / warning / failed | {'pass': 0, 'warning': 0, 'failed': 0} |
| 空白疑似 | 0 |
| 裁切风险疑似 | 0 |
| 严重模糊疑似 | 0 |
| 图幅分布 | — |
| 近似重复已计算 | False（`near_duplicate_not_computed_no_rendered_pages`） |
| 原始 PDF 哈希全部一致 | True |
| QA 未修改 PNG | True |

阈值：uncalibrated (configs/cad_render_qa.yaml)。机器**无法确认**：thin line breaks、title block cropped、text legibility、page content matches the PDF。

## human_visual_review（人看的）

- 状态：**pending**　human_verified：**False**
- 抽查页数：0
- No person has reviewed the sample. Unfilled answers are not passes.

OCR 调用 0 · VLM 调用 0 · 网络调用 0
