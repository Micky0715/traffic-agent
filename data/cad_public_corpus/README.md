# CAD 公开语料（Phase 0）

本目录由 `scripts/cad_corpus_inventory.py` 与 `scripts/render_cad_pdf.py` 生成。

- **原始文件不在这里。** 它们留在用户放置的原位置，manifest 用 `original_path`
  引用，不复制、不移动、不重编码。`raw_pdf/ raw_cad/ raw_raster/` 保留为归档目录，
  Phase 0 没有向其中复制任何文件，以免大文件无意义地存两份。
- `rendered_png_300dpi/`：只由 `pdftoppm` 以 300 DPI 渲染，不使用其他渲染器。
- `manifests/`：所有 `*_unreviewed` 文件中的人工字段（agency、source_url、类别、
  是否含标题栏、split）一律为空或 `unreviewed`，等待人工审核。
- 近似重复只是**候选**，阈值未校准；没有合并或删除任何文件。
- 原生 CAD（DWG / DXF / DGN）只登记，不渲染、不解析。

来源类别：本目录内容为**用户下载的公开图纸**，不是企业数据、不是自撰数据、
不是 OCR fixture，也不是真实推理结果。
