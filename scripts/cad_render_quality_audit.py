"""Automated render QA + a <=10 page human sample pack.

    python scripts/cad_render_quality_audit.py \\
        --manifest data/cad_public_corpus/manifests/source_manifest_unreviewed.jsonl

Two things are kept strictly apart in every output:

  automated_integrity_check  what this script measured
  human_visual_review        what a person has confirmed — `pending` until the
                             sample manifest's human fields are filled in; an
                             unfilled field is never counted as a pass

If Poppler (pdfinfo) or the rendered pages are unavailable the run is BLOCKED:
it says so in the report and exits 3. It does not render anything itself and
does not fall back to another library.

Reads PDFs and PNGs; writes only to outputs/. No OCR, no VLM, no network.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cad_corpus.corpus import (  # noqa: E402
    DUPLICATE_CLUSTERS, PAGE_MANIFEST, RendererUnavailable, load_config, read_jsonl,
    sha256_file,
)
from src.cad_corpus.render_qa import (  # noqa: E402
    FAILED, PASS, R_BLANK, R_BLUR, R_CROP, WARNING, audit, load_qa_config,
    make_thumbnail, pdfinfo_boxes, select_sample,
)

PACK = ROOT / "outputs" / "cad_render_qa_pack"
REPORT_JSON = ROOT / "outputs" / "cad_render_quality_report.json"
REPORT_MD = ROOT / "outputs" / "cad_render_quality_report.md"

HUMAN_QUESTIONS = [
    ("text_not_obviously_blurred", "文字是否明显发虚？（不发虚 = 通过）"),
    ("thin_lines_not_obviously_missing", "细线是否明显缺失？（未缺失 = 通过）"),
    ("orientation_correct", "页面方向是否正确？"),
    ("title_block_and_edges_not_cropped", "标题栏或边缘内容是否被裁切？（未裁切 = 通过）"),
    ("page_matches_pdf", "页面是否与 PDF 对应？"),
]


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def source_hash_check(sources: Sequence[Dict], repo_root: Path) -> Dict:
    mismatched = [s["original_path"] for s in sources
                  if sha256_file(repo_root / s["original_path"]) != s["source_sha256"]]
    return {"checked": len(sources), "all_match": not mismatched,
            "mismatched": mismatched}


def build_report(*, status: str, sources, pages, documents, sample, clusters,
                 hash_check, block_reason: Optional[str] = None) -> Dict:
    statuses = Counter(p["qa_status"] for p in pages)
    real_sample = [s for s in sample if s.get("page_id")]
    return {
        "status": status,
        "block_reason": block_reason,
        "automated_integrity_check": {
            "pdf_count": sum(1 for s in sources if s["source_class"] == "pdf"),
            "pdfinfo_total_pages": (sum(d["pdfinfo_page_count"] for d in documents)
                                    if documents else None),
            "provisional_total_pages": sum(s.get("page_count") or 0 for s in sources),
            "pages_rendered_and_readable": sum(1 for p in pages if p.get("image_readable")),
            "pages_failed": statuses.get(FAILED, 0),
            "documents_complete": sum(1 for d in documents if d["document_complete"]),
            "documents": documents,
            "sheet_size_distribution": dict(sorted(Counter(
                p["sheet"] for p in pages if p.get("sheet")).items())),
            "qa_status_counts": {s: statuses.get(s, 0) for s in (PASS, WARNING, FAILED)},
            "blank_suspected": sum(1 for p in pages if R_BLANK in p["reason_codes"]),
            "crop_risk_suspected": sum(1 for p in pages if R_CROP in p["reason_codes"]),
            "severe_blur_suspected": sum(1 for p in pages if R_BLUR in p["reason_codes"]),
            "reason_code_counts": dict(sorted(Counter(
                c for p in pages for c in p["reason_codes"]).items())),
            "thresholds": "uncalibrated (configs/cad_render_qa.yaml)",
            "near_duplicates_computed": clusters.get("status") == "computed",
            "near_duplicate_status": clusters.get("status"),
            "original_pdf_hashes": hash_check,
            "png_unchanged_by_qa": all(p.get("png_unchanged_by_qa", True) for p in pages),
            "what_this_cannot_confirm": [
                "thin line breaks", "title block cropped", "text legibility",
                "page content matches the PDF"],
        },
        "human_visual_review": {
            "status": "pending",
            "human_verified": False,
            "sample_pages": len(real_sample),
            "note": ("No person has reviewed the sample. Unfilled answers are "
                     "not passes."),
        },
        "ocr_calls": 0,
        "vlm_calls": 0,
        "network_calls": 0,
    }


def render_report_md(report: Dict) -> str:
    a, h = report["automated_integrity_check"], report["human_visual_review"]
    lines = ["# CAD 渲染质量报告", "",
             f"- 状态：**`{report['status']}`**"]
    if report["block_reason"]:
        lines.append(f"- 阻断原因：{report['block_reason']}")
    lines += [
        "", "## automated_integrity_check（机器测的）", "",
        "| 项 | 值 |", "|---|---|",
        f"| PDF 数 | {a['pdf_count']} |",
        f"| pdfinfo 总页数 | {a['pdfinfo_total_pages'] if a['pdfinfo_total_pages'] is not None else '未取得（pdfinfo 不可用）'} |",
        f"| 临时页数（PyMuPDF 元数据） | {a['provisional_total_pages']} |",
        f"| 成功渲染且可解码 | {a['pages_rendered_and_readable']} |",
        f"| failed 页 | {a['pages_failed']} |",
        f"| 完整文档 | {a['documents_complete']} / {a['pdf_count']} |",
        f"| pass / warning / failed | {a['qa_status_counts']} |",
        f"| 空白疑似 | {a['blank_suspected']} |",
        f"| 裁切风险疑似 | {a['crop_risk_suspected']} |",
        f"| 严重模糊疑似 | {a['severe_blur_suspected']} |",
        f"| 图幅分布 | {a['sheet_size_distribution'] or '—'} |",
        f"| 近似重复已计算 | {a['near_duplicates_computed']}（`{a['near_duplicate_status']}`） |",
        f"| 原始 PDF 哈希全部一致 | {a['original_pdf_hashes']['all_match']} |",
        f"| QA 未修改 PNG | {a['png_unchanged_by_qa']} |", "",
        f"阈值：{a['thresholds']}。机器**无法确认**：{'、'.join(a['what_this_cannot_confirm'])}。", "",
        "## human_visual_review（人看的）", "",
        f"- 状态：**{h['status']}**　human_verified：**{h['human_verified']}**",
        f"- 抽查页数：{h['sample_pages']}",
        f"- {h['note']}", "",
        f"OCR 调用 {report['ocr_calls']} · VLM 调用 {report['vlm_calls']} · "
        f"网络调用 {report['network_calls']}", ""]
    return "\n".join(lines)


def render_pack_html(report: Dict, sample: Sequence[Dict], pages_by_id: Dict,
                     pack: Path, repo_root: Path) -> str:
    esc = html.escape
    head = f"""<!doctype html><meta charset="utf-8"><title>渲染抽查包</title>
<style>
body{{font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;margin:24px;max-width:1100px}}
section{{border-top:2px solid #ddd;padding-top:14px;margin-top:26px}}
img{{max-width:100%;border:1px solid #bbb}}
table{{border-collapse:collapse;margin:6px 0}}td,th{{border:1px solid #ccc;padding:3px 8px}}
.blocked{{background:#fff3f3;border-left:4px solid #c00;padding:10px 14px}}
.note{{background:#fffbe6;border-left:4px solid #d9a400;padding:8px 12px}}
.q label{{margin-right:14px}}
</style>
<h1>CAD 渲染抽查包</h1>
<p class="note">这是<b>人工审核包</b>，不是审核结果。下面的选项默认全部未选；
未选的一律<b>不算通过</b>。报告中 <code>human_verified</code> 保持 false，
直到你把结论写进 <code>qa_sample_manifest.jsonl</code>。</p>"""
    if report["status"] != "audited":
        return head + f"""
<p class="blocked"><b>本轮没有可抽查的页面。</b><br>状态：<code>{esc(report['status'])}</code><br>
{esc(report['block_reason'] or '')}</p>
<p>安装 Poppler 后依次运行：</p>
<pre>python scripts/render_cad_pdf.py --manifest data/cad_public_corpus/manifests/source_manifest_unreviewed.jsonl --dpi 300
python scripts/cad_render_quality_audit.py --manifest data/cad_public_corpus/manifests/source_manifest_unreviewed.jsonl</pre>
"""
    body = [head, f"<p>共抽 <b>{len(sample)}</b> 页。抽样规则见 "
            "<code>qa_sample_manifest.jsonl</code> 中每页的 <code>selection_reasons</code>。</p>"]
    for item in sample:
        page = pages_by_id[item["page_id"]]
        thumb = f"thumbs/{item['page_id'].replace(':', '_')}.jpg"
        original = Path(os.path.relpath(repo_root / page["rendered_path"], pack)).as_posix() \
            if page.get("rendered_path") else None
        questions = "".join(
            f'<div class="q">{i}. {esc(text)} '
            f'<label><input type="radio" name="{esc(item["page_id"])}-{key}"> 通过</label>'
            f'<label><input type="radio" name="{esc(item["page_id"])}-{key}"> 不通过</label>'
            f'<label><input type="radio" name="{esc(item["page_id"])}-{key}"> 不确定</label></div>'
            for i, (key, text) in enumerate(HUMAN_QUESTIONS, start=1))
        body.append(f"""
<section>
<h2>#{item['rank']} {esc(page['file_name'])} · 第 {page['page_no']} 页</h2>
<table>
<tr><th>PDF SHA256</th><td><code>{esc(page['source_sha256_prefix'])}…</code></td>
    <th>图幅</th><td>{esc(str(page.get('sheet')))}</td></tr>
<tr><th>PNG 尺寸</th><td>{page.get('width')}×{page.get('height')}</td>
    <th>期望尺寸 @{page['dpi']} DPI</th><td>{page.get('expected_width')}×{page.get('expected_height')}</td></tr>
<tr><th>自动 QA</th><td><b>{esc(page['qa_status'])}</b></td>
    <th>reason codes</th><td>{esc(', '.join(page['reason_codes']) or '—')}</td></tr>
<tr><th>选入原因</th><td colspan="3">{esc('；'.join(item['selection_reasons']))}</td></tr>
</table>
<a href="{esc(original or '#')}" target="_blank"><img src="{esc(thumb)}" alt="缩略图"></a>
<p>点击缩略图打开原始 300 DPI PNG。</p>
{questions}
</section>""")
    return "".join(body)


def run(manifest: Path, *, pack: Path = PACK, report_json: Path = REPORT_JSON,
        report_md: Path = REPORT_MD, repo_root: Path = ROOT,
        boxes_for: Optional[Callable] = None) -> Dict:
    cfg = load_config()
    qa = load_qa_config()
    sources = read_jsonl(manifest)
    manifests = manifest.parent
    pages_in = read_jsonl(manifests / PAGE_MANIFEST)
    clusters_path = manifests / DUPLICATE_CLUSTERS
    clusters = json.loads(clusters_path.read_text(encoding="utf-8")) \
        if clusters_path.exists() else {}
    hash_check = source_hash_check(sources, repo_root)

    status, block_reason = "audited", None
    pages, documents, sample = [], [], []
    if boxes_for is None and not shutil.which("pdfinfo"):
        status = "blocked_poppler_unavailable"
        block_reason = ("Poppler (pdftoppm/pdfinfo) is not installed; no page was "
                        "rendered and no other renderer was substituted.")
    elif not pages_in:
        status = "blocked_no_rendered_pages"
        block_reason = "page manifest is empty; run scripts/render_cad_pdf.py first."
    else:
        pages, documents = audit(sources, pages_in, qa, dpi=cfg.render["dpi"],
                                 max_pixels=cfg.render["max_image_pixels"],
                                 repo_root=repo_root,
                                 boxes_for=boxes_for or pdfinfo_boxes)
        sample = select_sample(pages, documents, qa)

    report = build_report(status=status, sources=sources, pages=pages,
                          documents=documents, sample=sample, clusters=clusters,
                          hash_check=hash_check, block_reason=block_reason)
    report["pages"] = pages

    pack.mkdir(parents=True, exist_ok=True)
    real_sample = [s for s in sample if s.get("page_id")]
    pages_by_id = {p["page_id"]: p for p in pages}
    for item in real_sample:
        page = pages_by_id[item["page_id"]]
        if page.get("image_readable"):
            make_thumbnail(repo_root / page["rendered_path"],
                           pack / "thumbs" / f"{item['page_id'].replace(':', '_')}.jpg",
                           qa, cfg.render["max_image_pixels"])
    (pack / "qa_sample_manifest.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n" for s in sample),
        encoding="utf-8")
    (pack / "index.html").write_text(
        render_pack_html(report, real_sample, pages_by_id, pack, repo_root),
        encoding="utf-8")
    report_json.write_text(_dumps(report), encoding="utf-8")
    report_md.write_text(render_report_md(report), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2
    try:
        report = run(manifest)
    except RendererUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    a = report["automated_integrity_check"]
    print(f"status={report['status']}")
    if report["block_reason"]:
        print(report["block_reason"])
    print(f"readable pages {a['pages_rendered_and_readable']}  "
          f"qa {a['qa_status_counts']}  "
          f"sample {report['human_visual_review']['sample_pages']}  "
          f"human_verified={report['human_visual_review']['human_verified']}")
    print(f"-> {REPORT_JSON}\n-> {PACK / 'index.html'}")
    return 3 if report["status"].startswith("blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
