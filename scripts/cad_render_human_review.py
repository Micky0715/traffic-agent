"""Merge human render-review records with automated QA; build the supplement pack.

    python scripts/cad_render_human_review.py
    python scripts/cad_render_human_review.py --review <jsonl>

Reads the human review file and never writes it. Refuses to merge if any record
fails validation (missing field, unknown page, page hash mismatch, duplicate).
Writes only NEW versioned files:

  outputs/cad_render_human_review_report_v1.{json,md}
  outputs/cad_render_quality_report_v2.{json,md}     (v1 machine facts kept verbatim)
  outputs/cad_render_qa_pack_supplement_v1/          (only if some crop-warning
                                                      document has no reviewed page)

The supplement pack has an Export button: the first pack's answers existed only
in the browser and could not be saved, and that must not happen twice.

No OCR, no VLM, no threshold change, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cad_corpus.human_review import (  # noqa: E402
    CHECKS, CROP_CODE, HumanReviewError, merge, supplement_candidates, validate,
)

MANIFESTS = ROOT / "data" / "cad_public_corpus" / "manifests"
REVIEW = MANIFESTS / "cad_render_human_review_v1.jsonl"
PAGES = MANIFESTS / "page_manifest_unreviewed.jsonl"
SAMPLE = ROOT / "outputs" / "cad_render_qa_pack" / "qa_sample_manifest.jsonl"
QA_V1 = ROOT / "outputs" / "cad_render_quality_report_v1.json"
OUT = {
    "human_json": ROOT / "outputs" / "cad_render_human_review_report_v1.json",
    "human_md": ROOT / "outputs" / "cad_render_human_review_report_v1.md",
    "qa_v2_json": ROOT / "outputs" / "cad_render_quality_report_v2.json",
    "qa_v2_md": ROOT / "outputs" / "cad_render_quality_report_v2.md",
}
SUPPLEMENT = ROOT / "outputs" / "cad_render_qa_pack_supplement_v1"
PROTECTED = {QA_V1, QA_V1.with_suffix(".md"), REVIEW, SAMPLE}

QUESTIONS = {
    "text_not_obviously_blurred": "文字是否明显发虚？（不发虚 = 通过）",
    "thin_lines_not_obviously_missing": "细线是否明显缺失？（未缺失 = 通过）",
    "orientation_correct": "页面方向是否正确？",
    "title_block_and_edges_not_cropped": "标题栏或边缘内容是否被裁切？（未裁切 = 通过）",
    "page_matches_pdf": "PNG 是否与对应 PDF 页面一致？",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_new(path: Path, text: str, *, overwrite: bool) -> None:
    if path.resolve() in {p.resolve() for p in PROTECTED}:
        raise HumanReviewError(f"refusing to write protected file {path}")
    if path.exists() and path.read_text(encoding="utf-8") != text and not overwrite:
        raise HumanReviewError(f"{path} exists with different content; pass --overwrite "
                               f"to replace this (non-historical) output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def human_md(m: Dict, review_sha: str) -> str:
    scope = m["human_verified_scope"]
    L = ["# CAD 渲染人工审核报告 v1", "",
         f"- 人工审核文件：`{rel(REVIEW)}`（sha256 `{review_sha[:16]}…`，未修改）",
         f"- 抽样页：{len(m['selected_pages'])}　已人工审核：**{len(m['human_reviewed_pages'])}**"
         f"　待审：{len(m['pending_pages'])}",
         f"- 人工通过：**{len(m['pass_pages'])}**　人工不通过：**{len(m['failed_pages'])}**", "",
         "## 人工确认的问题", "", "| 问题 | 页数 |", "|---|---|"]
    for key in ("text_blur_confirmed", "thin_line_loss_confirmed",
                "orientation_error_confirmed", "crop_confirmed", "pdf_mismatch_confirmed"):
        L.append(f"| {key} | {m[key]} |")
    L += ["", "## 裁切报警（crop_risk_suspected）", "",
          f"- 自动报警总数：**{m['crop_warning_total']}**",
          f"- 人工确认真实裁切：**{m['crop_warning_true_positive']}**",
          f"- 人工确认误报：**{m['crop_warning_false_positive']}**",
          f"- 尚未人工验证：**{m['crop_warning_unreviewed']}**", "",
          f"> **{m['crop_conclusion']}**", "",
          "| 文档 | 报警页 | 人工看过 |", "|---|---|---|"]
    for doc, v in m["crop_warning_by_document"].items():
        L.append(f"| {doc} | {v['warnings']} | {v['reviewed']} |")
    L += ["", "## 每页人工结论与自动 QA 对照", "",
          "| 文档 | 页 | 人工结论 | 自动 QA | reason codes | 对照 | 采集方式 |",
          "|---|---|---|---|---|---|---|"]
    for p in m["per_page"]:
        L.append(f"| {p['file_name']} | {p['page_no']} | **{p['human_verdict']}** | "
                 f"{p['auto_qa_status']} | {', '.join(p['auto_reason_codes']) or '—'} | "
                 f"{p['comparison']} | {p['review_capture']} |")
    L += ["", "## human_verified_scope", "", f"> {scope['statement']}", "",
          "阈值未调整。本报告不包含 OCR、VLM 或任何模型调用。", ""]
    return "\n".join(L)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


# ---------------------------------------------------------------------------
# supplement pack (with export)
# ---------------------------------------------------------------------------

EXPORT_JS = r"""
<script>
const PAGES = __PAGES__;
function exportReview() {
  const annotator = document.getElementById('annotator').value.trim();
  if (!annotator) { alert('请先填写审核人'); return; }
  const now = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
  let unanswered = 0;
  const lines = PAGES.map(p => {
    const rec = {page_id: p.page_id, file_name: p.file_name, page_no: p.page_no,
                 page_sha256: p.page_sha256};
    for (const key of __CHECKS__) {
      const hit = document.querySelector(`input[name="${p.page_id}|${key}"]:checked`);
      rec[key] = hit ? hit.value : null;           // unanswered stays null, never pass
      if (!hit) unanswered++;
    }
    rec.annotator = annotator; rec.reviewed_at = now;
    rec.notes = document.getElementById('notes-' + p.page_id).value;
    rec.review_capture = 'html_export';
    return JSON.stringify(rec);
  });
  const text = lines.join('\n') + '\n';
  document.getElementById('out').value = text;
  const blob = new Blob([text], {type: 'application/x-ndjson'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cad_render_human_review_supplement_v1.jsonl';
  a.click();
  if (unanswered) alert(unanswered + ' 项未填写，已按 null 导出（不算通过）');
}
</script>"""


def supplement_html(pages: Sequence[Dict], pack: Path, reason: str) -> str:
    esc = html.escape
    payload = [{"page_id": p["page_id"], "file_name": p["file_name"],
                "page_no": p["page_no"], "page_sha256": p["rendered_sha256"]} for p in pages]
    parts = [f"""<!doctype html><meta charset="utf-8"><title>补充抽查包</title>
<style>body{{font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;margin:24px;max-width:1100px}}
section{{border-top:2px solid #ddd;padding-top:14px;margin-top:24px}}img{{max-width:100%;border:1px solid #bbb}}
.note{{background:#fffbe6;border-left:4px solid #d9a400;padding:8px 12px}}
button{{font-size:16px;padding:6px 16px}}textarea{{width:100%}}</style>
<h1>CAD 渲染补充抽查包 v1</h1>
<p class="note">{esc(reason)}<br>只针对裁切报警：请重点看第 4 项（标题栏或边缘是否被裁切），
但 5 项都请作答。<b>填完点页面底部"导出"按钮</b>，会下载
<code>cad_render_human_review_supplement_v1.jsonl</code>；未填写的项按 null 导出，不算通过。</p>"""]
    for page in pages:
        thumb = f"thumbs/{page['page_id'].replace(':', '_')}.jpg"
        original = Path(os.path.relpath(ROOT / page["rendered_path"], pack)).as_posix()
        radios = "".join(
            f"<div>{i}. {esc(text)} " + "".join(
                f'<label><input type="radio" name="{esc(page["page_id"])}|{key}" '
                f'value="{val}"> {label}</label> '
                for val, label in (("pass", "通过"), ("fail", "不通过"),
                                   ("uncertain", "不确定"))) + "</div>"
            for i, (key, text) in enumerate(QUESTIONS.items(), start=1))
        e = page["edge_content_ratio"]
        parts.append(f"""<section><h2>{esc(page['file_name'])} · 第 {page['page_no']} 页</h2>
<p>自动 QA：<b>{esc(page['qa_status'])}</b> · {esc(', '.join(page['reason_codes']))} ·
边缘非白比例 上 {e['top']:.2f} / 下 {e['bottom']:.2f} / 左 {e['left']:.2f} / 右 {e['right']:.2f}
· {page['width']}×{page['height']} @ {page['dpi']} DPI · PNG sha256 <code>{page['rendered_sha256'][:12]}…</code></p>
<a href="{esc(original)}" target="_blank"><img src="{esc(thumb)}"></a>
<p>点击缩略图打开原始 300 DPI PNG，重点看四条边。</p>{radios}
<p>备注：<textarea id="notes-{esc(page['page_id'])}" rows="2"></textarea></p></section>""")
    parts.append("""<section><p>审核人：<input id="annotator" size="20"></p>
<button onclick="exportReview()">导出</button>
<p>若下载被浏览器拦截，可复制下框内容保存为同名文件：</p>
<textarea id="out" rows="6" readonly></textarea></section>"""
                 + EXPORT_JS.replace("__PAGES__", json.dumps(payload, ensure_ascii=False))
                 .replace("__CHECKS__", json.dumps(list(CHECKS))))
    return "".join(parts)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(review: Path = REVIEW, *, overwrite: bool = False, repo_root: Path = ROOT,
        out: Optional[Dict[str, Path]] = None, supplement_dir: Path = SUPPLEMENT) -> Dict:
    out = out or OUT
    if not review.exists():
        raise HumanReviewError(f"human review file not found: {review}")
    guarded = {p: sha(p) for p in (review, SAMPLE, QA_V1, QA_V1.with_suffix(".md"))
               if p.exists()}

    records = jsonl(review)
    pages = {p["page_id"]: p for p in jsonl(PAGES)}
    problems = validate(records, pages, repo_root=repo_root)
    if problems:
        raise HumanReviewError("refusing to merge:\n  " + "\n  ".join(problems))

    qa_v1 = json.loads(QA_V1.read_text(encoding="utf-8"))
    sample_ids = [s["page_id"] for s in jsonl(SAMPLE) if s.get("page_id")]
    merged = merge(records, qa_v1["pages"], sample_ids)
    # QA rows carry the path, not the hash; the page manifest is the authority.
    extra = [{**p, "rendered_sha256": pages[p["page_id"]]["rendered_sha256"]}
             for p in supplement_candidates(qa_v1["pages"], merged)]
    review_sha = sha(review)

    human = {"source_review_file": rel(review), "source_review_sha256": review_sha,
             "review_capture": sorted({r.get("review_capture") for r in records}),
             "thresholds_changed": False, "ocr_calls": 0, "vlm_calls": 0, **merged,
             "supplement_pages": [p["page_id"] for p in extra]}
    write_new(out["human_json"], dumps(human), overwrite=overwrite)
    write_new(out["human_md"], human_md(merged, review_sha), overwrite=overwrite)

    # v2: every v1 key verbatim, plus the human layer beside it.
    v2 = dict(qa_v1)
    v2["human_visual_review_at_v1"] = qa_v1["human_visual_review"]
    v2["human_visual_review"] = {
        "status": ("sample_reviewed_supplement_pending" if extra else "sample_reviewed"),
        "human_verified": False,
        "human_verified_note": ("false at corpus level: only the pages in "
                                "human_verified_scope were looked at by a person"),
        "human_verified_scope": merged["human_verified_scope"],
        "pass_pages": len(merged["pass_pages"]),
        "failed_pages": len(merged["failed_pages"]),
        "crop_warning": {k: merged[k] for k in (
            "crop_warning_total", "crop_warning_true_positive",
            "crop_warning_false_positive", "crop_warning_unreviewed", "crop_conclusion")},
        "supplement_pages": [p["page_id"] for p in extra],
        "source_review_sha256": review_sha,
    }
    v2["derived_from"] = {"v1_report": rel(QA_V1), "v1_sha256": sha(QA_V1),
                          "v1_machine_facts_unchanged": True}
    write_new(out["qa_v2_json"], dumps(v2), overwrite=overwrite)
    v1_md = QA_V1.with_suffix(".md").read_text(encoding="utf-8")
    write_new(out["qa_v2_md"], v1_md.replace("# CAD 渲染质量报告 v1", "# CAD 渲染质量报告 v2", 1)
              + "\n## 人工审核（v2 新增）\n\n"
              + f"- status：**{v2['human_visual_review']['status']}**\n"
              + f"- 人工看过：{merged['human_verified_scope']['count']} / "
              + f"{merged['human_verified_scope']['of_total_pages']} 页；"
              + f"通过 {len(merged['pass_pages'])}，不通过 {len(merged['failed_pages'])}\n"
              + f"- 裁切报警：{merged['crop_conclusion']}\n"
              + f"- human_verified（语料级）：**False**。"
              + f"{merged['human_verified_scope']['statement']}\n"
              + "\n> 上方 human_visual_review 段落是 v1 当时的状态，保留原文未改；"
              + "以本节为准。\n", overwrite=overwrite)

    if extra:
        from src.cad_corpus.corpus import load_config
        from src.cad_corpus.render_qa import load_qa_config, make_thumbnail
        cfg, qa = load_config(), load_qa_config()
        for page in extra:
            make_thumbnail(repo_root / page["rendered_path"],
                           supplement_dir / "thumbs" / f"{page['page_id'].replace(':', '_')}.jpg",
                           qa, cfg.render["max_image_pixels"])
        write_new(supplement_dir / "index.html",
                  supplement_html(extra, supplement_dir, merged["crop_conclusion"]),
                  overwrite=overwrite)
        write_new(supplement_dir / "qa_sample_manifest.jsonl", "".join(
            json.dumps({"rank": i, "page_id": p["page_id"], "file_name": p["file_name"],
                        "page_no": p["page_no"], "page_sha256": p["rendered_sha256"],
                        "qa_status": p["qa_status"], "reason_codes": p["reason_codes"],
                        "selection_reasons": [f"{CROP_CODE}: document had no reviewed "
                                              f"warning page"],
                        "human_review_status": "pending"},
                       ensure_ascii=False, sort_keys=True) + "\n"
            for i, p in enumerate(extra, start=1)), overwrite=overwrite)

    changed = [rel(p) for p, h in guarded.items() if sha(p) != h]
    if changed:
        raise HumanReviewError(f"protected inputs changed during the run: {changed}")
    return {"human": human, "v2": v2, "supplement": extra}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", default=str(REVIEW))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    review = Path(args.review)
    if not review.is_absolute():
        review = ROOT / review
    try:
        result = run(review, overwrite=args.overwrite)
    except HumanReviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    h = result["human"]
    print(f"reviewed {len(h['human_reviewed_pages'])}  pass {len(h['pass_pages'])}  "
          f"fail {len(h['failed_pages'])}  pending {len(h['pending_pages'])}")
    print(f"crop warnings {h['crop_warning_total']}: true {h['crop_warning_true_positive']}  "
          f"false-positive {h['crop_warning_false_positive']}  "
          f"unreviewed {h['crop_warning_unreviewed']}")
    print(h["crop_conclusion"])
    print(f"supplement pages: {[p['file_name'] + ' p' + str(p['page_no']) for p in result['supplement']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
