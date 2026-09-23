"""Join rendering runtime facts onto the automated QA report (read-only).

    python scripts/cad_render_runtime_report.py \\
        --qa-report outputs/cad_render_quality_report_v1.json \\
        --batch-start-epoch <s> --batch-end-epoch <s>

The QA tool measures pages; it does not know how the batch ran. This script
adds what only the run artefacts can say — Poppler path and version, pages
rendered / skipped / failed / in conflict, PNG bytes on disk, per-document and
batch timing, duplicate candidates — and writes the combined report.

It renders nothing, audits nothing, calls no model and changes no threshold.
Source metadata (source_manifest_unreviewed.jsonl) is only read: runtime status
lives here, beside it, never written into it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "cad_public_corpus"
MANIFESTS = CORPUS / "manifests"
CONFLICT_REASONS = {"existing_png_hash_or_config_mismatch",
                    "unrecorded_existing_png_differs"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tool(name: str) -> dict:
    exe = shutil.which(name)
    if not exe:
        return {"path": None, "version": None}
    out = subprocess.run([exe, "-v"], capture_output=True, text=True)
    line = next((l for l in (out.stderr or out.stdout).splitlines()
                 if "version" in l.lower()), None)
    return {"path": exe, "version": line}


def build(qa: dict, start: int, end: int) -> dict:
    sources = jsonl(MANIFESTS / "source_manifest_unreviewed.jsonl")
    pages = jsonl(MANIFESTS / "page_manifest_unreviewed.jsonl")
    failures = jsonl(MANIFESTS / "conversion_failures.jsonl")
    clusters = json.loads((MANIFESTS / "duplicate_clusters.json").read_text(encoding="utf-8"))
    a = qa["automated_integrity_check"]
    docs = {d["source_id"]: d for d in a.get("documents", [])}

    png_bytes, skipped, rendered_now = 0, 0, 0
    per_doc = defaultdict(lambda: {"pages": 0, "render_ms": 0.0, "png_bytes": 0})
    for page in pages:
        path = ROOT / page["rendered_path"]
        size = path.stat().st_size if path.exists() else 0
        png_bytes += size
        # A page whose PNG predates this batch was verified and skipped.
        if path.exists() and path.stat().st_mtime < start:
            skipped += 1
        else:
            rendered_now += 1
        entry = per_doc[page["source_id"]]
        entry["pages"] += 1
        entry["render_ms"] += page.get("render_ms") or 0
        entry["png_bytes"] += size

    originals_ok = all(sha(ROOT / s["original_path"]) == s["source_sha256"] for s in sources)
    near = clusters.get("near_duplicate_page_candidates", {})
    real_sample = [s for s in qa.get("_sample", []) if s.get("page_id")]

    per_document = []
    for s in sorted(sources, key=lambda r: r["original_path"]):
        d = docs.get(s["source_id"], {})
        t = per_doc.get(s["source_id"], {"pages": 0, "render_ms": 0.0, "png_bytes": 0})
        per_document.append({
            "source_id": s["source_id"], "file_name": s["file_name"],
            "pdfinfo_page_count": d.get("pdfinfo_page_count"),
            "provisional_page_count": s.get("page_count"),
            "page_count_consistent": d.get("page_count_consistent"),
            "rendered_pages": t["pages"],
            "document_complete": d.get("document_complete"),
            "pages_failed": d.get("pages_failed"), "pages_warning": d.get("pages_warning"),
            "pages_pass": d.get("pages_pass"),
            "render_seconds": round(t["render_ms"] / 1000, 1),
            "png_mb": round(t["png_bytes"] / 1e6, 1),
            # Source metadata stays as the manifest has it: unreviewed, empty.
            "agency": s.get("agency"), "source_url": s.get("source_url"),
            "provenance_status": s.get("provenance_status"),
        })

    runtime = {
        "poppler": {"pdftoppm": tool("pdftoppm"), "pdfinfo": tool("pdfinfo")},
        "render_command": "pdftoppm -r 300 -png -f N -l N -singlefile <pdf> <out>",
        "dpi": pages[0]["dpi"] if pages else None,
        "pdf_count": sum(1 for s in sources if s["source_class"] == "pdf"),
        "pdfinfo_total_pages": a.get("pdfinfo_total_pages"),
        "pymupdf_provisional_total_pages": sum(s.get("page_count") or 0 for s in sources),
        "page_counts_consistent": all(d.get("page_count_consistent") for d in docs.values())
        if docs else None,
        "pages_in_manifest": len(pages),
        "pages_rendered_this_batch": rendered_now,
        "pages_skipped_existing_verified": skipped,
        "pages_failed": sum(1 for f in failures if f.get("page_no") is not None
                            and f["reason"] not in CONFLICT_REASONS),
        "file_conflicts": sum(1 for f in failures if f["reason"] in CONFLICT_REASONS),
        "document_level_failures": [f for f in failures if f.get("page_no") is None],
        "failure_records": failures,
        "png_total_bytes": png_bytes,
        "png_total_mb": round(png_bytes / 1e6, 1),
        "pixel_size_distribution": dict(sorted(Counter(
            f"{p['width']}x{p['height']}" for p in pages).items())),
        "batch_wall_seconds": end - start,
        "sum_page_render_seconds": round(sum((p.get("render_ms") or 0) for p in pages) / 1000, 1),
        "per_document": per_document,
        "exact_duplicate_clusters": len(clusters.get("exact_file_duplicates", [])),
        "near_duplicate": {
            "status": clusters.get("status"),
            "threshold_status": near.get("threshold_status"),
            "max_hamming_distance": near.get("max_hamming_distance"),
            "candidate_clusters": len(near.get("clusters", [])),
            "candidate_pairs": len(near.get("pairs", [])),
            "cross_document_pairs": sum(1 for p in near.get("pairs", []) if p.get("cross_source")),
            "clusters": near.get("clusters", []),
            "pairs": near.get("pairs", []),
            "note": "review candidates only; nothing merged, moved or deleted",
        },
        "original_pdf_hashes_all_match": originals_ok,
        "png_hashes_unchanged_by_qa": a.get("png_unchanged_by_qa"),
        "ocr_calls": 0, "vlm_calls": 0, "api_cost_usd": 0,
    }
    out = dict(qa)
    out.pop("_sample", None)
    out["runtime"] = runtime
    out["sample_pages"] = [{k: s[k] for k in ("rank", "page_id", "file_name", "page_no",
                                              "qa_status", "reason_codes",
                                              "selection_reasons")}
                           for s in real_sample]
    return out


def render_md(r: dict) -> str:
    rt, a, h = r["runtime"], r["automated_integrity_check"], r["human_visual_review"]
    nd = rt["near_duplicate"]
    L = ["# CAD 渲染质量报告 v1", "",
         f"- 状态：`{r['status']}`",
         f"- pdftoppm：`{rt['poppler']['pdftoppm']['path']}` — {rt['poppler']['pdftoppm']['version']}",
         f"- pdfinfo：`{rt['poppler']['pdfinfo']['path']}` — {rt['poppler']['pdfinfo']['version']}",
         f"- 渲染命令：`{rt['render_command']}`，DPI {rt['dpi']}", "",
         "## 渲染", "", "| 项 | 值 |", "|---|---|",
         f"| PDF 数 | {rt['pdf_count']} |",
         f"| pdfinfo 总页数 | {rt['pdfinfo_total_pages']} |",
         f"| PyMuPDF 临时页数 | {rt['pymupdf_provisional_total_pages']} |",
         f"| 两者一致 | {rt['page_counts_consistent']} |",
         f"| 本批渲染 | {rt['pages_rendered_this_batch']} |",
         f"| 跳过（已存在且哈希一致） | {rt['pages_skipped_existing_verified']} |",
         f"| 失败页 | {rt['pages_failed']} |",
         f"| 文件冲突 | {rt['file_conflicts']} |",
         f"| 文档级失败 | {len(rt['document_level_failures'])} |",
         f"| PNG 总大小 | {rt['png_total_mb']} MB |",
         f"| 像素尺寸分布 | {rt['pixel_size_distribution']} |",
         f"| 图幅分布 | {a['sheet_size_distribution']} |",
         f"| 批次墙钟耗时 | {rt['batch_wall_seconds']} s |",
         f"| 逐页渲染耗时合计 | {rt['sum_page_render_seconds']} s |", "",
         "## 逐文档", "",
         "| 文档 | pdfinfo | PyMuPDF | 渲染 | 完整 | failed | warning | pass | 耗时 s | PNG MB |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for d in rt["per_document"]:
        L.append(f"| {d['file_name']} | {d['pdfinfo_page_count']} | {d['provisional_page_count']} | "
                 f"{d['rendered_pages']} | {d['document_complete']} | {d['pages_failed']} | "
                 f"{d['pages_warning']} | {d['pages_pass']} | {d['render_seconds']} | {d['png_mb']} |")
    L += ["", "## 自动 QA（automated_integrity_check）", "", "| 项 | 值 |", "|---|---|",
          f"| pass / warning / failed | {a['qa_status_counts']} |",
          f"| blank suspected | {a['blank_suspected']} |",
          f"| crop risk suspected | {a['crop_risk_suspected']} |",
          f"| severe blur suspected | {a['severe_blur_suspected']} |",
          f"| document size outlier | {a['reason_code_counts'].get('document_size_outlier', 0)} |",
          f"| reason code 分布 | {a['reason_code_counts']} |",
          f"| QA 前后 PNG 哈希一致 | {rt['png_hashes_unchanged_by_qa']} |",
          f"| 原始 PDF 哈希全部一致 | {rt['original_pdf_hashes_all_match']} |", "",
          f"阈值：{a['thresholds']}。所有标记均为 suspected，不是确认。", "",
          "## 重复", "",
          f"- 文件级精确重复组：{rt['exact_duplicate_clusters']}",
          f"- 页面近似重复（dHash ≤ {nd['max_hamming_distance']}，`{nd['threshold_status']}`）："
          f"状态 `{nd['status']}`，候选组 {nd['candidate_clusters']}，候选对 {nd['candidate_pairs']}"
          f"（跨文档 {nd['cross_document_pairs']}）。仅为人工审核候选。", "",
          "## 人工抽查（human_visual_review）", "",
          f"- status：**{h['status']}**　human_verified：**{h['human_verified']}**",
          f"- 抽查页数：{len(r['sample_pages'])}", "",
          "| # | 文档 | 页 | 自动 QA | reason codes | 选入原因 |", "|---|---|---|---|---|---|"]
    for s in r["sample_pages"]:
        L.append(f"| {s['rank']} | {s['file_name']} | {s['page_no']} | {s['qa_status']} | "
                 f"{', '.join(s['reason_codes']) or '—'} | {'；'.join(s['selection_reasons'])} |")
    L += ["", f"OCR 调用 {rt['ocr_calls']} · VLM 调用 {rt['vlm_calls']} · API 费用 ${rt['api_cost_usd']}", ""]
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qa-report", required=True)
    parser.add_argument("--batch-start-epoch", type=int, required=True)
    parser.add_argument("--batch-end-epoch", type=int, required=True)
    args = parser.parse_args()
    path = Path(args.qa_report)
    if not path.is_absolute():
        path = ROOT / path
    qa = json.loads(path.read_text(encoding="utf-8"))
    sample_path = ROOT / "outputs" / "cad_render_qa_pack" / "qa_sample_manifest.jsonl"
    qa["_sample"] = jsonl(sample_path)
    report = build(qa, args.batch_start_epoch, args.batch_end_epoch)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    path.with_suffix(".md").write_text(render_md(report), encoding="utf-8")
    rt = report["runtime"]
    print(f"rendered {rt['pages_rendered_this_batch']} skipped {rt['pages_skipped_existing_verified']} "
          f"failed {rt['pages_failed']} conflicts {rt['file_conflicts']} "
          f"png {rt['png_total_mb']} MB  wall {rt['batch_wall_seconds']} s")
    print(f"-> {path}\n-> {path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
