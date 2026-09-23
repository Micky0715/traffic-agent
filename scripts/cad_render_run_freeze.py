"""Freeze / verify the first real render + QA run of the public CAD corpus.

Writes outputs/cad_corpus_render_run_freeze_manifest_v1.json.

This run INTENTIONALLY changed things two earlier manifests froze while no page
existed yet:

  outputs/cad_corpus_phase0_freeze_manifest.json      (0 rendered pages, renderer unavailable)
  outputs/cad_corpus_render_qa_freeze_manifest_v1.json (blocked QA pack)

Neither is edited, and neither is claimed to be OK: both now report FREEZE
VIOLATION, which is the honest record of "the corpus was rendered after they
were written". This manifest lists, for each, exactly which frozen files moved
and why, and refuses to write if anything moved that is not explained.

    python scripts/cad_render_run_freeze.py --write
    python scripts/cad_render_run_freeze.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "cad_corpus_render_run_freeze_manifest_v1.json"
CORPUS = ROOT / "data" / "cad_public_corpus"
MANIFESTS = CORPUS / "manifests"
PRIOR = {
    "cad_corpus_phase0": ROOT / "outputs" / "cad_corpus_phase0_freeze_manifest.json",
    "cad_render_qa_v1": ROOT / "outputs" / "cad_corpus_render_qa_freeze_manifest_v1.json",
}
# Files the render + QA run was EXPECTED to create or rewrite, and why.
EXPECTED_CHANGES = {
    "data/cad_public_corpus/manifests/conversion_failures.jsonl":
        "was 10 x renderer_unavailable; now empty after a successful render",
    "data/cad_public_corpus/manifests/duplicate_clusters.json":
        "near-duplicate dHash candidates now computed over rendered pages",
    "data/cad_public_corpus/manifests/page_manifest_unreviewed.jsonl":
        "created: 131 rendered pages",
    "data/cad_public_corpus/splits/split_candidates_unreviewed.jsonl":
        "recomputed with near-duplicate groups (rewritten by the render step)",
    "outputs/cad_render_qa_pack/index.html": "refreshed: 6 sampled pages",
    "outputs/cad_render_qa_pack/qa_sample_manifest.jsonl": "refreshed: 6 sampled pages",
}
EXPECTED_PREFIXES = {
    "data/cad_public_corpus/rendered_png_300dpi/": "131 PNGs rendered by pdftoppm 25.07.0",
    "outputs/cad_render_qa_pack/thumbs/": "thumbnails for the sampled pages",
}

GROUPS = {
    "runtime_code": ["src/cad_corpus/__init__.py", "src/cad_corpus/corpus.py",
                     "src/cad_corpus/render_qa.py", "scripts/render_cad_pdf.py",
                     "scripts/cad_render_quality_audit.py",
                     "scripts/cad_render_runtime_report.py",
                     "scripts/cad_render_run_freeze.py"],
    "qa_config": ["configs/cad_render_qa.yaml", "configs/cad_corpus.yaml"],
    "qa_reports_v1": ["outputs/cad_render_quality_report_v1.json",
                      "outputs/cad_render_quality_report_v1.md"],
    "sample_pack_manifest": ["outputs/cad_render_qa_pack/qa_sample_manifest.jsonl",
                             "outputs/cad_render_qa_pack/index.html"],
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes(paths: List[str]) -> Dict[str, str]:
    return {p: sha(ROOT / p) for p in paths if (ROOT / p).exists()}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def tool(name: str) -> Dict:
    exe = shutil.which(name)
    if not exe:
        return {"path": None, "version": None}
    out = subprocess.run([exe, "-v"], capture_output=True, text=True)
    return {"path": Path(exe).name,
            "version": next((l for l in (out.stderr or out.stdout).splitlines()
                             if "version" in l.lower()), None)}


def explain(path: str) -> str:
    if path in EXPECTED_CHANGES:
        return EXPECTED_CHANGES[path]
    return next((why for prefix, why in EXPECTED_PREFIXES.items()
                 if path.startswith(prefix)), "")


def prior_drift() -> Dict:
    """For each earlier manifest: which of its files moved, and whether each
    move is one this run was expected to cause."""
    out = {}
    for label, path in PRIOR.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        moved, unexplained = [], []
        for group in data["groups"].values():
            for rel_path, recorded in (group.get("files") or {}).items():
                target = ROOT / rel_path
                now = sha(target) if target.exists() else None
                if now != recorded:
                    reason = explain(rel_path)
                    (moved if reason else unexplained).append(
                        {"path": rel_path, "reason": reason or None})
        out[label] = {"manifest_sha256": sha(path), "left_unmodified": True,
                      "now_reports": "FREEZE VIOLATION (expected; not claimed OK)",
                      "explained_changes": moved, "unexplained_changes": unexplained}
    return out


def snapshot() -> Dict:
    sources = [json.loads(l) for l in (MANIFESTS / "source_manifest_unreviewed.jsonl")
               .read_text(encoding="utf-8").splitlines() if l.strip()]
    pages = [json.loads(l) for l in (MANIFESTS / "page_manifest_unreviewed.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    report = json.loads((ROOT / "outputs" / "cad_render_quality_report_v1.json")
                        .read_text(encoding="utf-8"))
    originals = {s["original_path"]: sha(ROOT / s["original_path"]) for s in sources}
    pngs = {p["rendered_path"]: sha(ROOT / p["rendered_path"]) for p in pages}
    generated = sorted(rel(p) for p in list(MANIFESTS.glob("*"))
                       + list((CORPUS / "splits").glob("*")) if p.is_file())
    groups = {
        "original_pdfs": originals,
        "rendered_pngs": pngs,
        "page_and_corpus_manifests": hashes(generated),
        **{name: hashes(paths) for name, paths in GROUPS.items()},
    }
    a, h, rt = (report["automated_integrity_check"], report["human_visual_review"],
                report["runtime"])
    return {
        "stage": "cad_public_corpus/render_run_v1",
        "environment": {"pdftoppm": tool("pdftoppm"), "pdfinfo": tool("pdfinfo"),
                        "dpi": rt["dpi"], "substitute_renderer_used": False},
        "facts": {
            "pages_in_manifest": len(pages),
            "pages_failed": rt["pages_failed"], "file_conflicts": rt["file_conflicts"],
            "original_pdf_hashes_match_manifest": all(
                s["source_sha256"] == originals[s["original_path"]] for s in sources),
            "png_hashes_match_page_manifest": all(
                p["rendered_sha256"] == pngs[p["rendered_path"]] for p in pages),
            "qa_status_counts": a["qa_status_counts"],
            "qa_thresholds": "uncalibrated",
            "human_review_status": h["status"], "human_verified": h["human_verified"],
            "ocr_calls": 0, "vlm_calls": 0, "api_cost_usd": 0,
        },
        "supersedes": prior_drift(),
        "groups": {name: {"files": files, "file_count": len(files),
                          "total_bytes": sum((ROOT / f).stat().st_size for f in files)}
                   for name, files in groups.items()},
    }


def write() -> None:
    data = snapshot()
    bad = {k: v["unexplained_changes"] for k, v in data["supersedes"].items()
           if v["unexplained_changes"]}
    if bad or not data["facts"]["original_pdf_hashes_match_manifest"] \
            or not data["facts"]["png_hashes_match_page_manifest"]:
        print(f"REFUSING to write. unexplained drift: {bad}  facts: {data['facts']}")
        sys.exit(1)
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, g in data["groups"].items():
        print(f"{name:<28} {g['file_count']:>4} files {g['total_bytes'] / 1e6:>8.1f} MB")
    for label, info in data["supersedes"].items():
        print(f"supersedes {label}: {len(info['explained_changes'])} explained change(s), "
              f"0 unexplained")
    print(f"environment: {data['environment']}")
    print(f"-> {MANIFEST}")


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest; run with --write first")
        return 2
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = snapshot()
    findings = []
    for name, g in recorded["groups"].items():
        before, after = g["files"], current["groups"][name]["files"]
        for path in sorted(set(before) | set(after)):
            if before.get(path) != after.get(path):
                findings.append(f"CHANGED  {name}: {path}")
    for label, info in recorded["supersedes"].items():
        if current["supersedes"][label]["manifest_sha256"] != info["manifest_sha256"]:
            findings.append(f"PRIOR_MANIFEST_MODIFIED  {label}")
    if not findings:
        print("FREEZE OK")
        return 0
    print("FREEZE VIOLATION")
    for line in findings:
        print(f"  {line}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        write()
    else:
        sys.exit(verify())


if __name__ == "__main__":
    main()
