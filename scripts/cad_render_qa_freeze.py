"""Freeze / verify CAD render QA tooling and its outputs.

Writes outputs/cad_corpus_render_qa_freeze_manifest_v1.json. The Phase 0
corpus manifest (outputs/cad_corpus_phase0_freeze_manifest.json) and every
older manifest are left untouched; this one records the QA layer on top.

State is recorded as measured, including when it is empty: while Poppler is
missing there are no rendered pages and no sample, and the manifest says
"blocked" rather than leaving the groups out.

    python scripts/cad_render_qa_freeze.py --write
    python scripts/cad_render_qa_freeze.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "cad_corpus_render_qa_freeze_manifest_v1.json"
REPORT = ROOT / "outputs" / "cad_render_quality_report.json"
PACK = ROOT / "outputs" / "cad_render_qa_pack"
SOURCES = ROOT / "data" / "cad_public_corpus" / "manifests" / "source_manifest_unreviewed.jsonl"

GROUPS = {
    "qa_code": ["src/cad_corpus/render_qa.py",
                "scripts/cad_render_quality_audit.py",
                "scripts/cad_render_qa_freeze.py"],
    "qa_config": ["configs/cad_render_qa.yaml"],
    "qa_tests": ["tests/test_cad_render_qa.py"],
    "qa_reports": ["outputs/cad_render_quality_report.json",
                   "outputs/cad_render_quality_report.md"],
    # The frozen render config: its DPI must not have moved.
    "render_config_unchanged": ["configs/cad_corpus.yaml"],
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes(paths: List[str]) -> Dict[str, str]:
    return {p: sha(ROOT / p) for p in paths if (ROOT / p).exists()}


def snapshot() -> Dict:
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    a = report.get("automated_integrity_check", {})
    h = report.get("human_visual_review", {})
    pack_files = sorted(p.relative_to(ROOT).as_posix() for p in PACK.rglob("*")
                        if p.is_file()) if PACK.exists() else []
    sources = [json.loads(l) for l in SOURCES.read_text(encoding="utf-8").splitlines()
               if l.strip()] if SOURCES.exists() else []
    originals = {s["original_path"]: sha(ROOT / s["original_path"]) for s in sources}
    groups = {name: hashes(paths) for name, paths in GROUPS.items()}
    groups["qa_pack"] = hashes(pack_files)
    groups["original_sources_reference"] = originals
    return {
        "stage": "cad_public_corpus/phase0_render_qa",
        "facts": {
            "qa_status": report.get("status"),
            "pages_rendered_and_readable": a.get("pages_rendered_and_readable"),
            "qa_status_counts": a.get("qa_status_counts"),
            "sample_pages": h.get("sample_pages"),
            "human_review_status": h.get("status"),
            "human_verified": h.get("human_verified"),
            "original_pdf_hashes_match_manifest": all(
                s["source_sha256"] == originals[s["original_path"]] for s in sources),
            "poppler_available": bool(shutil.which("pdftoppm") and shutil.which("pdfinfo")),
            "substitute_renderer_used": False,
            "thresholds": "uncalibrated",
            "ocr_calls": report.get("ocr_calls"), "vlm_calls": report.get("vlm_calls"),
        },
        "groups": {name: {"files": files, "file_count": len(files)}
                   for name, files in groups.items()},
    }


def write() -> None:
    data = snapshot()
    if not data["facts"]["original_pdf_hashes_match_manifest"]:
        print("REFUSING to write: an original PDF no longer matches the manifest")
        sys.exit(1)
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, g in data["groups"].items():
        print(f"{name:<28} {g['file_count']:>3} files")
    print(f"\nfacts: {data['facts']}")
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
        for rel in sorted(set(before) | set(after)):
            if before.get(rel) != after.get(rel):
                findings.append(f"CHANGED  {name}: {rel}")
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
