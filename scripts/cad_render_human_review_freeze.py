"""Freeze / verify the human render review merge.

Writes outputs/cad_render_human_review_freeze_manifest_v1.json. Earlier
manifests are left untouched. One of them,
outputs/cad_corpus_render_run_freeze_manifest_v1.json, snapshots the whole
manifests directory, so the new human review file shows there as an added
file; this manifest records that as the one expected difference and refuses to
write if that manifest reports anything else.

    python scripts/cad_render_human_review_freeze.py --write
    python scripts/cad_render_human_review_freeze.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "cad_render_human_review_freeze_manifest_v1.json"
REVIEW = "data/cad_public_corpus/manifests/cad_render_human_review_v1.jsonl"
SUPP = ROOT / "outputs" / "cad_render_qa_pack_supplement_v1"
EXPECTED_RENDER_RUN_DIFF = [
    f"CHANGED  page_and_corpus_manifests: {REVIEW}"]

GROUPS = {
    "human_review_source": [REVIEW],
    "reports": ["outputs/cad_render_human_review_report_v1.json",
                "outputs/cad_render_human_review_report_v1.md",
                "outputs/cad_render_quality_report_v2.json",
                "outputs/cad_render_quality_report_v2.md"],
    "first_round_pack_unchanged": ["outputs/cad_render_qa_pack/qa_sample_manifest.jsonl",
                                   "outputs/cad_render_quality_report_v1.json",
                                   "outputs/cad_render_quality_report_v1.md"],
    "runtime_code": ["src/cad_corpus/human_review.py",
                     "scripts/cad_render_human_review.py",
                     "scripts/cad_render_human_review_freeze.py",
                     "tests/test_cad_human_review.py"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(paths: List[str]) -> Dict[str, str]:
    return {p: sha(ROOT / p) for p in paths if (ROOT / p).exists()}


def render_run_diff() -> List[str]:
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "cad_render_run_freeze.py"),
                          "--verify"], capture_output=True, text=True, cwd=ROOT)
    return [l.strip() for l in out.stdout.splitlines()
            if l.strip() and not l.startswith("FREEZE")]


def snapshot() -> Dict:
    records = [json.loads(l) for l in (ROOT / REVIEW).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    pages = {json.loads(l)["page_id"]: json.loads(l) for l in
             (ROOT / "data/cad_public_corpus/manifests/page_manifest_unreviewed.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()}
    reviewed_pngs = {pages[r["page_id"]]["rendered_path"]:
                     sha(ROOT / pages[r["page_id"]]["rendered_path"]) for r in records}
    supp = sorted(p.relative_to(ROOT).as_posix() for p in SUPP.rglob("*") if p.is_file())
    report = json.loads((ROOT / "outputs/cad_render_human_review_report_v1.json")
                        .read_text(encoding="utf-8"))
    groups = {name: hashes(paths) for name, paths in GROUPS.items()}
    groups["reviewed_page_pngs"] = reviewed_pngs
    groups["supplement_pack"] = hashes(supp)
    return {
        "stage": "cad_public_corpus/human_render_review_v1",
        "facts": {
            "human_reviewed_pages": len(report["human_reviewed_pages"]),
            "pass_pages": len(report["pass_pages"]),
            "failed_pages": len(report["failed_pages"]),
            "review_capture": report["review_capture"],
            "crop_warning": {k: report[k] for k in (
                "crop_warning_total", "crop_warning_true_positive",
                "crop_warning_false_positive", "crop_warning_unreviewed")},
            "human_verified_corpus_level": False,
            "human_verified_scope_pages": report["human_verified_scope"]["count"],
            "supplement_pages_pending": report["supplement_pages"],
            "thresholds_changed": False, "ocr_calls": 0, "vlm_calls": 0,
        },
        "prior_manifests": {
            "cad_corpus_render_run_freeze_manifest_v1": {
                "left_unmodified": True,
                "expected_difference": EXPECTED_RENDER_RUN_DIFF,
                "why": "the human review file was added to the manifests directory it snapshots",
            }},
        "groups": {name: {"files": files, "file_count": len(files)}
                   for name, files in groups.items()},
    }


def write() -> None:
    diff = render_run_diff()
    if diff != EXPECTED_RENDER_RUN_DIFF:
        print(f"REFUSING to write: render-run manifest reports unexpected drift: {diff}")
        sys.exit(1)
    data = snapshot()
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, g in data["groups"].items():
        print(f"{name:<28} {g['file_count']:>3} files")
    print(f"facts: {data['facts']}")
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
