"""Freeze / verify the table-structure stage.

Its own manifest, alongside the vision and routing freezes rather than inside
them: this stage's inputs include images and OCR fixtures that no import
analysis can reach, and folding them into an existing group would make that
group fire on changes that have nothing to do with it.

Six groups, because they fail differently and a diff has to say which kind of
input moved:

  table_runtime_code   the parser
  table_config         thresholds that decide structure
  table_images         the pixels
  table_ocr_fixtures   the recognition results replayed over them
  table_gold           human-reviewed structure labels — EMPTY, and recorded
                       as empty so no report can imply otherwise
  table_evaluator      the audit and inventory scripts

    python scripts/table_freeze.py --write
    python scripts/table_freeze.py --verify
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "table_freeze_manifest.json"

ENTRY_POINTS = [
    "src/tables/grid.py",
    "src/tables/cells.py",
    "src/tables/headers.py",
    "src/tables/continuation.py",
    "src/tables/chunks.py",
    "src/tables/crop.py",
    "src/tables/review.py",
    "src/tables/schemas.py",
    "src/tables/config.py",
]

EXPLICIT_CONFIG = ["configs/tables.yaml"]
EXPLICIT_EVALUATOR = [
    "scripts/table_corpus_inventory.py",
    "scripts/table_bad_case_audit.py",
    "scripts/table_structure_audit.py",
]
# Human-reviewed structure labels. Nothing here yet; the empty list is the
# point — a report that wants to quote a structure-accuracy number has to find
# this group non-empty first.
EXPLICIT_GOLD: List[str] = []

IMAGE_DIRS = ["data/drawings", "data/adversarial"]
FIXTURE_DIR = "data/ocr_fixtures"


def import_closure(entries: List[str]) -> List[str]:
    seen: set = set()
    stack = list(entries)
    while stack:
        rel = stack.pop()
        path = ROOT / rel
        if rel in seen or not path.exists():
            continue
        seen.add(rel)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            if isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            for module in modules:
                if not module.startswith("src."):
                    continue
                as_module = module.replace(".", "/") + ".py"
                as_package = module.replace(".", "/") + "/__init__.py"
                if (ROOT / as_module).exists():
                    stack.append(as_module)
                elif (ROOT / as_package).exists():
                    stack.append(as_package)
    return sorted(seen)


def collect_images() -> List[str]:
    out = []
    for directory in IMAGE_DIRS:
        path = ROOT / directory
        if not path.exists():
            continue
        out.extend(str(p.relative_to(ROOT)).replace("\\", "/")
                   for p in sorted(path.glob("*.png"))
                   if ".processed" not in p.stem)
    return out


def collect_fixtures() -> List[str]:
    path = ROOT / FIXTURE_DIR
    if not path.exists():
        return []
    return [str(p.relative_to(ROOT)).replace("\\", "/")
            for p in sorted(path.glob("*.json"))]


def hashes(paths: List[str]) -> Dict[str, str]:
    out = {}
    for rel in paths:
        target = ROOT / rel
        if target.exists():
            out[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
    return out


def group_hash(values: Dict[str, str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(values):
        digest.update(rel.encode())
        digest.update(values[rel].encode())
    return digest.hexdigest()


def fixture_provenance() -> Dict[str, object]:
    """Which fixtures are real recognition output, and from what."""
    real, versions = 0, set()
    for rel in collect_fixtures():
        payload = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        provenance = payload.get("provenance", {})
        if provenance.get("paddleocr_version"):
            real += 1
            versions.add(f"paddleocr {provenance['paddleocr_version']} / "
                         f"paddlepaddle {provenance.get('paddlepaddle_version')}")
    return {
        "fixture_count": len(collect_fixtures()),
        "with_real_ocr_provenance": real,
        "parser_versions": sorted(versions),
        "is_live_inference": False,
        "note": ("fixtures replay a recorded real PaddleOCR run; real recognition "
                 "output, not live inference"),
    }


def snapshot() -> dict:
    code = hashes(import_closure(ENTRY_POINTS))
    config = hashes(EXPLICIT_CONFIG)
    images = hashes(collect_images())
    fixtures = hashes(collect_fixtures())
    gold = hashes(EXPLICIT_GOLD)
    evaluator = hashes(EXPLICIT_EVALUATOR)
    return {
        "static_analysis_limitation": (
            "代码范围来自静态 import 闭包；图片、配置与 OCR fixture 无法由 import "
            "分析发现，因此显式列入。"),
        "groups": {
            "table_runtime_code": {"source": "static import closure",
                                   "files": code, "file_count": len(code),
                                   "sha256": group_hash(code)},
            "table_config": {"source": "explicit list", "files": config,
                             "file_count": len(config), "sha256": group_hash(config)},
            "table_images": {"source": "explicit glob", "files": images,
                             "file_count": len(images), "sha256": group_hash(images)},
            "table_ocr_fixtures": {"source": "explicit glob", "files": fixtures,
                                   "file_count": len(fixtures),
                                   "sha256": group_hash(fixtures),
                                   "provenance": fixture_provenance()},
            "table_gold": {"source": "explicit list", "files": gold,
                           "file_count": len(gold), "sha256": group_hash(gold),
                           "human_reviewed": False,
                           "note": ("empty: no human-reviewed structure gold exists. "
                                    "No structure-accuracy number may be reported "
                                    "while this group is empty.")},
            "table_evaluator": {"source": "explicit list", "files": evaluator,
                                "file_count": len(evaluator),
                                "sha256": group_hash(evaluator)},
        },
        "parser_version": "tables/1.0",
    }


LABELS = {
    "table_runtime_code": "CHANGED",
    "table_config": "CONFIG_CHANGED",
    "table_images": "IMAGE_CHANGED",
    "table_ocr_fixtures": "FIXTURE_CHANGED",
    "table_gold": "GOLD_CHANGED",
    "table_evaluator": "EVALUATOR_CHANGED",
}


def write() -> None:
    data = snapshot()
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, group in data["groups"].items():
        print(f"{name:<24} {group['file_count']:>3} files  {group['sha256'][:16]}…")
    provenance = data["groups"]["table_ocr_fixtures"]["provenance"]
    print(f"\nfixtures with real OCR provenance: "
          f"{provenance['with_real_ocr_provenance']}/{provenance['fixture_count']}")
    print(f"human-reviewed gold: {data['groups']['table_gold']['human_reviewed']}")
    print(f"-> {MANIFEST}")


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest; run with --write first")
        return 2
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = snapshot()
    findings = []
    for name, label in LABELS.items():
        before = recorded["groups"][name]["files"]
        after = current["groups"][name]["files"]
        for rel in sorted(set(after) - set(before)):
            findings.append(f"ADDED             {name}: {rel}")
        for rel in sorted(set(before) - set(after)):
            findings.append(f"REMOVED           {name}: {rel}")
        for rel in sorted(set(before) & set(after)):
            if before[rel] != after[rel]:
                findings.append(f"{label:<17} {name}: {rel}")
    if not findings:
        print("FREEZE OK")
        for name, group in current["groups"].items():
            print(f"  {name:<24} {group['file_count']:>3} files  {group['sha256'][:16]}…")
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
