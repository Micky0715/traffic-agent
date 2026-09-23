"""Freeze / verify graded review-region resolution.

Writes outputs/multimodal_region_resolution_freeze_manifest.json. The previous
round's outputs/multimodal_freeze_manifest.json is NOT touched and NOT
re-hashed: it is the before-evidence, and a stage that rewrites its
predecessor's record can hide exactly the change a reader is looking for.

Import-closure drift is reported, not absorbed. If this stage starts depending
on a module it did not depend on before, the manifest names it under
`new_dependencies_vs_previous_round` instead of quietly carrying a new hash.

    python scripts/review_region_freeze.py --write
    python scripts/review_region_freeze.py --verify
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
MANIFEST = ROOT / "outputs" / "multimodal_region_resolution_freeze_manifest.json"
PREVIOUS = ROOT / "outputs" / "multimodal_freeze_manifest.json"

ENTRY_POINTS = [
    "src/vision/review_region_resolver.py",
    "src/vision/review_region_config.py",
]

EXPLICIT_CONFIG = ["configs/review_regions.yaml"]
EXPLICIT_TESTS = ["tests/test_review_region_resolver.py"]
EXPLICIT_EVAL_INPUTS = [
    "data/review_region_cases_registered.json",
    "data/multimodal_cases_registered.json",
]
EXPLICIT_EVALUATORS = [
    "scripts/review_region_baseline_audit.py",
    "scripts/review_region_resolution_report.py",
]
EXPLICIT_REPORTS = [
    "outputs/review_region_baseline_audit.json",
    "outputs/review_region_resolution_report.json",
]
# No human-reviewed bbox labels. Recorded empty on purpose: a report wanting to
# quote localisation ACCURACY has to find this group non-empty first.
EXPLICIT_BBOX_GOLD: List[str] = []


def import_closure(entries: List[str]) -> List[str]:
    seen: set = set()
    stack = list(entries)
    while stack:
        rel = stack.pop()
        path = ROOT / rel
        if rel in seen or not path.exists():
            continue
        seen.add(rel)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
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


def previous_cache_untouched() -> Dict[str, object]:
    """The two real VLM cache entries from last round must be byte-identical."""
    cache = ROOT / "data" / "multimodal_cache" / "vision_review_cache.json"
    if not PREVIOUS.exists() or not cache.exists():
        return {"checked": False, "note": "previous manifest or cache absent"}
    recorded = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    before = recorded["groups"]["multimodal_cache"]["files"].get(
        "data/multimodal_cache/vision_review_cache.json")
    now = hashlib.sha256(cache.read_bytes()).hexdigest()
    store = json.loads(cache.read_text(encoding="utf-8"))
    return {
        "checked": True,
        "unchanged": before == now,
        "recorded_sha256": before,
        "current_sha256": now,
        "entries": len(store),
        "recorded_from_live_inference": sum(
            1 for e in store.values() if e.get("recorded_from") == "live_inference"),
    }


def dependency_drift(code: List[str]) -> Dict[str, object]:
    if not PREVIOUS.exists():
        return {"comparable": False}
    before = set(json.loads(PREVIOUS.read_text(encoding="utf-8"))
                 ["groups"]["multimodal_runtime_code"]["files"])
    return {
        "comparable": True,
        "new_dependencies_vs_previous_round": sorted(set(code) - before),
        "note": ("listed explicitly rather than absorbed into an updated hash; "
                 "the previous manifest is unmodified"),
    }


def snapshot() -> dict:
    code = hashes(import_closure(ENTRY_POINTS))
    config = hashes(EXPLICIT_CONFIG)
    tests = hashes(EXPLICIT_TESTS)
    eval_inputs = hashes(EXPLICIT_EVAL_INPUTS)
    evaluators = hashes(EXPLICIT_EVALUATORS)
    reports = hashes(EXPLICIT_REPORTS)
    gold = hashes(EXPLICIT_BBOX_GOLD)
    return {
        "stage": "review_region_resolution/1.0",
        "real_vlm_calls": 0,
        "human_reviewed_bbox_gold": False,
        "threshold_calibration": ("engineering starting values; NOT calibrated "
                                  "against human-reviewed bbox gold"),
        "does_not_modify": [
            "outputs/multimodal_freeze_manifest.json",
            "outputs/vision_freeze_manifest.json",
            "outputs/table_freeze_manifest.json",
            "outputs/routing_freeze_manifest.json",
            "data/multimodal_cache/vision_review_cache.json",
        ],
        "previous_round_cache": previous_cache_untouched(),
        "dependency_drift": dependency_drift(sorted(code)),
        "groups": {
            "region_runtime_code": {"source": "static import closure", "files": code,
                                    "file_count": len(code), "sha256": group_hash(code)},
            "region_config": {"source": "explicit list", "files": config,
                              "file_count": len(config), "sha256": group_hash(config)},
            "region_tests": {"source": "explicit list", "files": tests,
                             "file_count": len(tests), "sha256": group_hash(tests)},
            "region_eval_inputs": {"source": "explicit list", "files": eval_inputs,
                                   "file_count": len(eval_inputs),
                                   "sha256": group_hash(eval_inputs)},
            "region_evaluators": {"source": "explicit list", "files": evaluators,
                                  "file_count": len(evaluators),
                                  "sha256": group_hash(evaluators)},
            "region_reports": {"source": "explicit list", "files": reports,
                               "file_count": len(reports), "sha256": group_hash(reports)},
            "region_bbox_gold": {"source": "explicit list", "files": gold,
                                 "file_count": len(gold), "sha256": group_hash(gold),
                                 "human_reviewed": False,
                                 "note": ("empty: no human-reviewed bbox gold. No "
                                          "localisation accuracy, crop recall or "
                                          "field recovery rate may be reported "
                                          "while this group is empty.")},
        },
    }


LABELS = {
    "region_runtime_code": "CHANGED",
    "region_config": "CONFIG_CHANGED",
    "region_tests": "TEST_CHANGED",
    "region_eval_inputs": "EVAL_INPUT_CHANGED",
    "region_evaluators": "EVALUATOR_CHANGED",
    "region_reports": "REPORT_CHANGED",
    "region_bbox_gold": "GOLD_CHANGED",
}


def write() -> None:
    data = snapshot()
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, group in data["groups"].items():
        print(f"{name:<22} {group['file_count']:>3} files  {group['sha256'][:16]}…")
    cache = data["previous_round_cache"]
    print(f"\nprevious round's VLM cache unchanged: {cache.get('unchanged')} "
          f"({cache.get('entries')} entries, "
          f"{cache.get('recorded_from_live_inference')} from live inference)")
    drift = data["dependency_drift"]
    if drift.get("comparable"):
        print(f"new dependencies vs previous round: "
              f"{drift['new_dependencies_vs_previous_round']}")
    print(f"real_vlm_calls: {data['real_vlm_calls']}   "
          f"human-reviewed bbox gold: {data['human_reviewed_bbox_gold']}")
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
    cache = current["previous_round_cache"]
    if cache.get("checked") and not cache.get("unchanged"):
        findings.append("PREVIOUS_CACHE_MODIFIED  data/multimodal_cache/"
                        "vision_review_cache.json")
    if not findings:
        print("FREEZE OK")
        for name, group in current["groups"].items():
            print(f"  {name:<22} {group['file_count']:>3} files  {group['sha256'][:16]}…")
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
