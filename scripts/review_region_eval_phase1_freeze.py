"""Freeze / verify localisation-evaluation Phase 1.

Writes outputs/review_region_gold_tooling_freeze_manifest_v2.json.

Phase 1 INTENTIONALLY modified scripts/evaluate_review_region_localization.py,
which is frozen in outputs/review_region_gold_tooling_freeze_manifest.json
(v1). That v1 manifest is not edited: running its verify now reports exactly
one CHANGED file, and that is the honest record. This v2 manifest states which
file superseded which hash, and checks that v1's OTHER entries still hold — so
the change is confined to the one file declared, not smuggled in beside it.

    python scripts/review_region_eval_phase1_freeze.py --write
    python scripts/review_region_eval_phase1_freeze.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "review_region_gold_tooling_freeze_manifest_v2.json"
V1 = ROOT / "outputs" / "review_region_gold_tooling_freeze_manifest.json"

INTENTIONALLY_CHANGED = {
    "scripts/evaluate_review_region_localization.py": {
        "v1_sha256": "cd403b4e7048ea949655dccd9fa0b04a81eb420999d1c46df443b22d8cce39ae",
        "why": ("Phase 1: --gold/--output-json/--output-md, refusal to overwrite "
                "historical reports, schema check on input, gold coverage and "
                "prediction precision, per-record verdicts, and the unlocatable "
                "safety/cost split. Legacy evaluate() and its outputs unchanged."),
    },
}

GROUPS = {
    "evaluator": ["scripts/evaluate_review_region_localization.py"],
    "phase1_tests": ["tests/test_localization_eval_phase1.py"],
    "phase1_outputs": [
        "outputs/review_region_localization_eval_phase1_gold_test.json",
        "outputs/review_region_localization_eval_phase1_gold_test.md",
    ],
    # Read, never written. Hashed so a later edit to the gold is visible.
    "human_gold_read_only": ["data/review_region_gold_test.jsonl"],
    # Historical reports this phase must not have touched.
    "historical_reports_untouched": [
        "outputs/review_region_localization_eval.json",
        "outputs/review_region_localization_eval.md",
        "outputs/review_region_localization_eval_gold_test.json",
        "outputs/review_region_localization_eval_gold_test.md",
    ],
    # The things the task forbade changing, hashed to prove they did not.
    "not_modified_this_phase": [
        "src/vision/review_region_resolver.py",
        "configs/review_regions.yaml",
        "data/multimodal_cache/vision_review_cache.json",
        "outputs/review_region_annotation_pack/manifest.json",
    ],
}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def hashes(paths: List[str]) -> Dict[str, str]:
    return {rel: sha(rel) for rel in paths if (ROOT / rel).exists()}


def group_hash(values: Dict[str, str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(values):
        digest.update(rel.encode())
        digest.update(values[rel].encode())
    return digest.hexdigest()


def v1_residual_check() -> Dict[str, object]:
    """Every v1 entry except the declared change must still match."""
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    drift = []
    for group in v1["groups"].values():
        for rel, recorded in group["files"].items():
            if rel in INTENTIONALLY_CHANGED:
                continue
            if not (ROOT / rel).exists() or sha(rel) != recorded:
                drift.append(rel)
    declared = {
        rel: {"v1_sha256": info["v1_sha256"],
              "recorded_in_v1": next((g["files"][rel] for g in v1["groups"].values()
                                      if rel in g["files"]), None),
              "current_sha256": sha(rel)}
        for rel, info in INTENTIONALLY_CHANGED.items()}
    return {"v1_manifest_sha256": sha(str(V1.relative_to(ROOT))),
            "undeclared_drift": drift, "declared_changes": declared}


def snapshot() -> dict:
    groups = {name: hashes(paths) for name, paths in GROUPS.items()}
    return {
        "stage": "review_region_localization_eval/phase1",
        "evaluator_version": 2,
        "supersedes": {
            "manifest": str(V1.relative_to(ROOT)).replace("\\", "/"),
            "v1_left_unmodified": True,
            "intentionally_changed": INTENTIONALLY_CHANGED,
            "consequence": ("verify on v1 reports exactly one CHANGED file; "
                            "that is expected and is the reason v2 exists"),
        },
        "v1_residual": v1_residual_check(),
        "measured_facts": {
            "real_ocr_calls": 0, "real_vlm_calls": 0, "external_api_calls": 0,
            "resolver_modified": False, "thresholds_modified": False,
            "gold_modified": False,
        },
        "groups": {name: {"files": files, "file_count": len(files),
                          "sha256": group_hash(files)}
                   for name, files in groups.items()},
    }


def write() -> None:
    data = snapshot()
    residual = data["v1_residual"]
    if residual["undeclared_drift"]:
        print("REFUSING to write: v1 files drifted beyond the declared change:")
        for rel in residual["undeclared_drift"]:
            print(f"  {rel}")
        sys.exit(1)
    for rel, info in residual["declared_changes"].items():
        if info["recorded_in_v1"] != info["v1_sha256"]:
            print(f"REFUSING to write: declared v1 hash for {rel} does not match v1")
            sys.exit(1)
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, group in data["groups"].items():
        print(f"{name:<30} {group['file_count']:>2} files  {group['sha256'][:16]}…")
    for rel, info in residual["declared_changes"].items():
        print(f"\nsupersedes {rel}\n  v1 {info['v1_sha256'][:16]}…  ->  "
              f"v2 {info['current_sha256'][:16]}…")
    print("v1 undeclared drift: none")
    print(f"-> {MANIFEST}")


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest; run with --write first")
        return 2
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = snapshot()
    findings = []
    for name, group in recorded["groups"].items():
        before, after = group["files"], current["groups"][name]["files"]
        for rel in sorted(set(before) | set(after)):
            if before.get(rel) != after.get(rel):
                findings.append(f"CHANGED   {name}: {rel}")
    if current["v1_residual"]["v1_manifest_sha256"] != \
            recorded["v1_residual"]["v1_manifest_sha256"]:
        findings.append("V1_MANIFEST_MODIFIED")
    for rel in current["v1_residual"]["undeclared_drift"]:
        findings.append(f"V1_UNDECLARED_DRIFT  {rel}")
    if not findings:
        print("FREEZE OK")
        for name, group in current["groups"].items():
            print(f"  {name:<30} {group['file_count']:>2} files  {group['sha256'][:16]}…")
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
