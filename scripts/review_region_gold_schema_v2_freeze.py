"""Freeze / verify gold schema v2 (Phase 2).

Writes outputs/review_region_gold_tooling_freeze_manifest_v3.json.

Phase 2 INTENTIONALLY changed five files frozen by earlier manifests:

  v1  outputs/review_region_gold_tooling_freeze_manifest.json
  v2  outputs/review_region_gold_tooling_freeze_manifest_v2.json

Neither is edited. Both now report FREEZE VIOLATION, and that is the honest
state: this manifest does NOT claim they are OK. It records which file moved
from which hash and why, then checks that every OTHER entry in v1 and v2 still
holds — so the change is exactly the declared set and nothing beside it.

    python scripts/review_region_gold_schema_v2_freeze.py --write
    python scripts/review_region_gold_schema_v2_freeze.py --verify
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
MANIFEST = ROOT / "outputs" / "review_region_gold_tooling_freeze_manifest_v3.json"
PRIOR = {
    "v1": ROOT / "outputs" / "review_region_gold_tooling_freeze_manifest.json",
    "v2": ROOT / "outputs" / "review_region_gold_tooling_freeze_manifest_v2.json",
}

GOLD_SCHEMA_VERSION = "region_gold/2"

INTENTIONALLY_CHANGED = {
    "src/vision/region_gold.py": {
        "before_sha256": "2522dc10978e1b02e218b245001706fb3ec9e950964f27e8d0a444e3582e3bd9",
        "last_frozen_in": "v1",
        "why": ("schema v2: value_legible, association_status, candidate_bbox, "
                "candidate_fields, migrated_from, schema_version; v2 validation; "
                "v1 rule value_visible&!field_visible relaxed ONLY for v2 "
                "ambiguous/unassigned"),
    },
    "scripts/validate_review_region_gold.py": {
        "before_sha256": "ee385bccc74d96b63ec169977c1d0a1c0e398fe2026eea41662f147b01de6e0b",
        "last_frozen_in": "v1",
        "why": ("reports schema versions and attribution distribution; flags "
                "candidate_bbox look-alikes; explicit missing-file error"),
    },
    "scripts/evaluate_review_region_localization.py": {
        "before_sha256": "5c4e106b9cde33cb7e4fe56dfe3dd6a812bc80e275f8937a70e016f52ca81363",
        "last_frozen_in": "v2",
        "why": ("phase2 key: attribution-aware formal localisation on confirmed "
                "only; ambiguous/unassigned safety; unknown counted; phase-1 "
                "report added to protected outputs. Legacy and phase-1 output "
                "unchanged."),
    },
    "docs/review_region_annotation_guide.md": {
        "before_sha256": "64102660853be713abbcd35d8c27ada0c17eb7b24c3a219ea864647fb2a94ae4",
        "last_frozen_in": "v1",
        "why": "section 9 appended: visible / legible / attributed, v2 examples",
    },
    "tests/test_localization_eval_phase1.py": {
        "before_sha256": "cc502c5e9fe8e7652e7cb3e056ef3953004b0b7affcb8bcb32ac48e00adfdd09",
        "last_frozen_in": "v2",
        "why": ("one assertion widened: it pinned the report's extra keys to "
                "exactly {phase1, legacy_metric_notes}; phase 2 adds 'phase2'"),
    },
}

GROUPS = {
    "gold_schema": ["src/vision/region_gold.py"],
    "gold_tooling": [
        "scripts/validate_review_region_gold.py",
        "scripts/evaluate_review_region_localization.py",
        "scripts/migrate_review_region_gold.py",
    ],
    "annotation_guide": ["docs/review_region_annotation_guide.md"],
    "tests": [
        "tests/test_region_gold_schema_v2.py",
        "tests/test_localization_eval_phase1.py",
    ],
    "phase2_outputs": [
        "outputs/review_region_localization_eval_phase2_gold_test.json",
        "outputs/review_region_localization_eval_phase2_gold_test.md",
    ],
    # Read, never written. Its hash proves no human judgement was rewritten.
    "human_gold_read_only": ["data/review_region_gold_test.jsonl"],
    "historical_reports_untouched": [
        "outputs/review_region_localization_eval.json",
        "outputs/review_region_localization_eval.md",
        "outputs/review_region_localization_eval_gold_test.json",
        "outputs/review_region_localization_eval_gold_test.md",
        "outputs/review_region_localization_eval_phase1_gold_test.json",
        "outputs/review_region_localization_eval_phase1_gold_test.md",
    ],
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


def prior_residual() -> Dict[str, object]:
    """Every v1/v2 entry except the declared changes must still hold."""
    drift, recorded_for_declared = [], {}
    for label, path in PRIOR.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for group in data["groups"].values():
            for rel, recorded in group["files"].items():
                if rel in INTENTIONALLY_CHANGED:
                    recorded_for_declared.setdefault(rel, {})[label] = recorded
                    continue
                if not (ROOT / rel).exists() or sha(rel) != recorded:
                    drift.append(f"{label}:{rel}")
    return {
        "prior_manifest_sha256": {label: sha(str(p.relative_to(ROOT)))
                                  for label, p in PRIOR.items()},
        "undeclared_drift": sorted(set(drift)),
        "declared_changes": {
            rel: {**info, "recorded_in_prior": recorded_for_declared.get(rel, {}),
                  "current_sha256": sha(rel)}
            for rel, info in INTENTIONALLY_CHANGED.items()},
    }


def snapshot() -> dict:
    groups = {name: hashes(paths) for name, paths in GROUPS.items()}
    return {
        "stage": "review_region_gold_schema/phase2",
        "gold_schema_version": GOLD_SCHEMA_VERSION,
        "supersedes": {
            "manifests": {k: str(v.relative_to(ROOT)).replace("\\", "/")
                          for k, v in PRIOR.items()},
            "prior_manifests_left_unmodified": True,
            "prior_manifests_now_report": "FREEZE VIOLATION (expected; not claimed OK)",
            "intentionally_changed": INTENTIONALLY_CHANGED,
        },
        "compatibility": {
            "v1_read_as": "schema_version region_gold/1, association_status unknown, "
                          "value_legible None",
            "v1_value_visible": "keeps v1 meaning; never reinterpreted as presence "
                                "and never copied into value_legible",
            "migration": "scripts/migrate_review_region_gold.py; dry run by default; "
                         "writes only to a NEW path; infers nothing",
        },
        "conclusions": {
            "still_valid": [
                "legacy IoU metrics and phase-1 coverage as statements about "
                "boxes vs region-type gold",
                "phase-1 unlocatable safety/cost split",
                "all resolver, crop and multimodal manifests",
            ],
            "superseded": [
                "reading legacy/phase-1 localisation recall as FIELD localisation; "
                "under schema v2 only association_status=confirmed records support "
                "that, and the current gold has none",
            ],
        },
        "prior_residual": prior_residual(),
        "measured_facts": {
            "real_ocr_calls": 0, "real_vlm_calls": 0, "external_api_calls": 0,
            "gold_modified": False, "resolver_modified": False,
            "thresholds_modified": False,
        },
        "groups": {name: {"files": files, "file_count": len(files),
                          "sha256": group_hash(files)}
                   for name, files in groups.items()},
    }


def write() -> None:
    data = snapshot()
    residual = data["prior_residual"]
    if residual["undeclared_drift"]:
        print("REFUSING to write: files drifted beyond the declared set:")
        for rel in residual["undeclared_drift"]:
            print(f"  {rel}")
        sys.exit(1)
    for rel, info in residual["declared_changes"].items():
        recorded = info["recorded_in_prior"].get(info["last_frozen_in"])
        if recorded != info["before_sha256"]:
            print(f"REFUSING to write: {rel} before-hash does not match "
                  f"{info['last_frozen_in']} ({recorded})")
            sys.exit(1)
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, group in data["groups"].items():
        print(f"{name:<30} {group['file_count']:>2} files  {group['sha256'][:16]}…")
    print("\nsupersedes:")
    for rel, info in residual["declared_changes"].items():
        print(f"  {rel:<48} {info['last_frozen_in']} "
              f"{info['before_sha256'][:12]}… -> {info['current_sha256'][:12]}…")
    print("undeclared drift in v1/v2: none")
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
    for label, digest in recorded["prior_residual"]["prior_manifest_sha256"].items():
        if current["prior_residual"]["prior_manifest_sha256"][label] != digest:
            findings.append(f"PRIOR_MANIFEST_MODIFIED  {label}")
    for rel in current["prior_residual"]["undeclared_drift"]:
        findings.append(f"UNDECLARED_DRIFT  {rel}")
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
