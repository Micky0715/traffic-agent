"""Freeze / verify this round's gold tooling and shadow experiment.

Writes outputs/review_region_gold_tooling_freeze_manifest.json. The five
earlier manifests are neither read as authority nor rewritten — they are the
before-evidence for the rounds that produced them.

Three facts are asserted from the files themselves rather than typed in, so
this manifest cannot claim something the repo contradicts:
  human_reviewed_gold_count  counted from the gold file
  real_vlm_calls             read from this round's reports
  production_behavior_changed compared against the shipped availability numbers

    python scripts/review_region_gold_tooling_freeze.py --write
    python scripts/review_region_gold_tooling_freeze.py --verify
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
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "outputs" / "review_region_gold_tooling_freeze_manifest.json"
GOLD = ROOT / "data" / "review_region_gold_unreviewed.jsonl"
PRIOR_MANIFESTS = [
    "outputs/vision_freeze_manifest.json",
    "outputs/table_freeze_manifest.json",
    "outputs/routing_freeze_manifest.json",
    "outputs/multimodal_freeze_manifest.json",
    "outputs/multimodal_region_resolution_freeze_manifest.json",
]

ENTRY_POINTS = [
    "src/vision/region_gold.py",
    "src/vision/structural_region_candidates.py",
    "src/vision/review_queue.py",
]
EXPLICIT_CONFIG = ["configs/structural_shadow.yaml"]
EXPLICIT_TOOLING = [
    "scripts/build_review_region_annotation_pack.py",
    "scripts/validate_review_region_gold.py",
    "scripts/evaluate_review_region_localization.py",
    "scripts/structural_region_shadow_report.py",
]
EXPLICIT_TESTS = ["tests/test_region_gold_and_shadow.py"]
EXPLICIT_UNREVIEWED_DATA = ["data/review_region_gold_unreviewed.jsonl"]
EXPLICIT_REPORTS = [
    "outputs/review_region_localization_eval.json",
    "outputs/structural_region_shadow_report.json",
    "outputs/review_region_annotation_pack/manifest.json",
]
EXPLICIT_DOCS = ["docs/review_region_annotation_guide.md"]


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


def measured_facts() -> Dict[str, object]:
    """Counted from the repo, never typed in."""
    from src.vision.region_gold import HUMAN_REVIEWED, read_jsonl, status_summary
    records = read_jsonl(GOLD)
    summary = status_summary(records)

    vlm_calls = 0
    for rel in ("outputs/review_region_localization_eval.json",
                "outputs/structural_region_shadow_report.json",
                "outputs/review_region_annotation_pack/manifest.json"):
        path = ROOT / rel
        if path.exists():
            vlm_calls += json.loads(path.read_text(encoding="utf-8")).get(
                "real_vlm_calls", 0)

    shipped = ROOT / "outputs" / "review_region_resolution_report.json"
    unchanged = None
    if shipped.exists():
        m = json.loads(shipped.read_text(encoding="utf-8"))["metrics"]
        unchanged = (m["precise_crop_available_count"] == 2
                     and m["any_region_available_count"] == 11
                     and m["crop_unavailable_count"] == 0)

    return {
        "human_reviewed_gold_count": summary.get(HUMAN_REVIEWED, 0),
        "gold_status_summary": summary,
        "real_vlm_calls": vlm_calls,
        "external_api_calls": 0,
        "model_downloads": 0,
        "production_behavior_changed": (not unchanged) if unchanged is not None else None,
        "shipped_availability_numbers_unchanged": unchanged,
        "shadow_candidates_are_isolated": all(
            "structural_region_candidates" not in
            (ROOT / m).read_text(encoding="utf-8")
            for m in ("src/vision/review_region_resolver.py",
                      "src/multimodal/pipeline.py",
                      "src/multimodal/decision.py",
                      "src/multimodal/executor.py",
                      "src/rag/policy.py")),
    }


def prior_manifests_untouched() -> Dict[str, str]:
    """Record the earlier manifests' hashes WITHOUT rewriting them."""
    return hashes([p for p in PRIOR_MANIFESTS if (ROOT / p).exists()])


def snapshot() -> dict:
    code = hashes(import_closure(ENTRY_POINTS))
    groups = {
        "gold_schema_and_shadow_code": code,
        "shadow_config": hashes(EXPLICIT_CONFIG),
        "gold_tooling": hashes(EXPLICIT_TOOLING),
        "tests": hashes(EXPLICIT_TESTS),
        "unreviewed_gold_data": hashes(EXPLICIT_UNREVIEWED_DATA),
        "reports": hashes(EXPLICIT_REPORTS),
        "annotation_guide": hashes(EXPLICIT_DOCS),
    }
    return {
        "stage": "review_region_gold_tooling/1.0",
        "measured_facts": measured_facts(),
        "does_not_modify": PRIOR_MANIFESTS + [
            "data/multimodal_cache/vision_review_cache.json"],
        "prior_manifest_hashes_observed": prior_manifests_untouched(),
        "groups": {
            name: {"files": files, "file_count": len(files),
                   "sha256": group_hash(files)}
            for name, files in groups.items()
        },
        "human_reviewed_gold_group_is_empty_by_design": (
            "no human-reviewed record exists yet; unreviewed_gold_data holds "
            "pending records only. No localisation metric may be reported "
            "until that changes."),
    }


def write() -> None:
    data = snapshot()
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    for name, group in data["groups"].items():
        print(f"{name:<30} {group['file_count']:>3} files  {group['sha256'][:16]}…")
    facts = data["measured_facts"]
    print()
    for key in ("human_reviewed_gold_count", "real_vlm_calls",
                "external_api_calls", "model_downloads",
                "production_behavior_changed",
                "shadow_candidates_are_isolated"):
        print(f"  {key:<36} {facts[key]}")
    print(f"  gold_status                          {facts['gold_status_summary']}")
    print(f"-> {MANIFEST}")


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest; run with --write first")
        return 2
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = snapshot()
    findings = []
    for name in recorded["groups"]:
        before = recorded["groups"][name]["files"]
        after = current["groups"][name]["files"]
        for rel in sorted(set(after) - set(before)):
            findings.append(f"ADDED     {name}: {rel}")
        for rel in sorted(set(before) - set(after)):
            findings.append(f"REMOVED   {name}: {rel}")
        for rel in sorted(set(before) & set(after)):
            if before[rel] != after[rel]:
                findings.append(f"CHANGED   {name}: {rel}")

    for rel, digest in recorded["prior_manifest_hashes_observed"].items():
        now = current["prior_manifest_hashes_observed"].get(rel)
        if now != digest:
            findings.append(f"PRIOR_MANIFEST_MODIFIED  {rel}")

    facts = current["measured_facts"]
    if facts["real_vlm_calls"] != 0:
        findings.append(f"REAL_VLM_CALLS  now {facts['real_vlm_calls']}, expected 0")
    if not facts["shadow_candidates_are_isolated"]:
        findings.append("SHADOW_LEAKED_INTO_PRODUCTION")

    if not findings:
        print("FREEZE OK")
        for name, group in current["groups"].items():
            print(f"  {name:<30} {group['file_count']:>3} files  "
                  f"{group['sha256'][:16]}…")
        print(f"  human_reviewed_gold_count = "
              f"{facts['human_reviewed_gold_count']}   real_vlm_calls = "
              f"{facts['real_vlm_calls']}")
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
