"""Freeze / verify the multimodal review stage.

Its own manifest. The vision, table and routing manifests are NOT touched and
NOT re-hashed by this script — a stage that rewrote its predecessors' records
could hide a change in either.

Eight groups, because they fail differently:

  multimodal_runtime_code   the loop
  multimodal_config         thresholds, call limits, the two live switches
  multimodal_prompts        what the model was actually asked
  multimodal_images         the source pages
  multimodal_crops          the regions that were cut
  multimodal_cache          recorded real replies, with their provenance
  multimodal_evaluator      the eval script and the registered cases
  multimodal_gold           human-reviewed field labels for this stage — EMPTY

    python scripts/multimodal_freeze.py --write
    python scripts/multimodal_freeze.py --verify
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
MANIFEST = ROOT / "outputs" / "multimodal_freeze_manifest.json"

ENTRY_POINTS = [
    "src/multimodal/cache.py",
    "src/multimodal/config.py",
    "src/multimodal/crop.py",
    "src/multimodal/decision.py",
    "src/multimodal/executor.py",
    "src/multimodal/fusion.py",
    "src/multimodal/pipeline.py",
    "src/multimodal/reasons.py",
    "src/multimodal/response_schema.py",
    "src/multimodal/schemas.py",
    "src/multimodal/triggers.py",
]

EXPLICIT_CONFIG = ["configs/multimodal.yaml"]
EXPLICIT_PROMPTS = ["prompts/10_vision_field_review.md"]
EXPLICIT_EVALUATOR = [
    "scripts/multimodal_review_eval.py",
    "data/multimodal_cases_registered.json",
]
# No human-reviewed field gold for this stage. Recorded as empty on purpose:
# a report wanting to quote a VLM recovery RATE has to find this non-empty.
EXPLICIT_GOLD: List[str] = []

CROP_DIR = "data/multimodal_cache/crops"
CACHE_FILE = "data/multimodal_cache/vision_review_cache.json"


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


def images_used() -> List[str]:
    """Only the pages the registered cases actually name."""
    registered = ROOT / "data" / "multimodal_cases_registered.json"
    if not registered.exists():
        return []
    cases = json.loads(registered.read_text(encoding="utf-8"))["cases"]
    return sorted({c["image"] for c in cases if (ROOT / c["image"]).exists()})


def crops() -> List[str]:
    path = ROOT / CROP_DIR
    if not path.exists():
        return []
    return sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                  for p in path.glob("*.png"))


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


def cache_provenance() -> Dict[str, object]:
    path = ROOT / CACHE_FILE
    if not path.exists():
        return {"entries": 0, "recorded_from_live_inference": 0,
                "note": "no cache file"}
    store = json.loads(path.read_text(encoding="utf-8"))
    live = sum(1 for e in store.values()
               if e.get("recorded_from") == "live_inference")
    return {
        "entries": len(store),
        "recorded_from_live_inference": live,
        "models": sorted({e.get("model_name") for e in store.values() if e.get("model_name")}),
        "prompt_versions": sorted({e.get("prompt_version") for e in store.values()
                                   if e.get("prompt_version")}),
        "every_entry_carries_image_and_crop_hash": all(
            e.get("input_image_sha256") and e.get("crop_sha256")
            for e in store.values()),
        "note": ("entries were produced by real live calls and are replayed "
                 "offline; a replay is not inference"),
    }


def snapshot() -> dict:
    code = hashes(import_closure(ENTRY_POINTS))
    config = hashes(EXPLICIT_CONFIG)
    prompts = hashes(EXPLICIT_PROMPTS)
    images = hashes(images_used())
    crop_files = hashes(crops())
    cache = hashes([CACHE_FILE])
    gold = hashes(EXPLICIT_GOLD)
    evaluator = hashes(EXPLICIT_EVALUATOR)
    return {
        "static_analysis_limitation": (
            "代码范围来自静态 import 闭包；配置、Prompt、图片、裁剪图与缓存"
            "无法由 import 分析发现，因此显式列入。"),
        "does_not_touch": ["outputs/vision_freeze_manifest.json",
                           "outputs/table_freeze_manifest.json",
                           "outputs/routing_freeze_manifest.json"],
        "groups": {
            "multimodal_runtime_code": {"source": "static import closure", "files": code,
                                        "file_count": len(code), "sha256": group_hash(code)},
            "multimodal_config": {"source": "explicit list", "files": config,
                                  "file_count": len(config), "sha256": group_hash(config)},
            "multimodal_prompts": {"source": "explicit list", "files": prompts,
                                   "file_count": len(prompts), "sha256": group_hash(prompts)},
            "multimodal_images": {"source": "registered cases", "files": images,
                                  "file_count": len(images), "sha256": group_hash(images)},
            "multimodal_crops": {"source": "explicit glob", "files": crop_files,
                                 "file_count": len(crop_files), "sha256": group_hash(crop_files)},
            "multimodal_cache": {"source": "explicit list", "files": cache,
                                 "file_count": len(cache), "sha256": group_hash(cache),
                                 "provenance": cache_provenance()},
            "multimodal_gold": {"source": "explicit list", "files": gold,
                                "file_count": len(gold), "sha256": group_hash(gold),
                                "human_reviewed": False,
                                "note": ("empty: no human-reviewed field gold for this "
                                         "stage. No VLM recovery RATE may be reported "
                                         "while this group is empty.")},
            "multimodal_evaluator": {"source": "explicit list", "files": evaluator,
                                     "file_count": len(evaluator),
                                     "sha256": group_hash(evaluator)},
        },
        "stage_version": "multimodal/1.0",
    }


LABELS = {
    "multimodal_runtime_code": "CHANGED",
    "multimodal_config": "CONFIG_CHANGED",
    "multimodal_prompts": "PROMPT_CHANGED",
    "multimodal_images": "IMAGE_CHANGED",
    "multimodal_crops": "CROP_CHANGED",
    "multimodal_cache": "CACHE_CHANGED",
    "multimodal_gold": "GOLD_CHANGED",
    "multimodal_evaluator": "EVALUATOR_CHANGED",
}


def write() -> None:
    data = snapshot()
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, group in data["groups"].items():
        print(f"{name:<26} {group['file_count']:>3} files  {group['sha256'][:16]}…")
    prov = data["groups"]["multimodal_cache"]["provenance"]
    print(f"\ncache entries: {prov['entries']}  "
          f"from real live calls: {prov['recorded_from_live_inference']}")
    print(f"models: {prov.get('models')}  prompts: {prov.get('prompt_versions')}")
    print(f"human-reviewed gold: {data['groups']['multimodal_gold']['human_reviewed']}")
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
            print(f"  {name:<26} {group['file_count']:>3} files  {group['sha256'][:16]}…")
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
