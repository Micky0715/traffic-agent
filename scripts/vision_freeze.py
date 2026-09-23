"""Freeze / verify the visual-parsing pipeline at the correct scope.

Replaces a defective earlier approach that hashed the ENTIRE src/ tree. That
scope was both too wide and, as it turned out, would also have been too narrow
if narrowed naively:

  too wide — adding src/routing/ (a package the vision chain never imports)
             broke the freeze check, so the check reported a change that was
             not one. A tripwire that fires on unrelated edits stops being
             read, which is worse than not having one.

  too narrow — hashing only src/vision/ would have missed
             src/drawing_extractor.py, src/llm_router.py and src/models.py,
             which the vision chain genuinely imports. Editing llm_router.py
             would then have changed parsing behaviour with the freeze still
             reporting green.

So the scope is neither a directory nor a hand-kept list: it is the transitive
import closure of the declared entry points, recomputed on every run. The
manifest records the resolved file list, so a change in the SCOPE itself shows
up as a diff rather than silently widening or narrowing.

    python scripts/vision_freeze.py --write     # record a new freeze
    python scripts/vision_freeze.py --verify    # check against the manifest
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "vision_freeze_manifest.json"

# Every module the parsing pipeline is entered through. The closure is computed
# from these; nothing below needs maintaining by hand.
ENTRY_POINTS = [
    "src/vision/ocr_engine.py",
    "src/vision/page_router.py",
    "src/vision/field_completeness.py",
    "src/vision/value_validation.py",
    "src/vision/entity_ref.py",
    "src/vision/evidence.py",
    "src/vision/config.py",
    "src/vision/table_structure.py",
    "src/vision/qwen_vl_adapter.py",
    "src/vision/image_quality.py",
    "src/vision/preprocess.py",
    "src/vision/validator.py",
    "src/vision/quality_judge.py",
    "src/vision/chunk_builder.py",
    "src/vision/cache.py",
    "src/vision/base.py",
    "src/vision/schemas.py",
]

# Configuration that changes parsing behaviour without changing any code.
CONFIG_FILES = ["configs/visual_parser.yaml"]


def import_closure(entry_points: list[str]) -> list[str]:
    """Transitive closure over first-party `src.*` imports."""
    seen: set[str] = set()
    stack = list(entry_points)
    while stack:
        rel = stack.pop()
        path = ROOT / rel
        if rel in seen or not path.exists():
            continue
        seen.add(rel)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
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


def file_hashes(paths: list[str]) -> dict:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def combined(hashes: dict) -> str:
    digest = hashlib.sha256()
    for path in sorted(hashes):
        digest.update(path.encode())
        digest.update(hashes[path].encode())
    return digest.hexdigest()


def snapshot() -> dict:
    code_files = import_closure(ENTRY_POINTS)
    code = file_hashes(code_files)
    config = file_hashes(CONFIG_FILES)
    return {
        "scope": ("transitive src.* import closure of the declared vision entry "
                  "points, plus the parser configuration"),
        "entry_points": ENTRY_POINTS,
        "code_files": code,
        "config_files": config,
        "code_file_count": len(code),
        "combined_sha256": combined({**code, **config}),
    }


def write() -> None:
    data = snapshot()
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frozen {data['code_file_count']} code files + {len(data['config_files'])} config")
    print(f"combined sha256: {data['combined_sha256']}")
    outside = [p for p in data["code_files"] if not p.startswith("src/vision/")]
    if outside:
        print("in scope but outside src/vision/ (would be missed by a directory freeze):")
        for path in outside:
            print(f"  {path}")
    print(f"-> {MANIFEST}")


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest; run with --write first")
        return 2
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = snapshot()

    recorded_files = set(recorded["code_files"]) | set(recorded["config_files"])
    current_files = set(current["code_files"]) | set(current["config_files"])

    added = sorted(current_files - recorded_files)
    removed = sorted(recorded_files - current_files)
    changed = sorted(
        p for p in recorded_files & current_files
        if {**recorded["code_files"], **recorded["config_files"]}[p]
        != {**current["code_files"], **current["config_files"]}[p]
    )

    if not (added or removed or changed):
        print("FREEZE OK")
        print(f"  {current['code_file_count']} code files + "
              f"{len(current['config_files'])} config")
        print(f"  combined sha256 {current['combined_sha256']}")
        return 0

    print("FREEZE VIOLATION")
    for path in changed:
        print(f"  CHANGED  {path}")
    for path in added:
        # A new file inside the closure means the pipeline gained a dependency:
        # a scope change, which is exactly what the old whole-tree hash could
        # not distinguish from an edit.
        print(f"  ADDED    {path}  (pipeline gained a dependency)")
    for path in removed:
        print(f"  REMOVED  {path}")
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
