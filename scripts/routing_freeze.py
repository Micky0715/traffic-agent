"""Freeze / verify the routing experiment.

An experiment is reproducible only if everything that can change its numbers is
pinned. For routing that is six different kinds of input, and they fail
differently, so they are hashed as separate groups rather than one digest:

  routing_runtime_code       what decides the route
  routing_config_and_prompts what that decision reads
  routing_evaluator          what turns a route into a number
  routing_datasets           what it is scored against
  historical_llm_predictions the only real-LLM evidence on record
  runtime_environment        versions the above ran under

Separate groups mean a diff says WHICH kind of thing moved. A single combined
hash would say only "something changed", which is the failure the earlier
whole-tree vision freeze had.

    python scripts/routing_freeze.py --write
    python scripts/routing_freeze.py --verify

LIMITS OF STATIC ANALYSIS — stated here and repeated in the manifest:

    代码依赖范围来自静态 import 分析，动态导入、配置加载、Prompt、模型文件和
    subprocess 依赖无法仅靠 import 闭包发现，因此非代码输入采用显式清单补充。

Two concrete instances in this repo:

  llm_router._call_json() loads a prompt by a RUNTIME STRING
  ((ROOT / "prompts" / prompt_file).read_text()). No import closure can see
  that, so the prompt files are declared explicitly and cross-checked against
  the `.md` literals actually present in the runtime code.

  validator.validate_and_repair() is imported by no router — run_eval applies
  it around them — yet it rewrites every plan's tool mapping and can force a
  whole plan to clarify. It is declared as its own entry point rather than
  waiting for a closure to find it, because the closure never will.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "routing_freeze_manifest.json"

# Each routing strategy's entry module. Closures are recorded per entry AND as
# a union: per-entry shows which strategy a change can affect, the union is
# what the freeze check compares.
ROUTING_ENTRY_POINTS = {
    "rule_v1": "src/router_v1.py",
    "rule_v2": "src/router_v2.py",
    "rule_v3": "src/router_v3.py",
    "llm": "src/llm_router.py",
    # Not imported by any router; applied around them. See module docstring.
    "validator": "src/validator.py",
}

# Non-code inputs that static analysis cannot reach.
EXPLICIT_CONFIG_AND_PROMPTS = [
    "configs/routing.yaml",
    "prompts/00_intent_router_v1_bad.md",   # route_with_llm_naive
    "prompts/01_intent_router_v2.md",       # route_with_llm_structured
]

EXPLICIT_EVALUATOR = [
    "src/evaluator.py",
    "src/run_eval.py",
    "src/run_llm_eval.py",
    "scripts/audit_routing_baseline.py",
]

EXPLICIT_DATASETS = [
    "data/eval_cases.jsonl",        # base46
    "data/challenge_cases.jsonl",   # challenge75 (regression set)
]

# The six ORIGINAL prediction files. The converted copies under
# outputs/unified_routing_predictions/ are derived artifacts and deliberately
# excluded — freezing a derivative alongside its source would make a
# regeneration look like tampering.
EXPLICIT_PREDICTIONS = sorted(
    str(p.relative_to(ROOT)).replace("\\", "/")
    for p in (ROOT / "outputs").glob("llm_*predictions.jsonl")
)

STATIC_ANALYSIS_LIMITATION = (
    "代码依赖范围来自静态 import 分析，动态导入、配置加载、Prompt、模型文件和 "
    "subprocess 依赖无法仅靠 import 闭包发现，因此非代码输入采用显式清单补充。"
)


# ---------------------------------------------------------------------------
# closure + dynamic-import detection
# ---------------------------------------------------------------------------

def _analyse(rel_path: str) -> tuple[List[str], List[dict], List[str]]:
    """Returns (first-party imports, unresolved dynamic imports, .md/.yaml/.json
    string literals) for one module."""
    path = ROOT / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: List[str] = []
    unresolved: List[dict] = []
    literals: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            target = node.func
            name = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name in {"import_module", "__import__"}:
                argument = node.args[0] if node.args else None
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    imports.append(argument.value)
                else:
                    # Never guessed at: a dynamic target that cannot be read off
                    # the source is recorded as unknown.
                    unresolved.append({
                        "file": rel_path, "line": node.lineno,
                        "call": name,
                        "detail": "dynamic import target is not a string literal",
                    })
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.endswith((".md", ".yaml", ".yml", ".json", ".jsonl")):
                literals.append(node.value)

    return imports, unresolved, literals


def import_closure(entry: str) -> tuple[List[str], List[dict], List[str]]:
    seen: set[str] = set()
    unresolved: List[dict] = []
    literals: set[str] = set()
    stack = [entry]

    while stack:
        rel = stack.pop()
        if rel in seen or not (ROOT / rel).exists():
            continue
        seen.add(rel)
        imports, module_unresolved, module_literals = _analyse(rel)
        unresolved.extend(module_unresolved)
        literals.update(module_literals)
        for module in imports:
            if not module.startswith("src."):
                continue
            as_module = module.replace(".", "/") + ".py"
            as_package = module.replace(".", "/") + "/__init__.py"
            if (ROOT / as_module).exists():
                stack.append(as_module)
            elif (ROOT / as_package).exists():
                stack.append(as_package)
            else:
                unresolved.append({
                    "file": rel, "line": None, "call": "import",
                    "detail": f"first-party module {module!r} could not be resolved to a file",
                })

    return sorted(seen), unresolved, sorted(literals)


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

def _hashes(paths: List[str]) -> Dict[str, str]:
    out = {}
    for rel in paths:
        path = ROOT / rel
        if path.exists():
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _group_hash(hashes: Dict[str, str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(hashes):
        digest.update(rel.encode())
        digest.update(hashes[rel].encode())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _environment_cached() -> str:
    return json.dumps(_environment_uncached(), ensure_ascii=False)


def _environment() -> dict:
    """Memoized: snapshot() runs it on every call and the git subprocesses
    dominate runtime in the test suite, where verify() is invoked dozens of
    times within one process."""
    return json.loads(_environment_cached())


def _environment_uncached() -> dict:
    versions = {}
    for name in ["pydantic", "openai", "yaml", "langgraph"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001
            versions[name] = None

    def git(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  cwd=ROOT, timeout=30).stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "dependency_versions": versions,
        "git_head": git("git", "rev-parse", "HEAD"),
        "git_dirty": bool(git("git", "status", "--porcelain")),
    }


def snapshot() -> dict:
    per_entry = {}
    union: set[str] = set()
    unresolved: List[dict] = []
    literals: set[str] = set()

    for name, entry in ROUTING_ENTRY_POINTS.items():
        files, entry_unresolved, entry_literals = import_closure(entry)
        per_entry[name] = {"entry": entry, "files": files, "file_count": len(files)}
        union.update(files)
        unresolved.extend(entry_unresolved)
        literals.update(entry_literals)

    # Cross-check: a resource named in runtime code but not declared explicitly
    # is exactly the gap static analysis leaves.
    declared = set(EXPLICIT_CONFIG_AND_PROMPTS)
    for literal in sorted(literals):
        if not literal.endswith(".md"):
            continue
        candidate = f"prompts/{literal}"
        if (ROOT / candidate).exists() and candidate not in declared:
            unresolved.append({
                "file": "routing runtime code", "line": None, "call": "runtime path load",
                "detail": (f"{candidate!r} is referenced by a string literal in routing "
                           "code but is not in the explicit config/prompt list"),
            })

    code = _hashes(sorted(union))
    config = _hashes(EXPLICIT_CONFIG_AND_PROMPTS)
    evaluator = _hashes(EXPLICIT_EVALUATOR)
    datasets = _hashes(EXPLICIT_DATASETS)
    predictions = _hashes(EXPLICIT_PREDICTIONS)

    return {
        "static_analysis_limitation": STATIC_ANALYSIS_LIMITATION,
        "groups": {
            "routing_runtime_code": {
                "source": "static import closure of the routing entry points",
                "entry_points": ROUTING_ENTRY_POINTS,
                "per_entry_closure": per_entry,
                "union_files": code,
                "file_count": len(code),
                "sha256": _group_hash(code),
            },
            "routing_config_and_prompts": {
                "source": ("explicit list: prompts are loaded by a runtime string and "
                           "cannot be found by import analysis"),
                "files": config, "file_count": len(config), "sha256": _group_hash(config),
            },
            "routing_evaluator": {
                "source": "explicit list",
                "files": evaluator, "file_count": len(evaluator),
                "sha256": _group_hash(evaluator),
            },
            "routing_datasets": {
                "source": "explicit list",
                "files": datasets, "file_count": len(datasets),
                "sha256": _group_hash(datasets),
                "case_counts": {
                    rel: sum(1 for line in (ROOT / rel).read_text(encoding="utf-8").splitlines()
                             if line.strip())
                    for rel in EXPLICIT_DATASETS if (ROOT / rel).exists()},
            },
            "historical_llm_predictions": {
                "source": ("the six ORIGINAL prediction files; converted copies under "
                           "outputs/unified_routing_predictions/ are excluded as derivatives"),
                "files": predictions, "file_count": len(predictions),
                "sha256": _group_hash(predictions),
            },
        },
        "runtime_environment": _environment(),
        "unresolved_dependencies": unresolved,
    }


# ---------------------------------------------------------------------------
# write / verify
# ---------------------------------------------------------------------------

GROUP_CHANGE_LABEL = {
    "routing_runtime_code": "CHANGED",
    "routing_config_and_prompts": "CONFIG_CHANGED",
    "routing_evaluator": "EVALUATOR_CHANGED",
    "routing_datasets": "DATASET_CHANGED",
    "historical_llm_predictions": "PREDICTION_CHANGED",
}


def _group_files(group: dict) -> Dict[str, str]:
    return group.get("union_files") or group.get("files") or {}


def write() -> None:
    data = snapshot()
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, group in data["groups"].items():
        print(f"{name:<30} {group['file_count']:>3} files  {group['sha256'][:16]}…")
    if data["unresolved_dependencies"]:
        print(f"\nUNRESOLVED_DEPENDENCY x{len(data['unresolved_dependencies'])}")
        for item in data["unresolved_dependencies"]:
            print(f"  {item['file']}:{item['line']} {item['detail']}")
    else:
        print("\nunresolved dependencies: none")
    print(f"\n-> {MANIFEST}")


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest; run with --write first")
        return 2
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = snapshot()
    findings: List[str] = []

    for name, label in GROUP_CHANGE_LABEL.items():
        before = _group_files(recorded["groups"][name])
        after = _group_files(current["groups"][name])
        for rel in sorted(set(after) - set(before)):
            findings.append(f"ADDED             {name}: {rel}")
        for rel in sorted(set(before) - set(after)):
            findings.append(f"REMOVED           {name}: {rel}")
        for rel in sorted(set(before) & set(after)):
            if before[rel] != after[rel]:
                findings.append(f"{label:<17} {name}: {rel}")

    for item in current["unresolved_dependencies"]:
        findings.append(f"UNRESOLVED_DEPENDENCY  {item['file']}:{item['line']} {item['detail']}")

    if not findings:
        print("FREEZE OK")
        for name, group in current["groups"].items():
            print(f"  {name:<30} {group['file_count']:>3} files  {group['sha256'][:16]}…")
        return 0

    print("FREEZE VIOLATION")
    for line in findings:
        print(f"  {line}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze / verify the routing experiment")
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
