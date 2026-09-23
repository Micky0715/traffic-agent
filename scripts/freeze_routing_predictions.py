"""Freeze the historical real-LLM routing predictions.

These six files are the only real-LLM evidence this project has for the routing
chain, and the phase-2 work will change how plans are represented. Once that
happens they can no longer be compared against fresh output unless their exact
current bytes are on record.

Read-only. Records what is knowable and writes null for what is not — the
files carry no model name, no prompt version and no timestamp, and inventing
any of those would turn an unknown into a fabricated provenance claim.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "outputs" / "routing_predictions_freeze_manifest.json"
OUT_MD = ROOT / "docs" / "routing_predictions_freeze_manifest.md"

# dataset each prediction file was produced against, by file-name convention:
#   llm_*                -> data/eval_cases.jsonl        (46)
#   llm_challenge_*      -> the ORIGINAL 10-case adversarial set
#   llm_challenge_v2_*   -> data/challenge_cases.jsonl   (75)
DATASET_BY_PREFIX = [
    ("llm_challenge_v2_", "challenge75", ROOT / "data" / "challenge_cases.jsonl"),
    ("llm_challenge_", "challenge10_original", None),
    ("llm_", "base46", ROOT / "data" / "eval_cases.jsonl"),
]

REASON_MODEL_UNKNOWN = "MODEL_INFO_UNAVAILABLE_LEGACY"
REASON_PROMPT_UNKNOWN = "PROMPT_VERSION_UNAVAILABLE_LEGACY"
REASON_TIME_UNKNOWN = "GENERATION_TIME_UNAVAILABLE_LEGACY"
REASON_DATASET_MISSING = "SOURCE_DATASET_FILE_MISSING"


def classify(name: str):
    for prefix, dataset, path in DATASET_BY_PREFIX:
        if name.startswith(prefix):
            return dataset, path
    return "unknown", None


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  cwd=ROOT, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    return {
        "head": run("git", "rev-parse", "HEAD") or None,
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def main() -> None:
    entries = []
    for path in sorted((ROOT / "outputs").glob("llm_*predictions.jsonl")):
        raw = path.read_bytes()
        lines = [l for l in raw.decode("utf-8").splitlines() if l.strip()]
        rows = [json.loads(l) for l in lines]
        dataset_name, dataset_path = classify(path.name)

        reason_codes = [REASON_MODEL_UNKNOWN, REASON_PROMPT_UNKNOWN, REASON_TIME_UNKNOWN]
        matched = None
        if dataset_path is not None and dataset_path.exists():
            gold_queries = {}
            for line in dataset_path.open(encoding="utf-8"):
                if line.strip():
                    case = json.loads(line)
                    gold_queries[case["query"]] = case["id"]
            matched = sum(1 for r in rows if r.get("query") in gold_queries)
        else:
            reason_codes.append(REASON_DATASET_MISSING)

        entries.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "case_count": len(rows),
            "dataset_name": dataset_name,
            "dataset_file": (str(dataset_path.relative_to(ROOT)).replace("\\", "/")
                             if dataset_path and dataset_path.exists() else None),
            "cases_matched_to_dataset": matched,
            # Nothing in these files records which model or prompt produced
            # them. Left null rather than guessed.
            "model_name": None,
            "prompt_version": None,
            "generated_at": None,
            "row_fields": sorted(rows[0].keys()) if rows else [],
            "rows_with_error": sum(1 for r in rows if r.get("error")),
            "reason_codes": reason_codes,
        })

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": ("Phase-2 will change how routing plans are represented. These files "
                    "are the only real-LLM routing evidence in the repo; their bytes are "
                    "recorded here so they stay comparable afterwards."),
        "source_tree_sha256": source_hash(),
        "git": git_state(),
        "file_count": len(entries),
        "total_cases": sum(e["case_count"] for e in entries),
        "reason_code_glossary": {
            REASON_MODEL_UNKNOWN: "prediction file records no model name; not inferred",
            REASON_PROMPT_UNKNOWN: "prediction file records no prompt version; not inferred",
            REASON_TIME_UNKNOWN: "prediction file records no generation timestamp",
            REASON_DATASET_MISSING: ("the dataset these predictions were produced against "
                                     "is no longer present in the repo"),
        },
        "files": entries,
    }
    OUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 历史真实 LLM 路由预测 · 冻结清单",
        "",
        f"冻结时间：{manifest['frozen_at']}",
        f"源码树 sha256：`{manifest['source_tree_sha256']}`",
        f"git HEAD：`{manifest['git']['head']}`　工作区有未提交改动：{manifest['git']['dirty']}",
        "",
        "> 这 6 个文件是本仓库路由链路**唯一的真实 LLM 证据**。第二阶段会改变计划的表示方式，",
        "> 因此在动手之前先把它们的字节固定下来。**原文件不被修改，转换结果写入新目录。**",
        "",
        "## 文件",
        "",
        "| 文件 | sha256 | 字节 | 条数 | 数据集 | 可对上 case | 有 error 的行 |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        matched = "—" if e["cases_matched_to_dataset"] is None else str(e["cases_matched_to_dataset"])
        lines.append(f"| `{e['path']}` | `{e['sha256'][:16]}…` | {e['size_bytes']} | "
                     f"{e['case_count']} | {e['dataset_name']} | {matched} | {e['rows_with_error']} |")

    lines += [
        "",
        "## 未知字段（一律 null，不猜测）",
        "",
        "每个文件的行字段只有 `query` / `plan` / `error`，**不包含模型名、prompt 版本或生成时间**。",
        "因此清单中这三项全部记为 `null`，并带上对应 reason_code：",
        "",
        "| reason_code | 含义 |",
        "|---|---|",
    ]
    for code, meaning in manifest["reason_code_glossary"].items():
        lines.append(f"| `{code}` | {meaning} |")

    lines += [
        "",
        "## 需要特别说明的一处",
        "",
        "`llm_challenge_llm_v1_naive_predictions.jsonl` 与 "
        "`llm_challenge_llm_v2_structured_predictions.jsonl` 各 10 条，对应的是**原始 10 条对抗集**。",
        "该数据集文件已不在仓库中——`data/challenge_cases.jsonl` 现在是扩充后的 75 条版本，",
        "文件名前缀 `llm_challenge_v2_` 才对应它。",
        "",
        "因此这两个文件标记 `SOURCE_DATASET_FILE_MISSING`：**预测留存，gold 已失，无法重新评分**，",
        "转换时 case_id 只能用行号占位。这不是可以补救的，补一份 gold 等于重新发明答案。",
        "",
        "## 使用约束",
        "",
        "1. 不修改原 predictions 文件；",
        "2. 转换结果写入 `outputs/unified_routing_predictions/`，不覆盖原文件；",
        "3. 未知字段保持 null，不得事后补写推测值；",
        "4. 任何引用这些数字的场合，必须同时说明模型名与 prompt 版本未知。",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"frozen {len(entries)} files, {manifest['total_cases']} cases")
    for e in entries:
        print(f"  {e['path']:<62} {e['case_count']:>3}  {e['dataset_name']:<22} "
              f"matched={e['cases_matched_to_dataset']}")
    print(f"\n-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
