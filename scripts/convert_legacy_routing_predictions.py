"""Convert the frozen historical LLM predictions to UnifiedIntentPlan.

Offline and read-only with respect to the originals: reads the files named in
the freeze manifest, writes converted copies to a new directory, and never
touches the source bytes. Zero API calls — the predictions are already on disk.

Failures are reported per record with the reason. Nothing is silently dropped
and no record is replaced by a placeholder plan.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import IntentPlan  # noqa: E402
from src.routing import (  # noqa: E402
    SchemaAdaptationError, legacy_request_id, to_unified,
)

MANIFEST = ROOT / "outputs" / "routing_predictions_freeze_manifest.json"
OUT_DIR = ROOT / "outputs" / "unified_routing_predictions"
SUMMARY = OUT_DIR / "_conversion_summary.json"

DATASET_FILES = {
    "base46": ROOT / "data" / "eval_cases.jsonl",
    "challenge75": ROOT / "data" / "challenge_cases.jsonl",
}


def load_case_ids(dataset_name: str) -> dict:
    """query -> case_id. Empty when the dataset file no longer exists."""
    path = DATASET_FILES.get(dataset_name)
    if path is None or not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            case = json.loads(line)
            out[case["query"]] = case["id"]
    return out


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("freeze manifest missing — run scripts/freeze_routing_predictions.py")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    totals = Counter()
    reason_counts = Counter()
    failures = []
    per_file = []

    for entry in manifest["files"]:
        source = ROOT / entry["path"]
        raw = source.read_bytes()
        # Guard against converting something that changed after freezing.
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise SystemExit(f"{entry['path']} no longer matches its frozen sha256")

        dataset_name = entry["dataset_name"]
        case_ids = load_case_ids(dataset_name)
        strategy = "llm"

        converted_rows = []
        file_stats = Counter()

        for index, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            totals["total"] += 1
            file_stats["total"] += 1
            query = row.get("query", "")
            case_id = case_ids.get(query)
            if case_id is None:
                # The original 10-case adversarial set no longer exists, so its
                # rows have no case id to recover. Row position is used as a
                # stable placeholder rather than inventing an id.
                case_id = f"row{index:03d}"
                file_stats["case_id_from_row_index"] += 1

            if row.get("error"):
                totals["source_error"] += 1
                file_stats["source_error"] += 1
                failures.append({
                    "file": entry["path"], "dataset_name": dataset_name,
                    "case_id": case_id, "query": query,
                    "stage": "source_record", "error": str(row["error"])[:300],
                })
                continue

            try:
                legacy = IntentPlan.model_validate(row["plan"])
            except Exception as exc:  # noqa: BLE001
                totals["failed"] += 1
                file_stats["failed"] += 1
                failures.append({
                    "file": entry["path"], "dataset_name": dataset_name,
                    "case_id": case_id, "query": query,
                    "stage": "legacy_parse", "error": f"{type(exc).__name__}: {exc}"[:300],
                })
                continue

            try:
                unified = to_unified(
                    legacy,
                    original_query=query,
                    source_strategy=strategy,
                    request_id=legacy_request_id(dataset_name, case_id),
                    # The frozen files record neither, so both stay None and
                    # the adapter stamps the matching *_UNAVAILABLE_LEGACY code.
                    model_info=None,
                    prompt_version=None,
                )
            except SchemaAdaptationError as exc:
                totals["failed"] += 1
                file_stats["failed"] += 1
                failures.append({
                    "file": entry["path"], "dataset_name": dataset_name,
                    "case_id": case_id, "query": query,
                    "stage": "to_unified", "error": str(exc)[:300],
                })
                continue

            totals["converted"] += 1
            file_stats["converted"] += 1
            if unified.model_info is None:
                totals["missing_model_info"] += 1
            if all(t.query_span is None for t in unified.subtasks):
                totals["missing_query_span"] += 1
            for code in unified.reason_codes:
                reason_counts[code.value] += 1
            for task in unified.subtasks:
                for code in task.reason_codes:
                    reason_counts[code.value] += 1
            totals[f"decision_{unified.decision}"] += 1

            converted_rows.append({
                "dataset_name": dataset_name,
                "case_id": case_id,
                "source_file": entry["path"],
                "plan": unified.model_dump(mode="json"),
            })

        out_path = OUT_DIR / source.name.replace(".jsonl", ".unified.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for record in converted_rows:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        per_file.append({"source": entry["path"], "output": str(out_path.relative_to(ROOT)).replace("\\", "/"),
                         **dict(file_stats)})

    summary = {
        "note": ("Offline conversion of frozen historical LLM predictions. "
                 "Source files unmodified; zero API calls."),
        "totals": dict(totals),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "per_file": per_file,
        "failures": failures,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"total        {totals['total']}")
    print(f"converted    {totals['converted']}")
    print(f"failed       {totals['failed']}")
    print(f"source_error {totals['source_error']}")
    print(f"missing model_info  {totals['missing_model_info']}")
    print(f"missing query_span  {totals['missing_query_span']}")
    print("decisions:", {k.replace("decision_", ""): v for k, v in totals.items()
                         if k.startswith("decision_")})
    print("reason codes:")
    for code, count in sorted(reason_counts.items()):
        print(f"  {code:<40} {count}")
    if failures:
        print(f"\nfailures ({len(failures)}):")
        for f in failures[:12]:
            print(f"  [{f['dataset_name']}/{f['case_id']}] {f['stage']}: {f['error'][:120]}")
        if len(failures) > 12:
            print(f"  ... and {len(failures) - 12} more (see {SUMMARY.name})")
    print(f"\n-> {OUT_DIR}")


if __name__ == "__main__":
    main()
