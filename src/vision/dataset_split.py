from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = ROOT / "data" / "ocr_pipeline_cases.jsonl"


def load_dev_holdout(path: Path | None = None) -> Tuple[List[dict], List[dict]]:
    """Split data/ocr_pipeline_cases.jsonl by its `split` field.

    Holdout cases must not be used to tune Router thresholds, Prompts, or
    Validator rules before a final report is generated — run_ocr_eval.py's
    --holdout-only mode is meant to be run once, read-only, at the end.
    """
    case_path = path or DEFAULT_CASES_PATH
    dev: List[dict] = []
    holdout: List[dict] = []
    with case_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            (holdout if case.get("split") == "holdout" else dev).append(case)
    return dev, holdout
