from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluator import ROOT
from src.vision.bad_case_report import build_report, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="从 outputs/ocr_eval_report.json 生成 Bad Case 报告")
    parser.add_argument("--report", default=str(ROOT / "outputs" / "ocr_eval_report.json"))
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    cases = build_report(report)
    write_outputs(cases, ROOT / "outputs", ROOT / "interview")

    attributed = sum(1 for c in cases if c.failure_type)
    print(f"收集到 {len(cases)} 条 Bad Case，其中 {attributed} 条已人工归因。")
    print("已写入 outputs/ocr_bad_cases.jsonl 和 interview/ocr_bad_cases.md")


if __name__ == "__main__":
    main()
