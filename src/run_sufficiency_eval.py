from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.evaluator import ROOT
from src.models import IntentPlan, SufficiencyCase, SufficiencyJudgment
from src.sufficiency_judge import judge_sufficiency


def load_cases(path: str | None = None) -> list[SufficiencyCase]:
    case_path = Path(path) if path else (ROOT / "data" / "sufficiency_cases.jsonl")
    rows = []
    with case_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(SufficiencyCase.model_validate_json(line))
    return rows


def _judge_with_retry(case: SufficiencyCase, retries: int, backoff: float):
    plan = IntentPlan(normalized_query=case.query, user_goal=case.query, subtasks=case.subtasks)
    last_err = None
    for attempt in range(retries + 1):
        try:
            return judge_sufficiency(case.query, plan, case.tool_results, case.extra_context), None
        except Exception as e:  # noqa: BLE001 - a failed/unparsable call is itself a bad case
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return None, str(last_err)


def main() -> None:
    parser = argparse.ArgumentParser(description="用真实 LLM 跑 03_result_sufficiency 充分度/拒答判断评测")
    parser.add_argument("--dataset", default=None, help="默认 data/sufficiency_cases.jsonl")
    parser.add_argument("--prefix", default="sufficiency_", help="输出文件名前缀")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    records = []
    correct_sufficient = 0
    correct_next_action = 0
    correct_both = 0
    failures = []

    for i, case in enumerate(cases, start=1):
        print(f"\r  {i}/{len(cases)} {case.id}", end="", flush=True)
        judgment, error = _judge_with_retry(case, args.retries, args.backoff)

        if judgment is None:
            pred_sufficient = None
            pred_next_action = None
        else:
            pred_sufficient = judgment.sufficient
            pred_next_action = judgment.next_action

        sufficient_ok = pred_sufficient == case.gold_sufficient
        next_action_ok = pred_next_action in case.gold_next_action
        both_ok = sufficient_ok and next_action_ok

        correct_sufficient += int(sufficient_ok)
        correct_next_action += int(next_action_ok)
        correct_both += int(both_ok)

        record = {
            "id": case.id,
            "description": case.description,
            "query": case.query,
            "tags": case.tags,
            "gold_sufficient": case.gold_sufficient,
            "gold_next_action": case.gold_next_action,
            "pred_sufficient": pred_sufficient,
            "pred_next_action": pred_next_action,
            "judgment": judgment.model_dump(mode="json") if judgment else None,
            "error": error,
            "sufficient_ok": sufficient_ok,
            "next_action_ok": next_action_ok,
        }
        records.append(record)
        if not both_ok:
            failures.append(record)
    print()

    n = len(cases)
    report = {
        "case_count": n,
        "sufficient_accuracy": correct_sufficient / n,
        "next_action_accuracy": correct_next_action / n,
        "both_accuracy": correct_both / n,
        "failures": failures,
    }

    (output_dir / f"{args.prefix}report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{args.prefix}predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lines = [
        "# 真实 LLM 充分度/拒答判断评测报告",
        "",
        f"> 数据集：{args.dataset or 'data/sufficiency_cases.jsonl'}，样本数：{n}。"
        "验证 prompts/03_result_sufficiency.md 这一步（此前从未被任何代码调用过）。",
        "",
        f"- sufficient 判断准确率：{report['sufficient_accuracy']:.1%}",
        f"- next_action 判断准确率（在允许集合内即算对）：{report['next_action_accuracy']:.1%}",
        f"- 两者都对：{report['both_accuracy']:.1%}",
        "",
    ]
    (output_dir / f"{args.prefix}summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
