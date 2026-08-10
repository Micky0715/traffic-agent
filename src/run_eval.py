from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluator import ROOT, evaluate, load_cases
from src.router_v1 import route_v1
from src.router_v2 import route_v2
from src.router_v3 import route_v3
from src.validator import validate_and_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", choices=["v1", "v2", "v3", "both", "all"], default="both")
    parser.add_argument("--dataset", default=None, help="JSONL评测集路径；默认使用data/eval_cases.jsonl")
    parser.add_argument("--prefix", default="", help="输出文件名前缀")
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    routers = {}
    if args.router in {"v1", "both", "all"}:
        routers["v1_single_label"] = route_v1
    if args.router in {"v2", "both", "all"}:
        routers["v2_multi_intent"] = lambda q: validate_and_repair(route_v2(q))
    if args.router in {"v3", "all"}:
        routers["v3_guarded"] = lambda q: validate_and_repair(route_v3(q))

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    reports = {}
    for name, router in routers.items():
        report = evaluate(router, cases)
        reports[name] = report
        (output_dir / f"{args.prefix}{name}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output_dir / f"{args.prefix}{name}_failures.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fields = ["id", "query", "gold_intents", "pred_intents", "gold_tools", "pred_tools", "gold_clarify", "pred_clarify", "gold_mode", "pred_mode", "expected_slots", "predicted_slots", "tags"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report["failures"])

    lines = ["# 离线回归评测报告", "", "> 说明：这是合成测试集上的路由回归，不代表真实线上模型指标。真实简历数字必须用实际200条人工标注集复测。", ""]
    for name, r in reports.items():
        lines += [
            f"## {name}",
            f"- 样本数：{r['case_count']}",
            f"- 意图集合完全匹配：{r['intent_set_exact']:.1%}",
            f"- 意图 micro-F1：{r['intent_micro_f1']:.1%}",
            f"- 工具集合完全匹配：{r['tool_set_exact']:.1%}",
            f"- 工具 micro-F1：{r['tool_micro_f1']:.1%}",
            f"- 澄清判断准确率：{r['clarification_accuracy']:.1%}",
            f"- 执行模式准确率：{r['plan_mode_accuracy']:.1%}",
            f"- 必需槽位准确率：{r['required_slot_accuracy']:.1%}",
            f"- 端到端用例成功率：{r['end_to_end_case_success']:.1%}",
            "",
        ]
    (output_dir / f"{args.prefix}eval_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
