from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.evaluator import ROOT, evaluate, load_cases
from src.llm_router import route_with_llm_naive, route_with_llm_structured
from src.models import Intent, IntentPlan, SubTask, ToolName
from src.validator import validate_and_repair


def _error_plan(query: str, err: Exception) -> IntentPlan:
    """A failed API call or unparsable response is a real bad case, not a crash.

    Recording it as a general_qa/none plan makes it show up as a normal set
    mismatch against gold_intents, with the actual error kept in the reason
    field for later inspection.
    """
    return IntentPlan(
        normalized_query=query,
        user_goal=query,
        plan_mode="none",
        subtasks=[SubTask(
            id="T1",
            intent=Intent.GENERAL_QA,
            tool=ToolName.NONE,
            query=query,
            confidence=0.0,
            reason=f"llm_call_or_parse_error: {err}",
        )],
        answer_constraints=[],
    )


def _call_with_retry(call_fn, query: str, retries: int, backoff: float):
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return validate_and_repair(call_fn(query)), None
        except Exception as e:  # noqa: BLE001 - network/parse errors are retried, then recorded
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return _error_plan(query, last_err), str(last_err)


def run_concurrent(call_fn, queries: list[str], workers: int, retries: float, backoff: float, throttle: float):
    """Fetch all plans concurrently, keyed by query text.

    evaluate() (in evaluator.py) is intentionally left untouched: it calls a
    plain router(query) -> IntentPlan sequentially. Concurrency happens here,
    up front, by resolving every unique query in a thread pool; the router
    handed to evaluate() then does an O(1) dict lookup, so the shared,
    rule-router-facing evaluation code never has to know about threading.
    """
    unique_queries = list(dict.fromkeys(queries))
    results: dict[str, IntentPlan] = {}
    records: list[dict] = []
    lock = threading.Lock()
    done_count = 0

    def task(q: str):
        plan, error = _call_with_retry(call_fn, q, retries, backoff)
        if throttle:
            time.sleep(throttle)
        return q, plan, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, q): q for q in unique_queries}
        for fut in as_completed(futures):
            q, plan, error = fut.result()
            with lock:
                results[q] = plan
                records.append({"query": q, "plan": plan.model_dump(mode="json"), "error": error})
                done_count += 1
                print(f"\r  {done_count}/{len(unique_queries)}", end="", flush=True)
    print()
    return results, records


def make_lookup_router(results: dict[str, IntentPlan]):
    def router(query: str) -> IntentPlan:
        return results[query]
    return router


def main() -> None:
    parser = argparse.ArgumentParser(description="用真实 LLM 跑意图路由评测（naive V1 vs structured V2），支持并发")
    parser.add_argument("--mode", choices=["naive", "structured", "both"], default="both")
    parser.add_argument("--dataset", default=None, help="默认 data/eval_cases.jsonl")
    parser.add_argument("--prefix", default="llm_", help="输出文件名前缀")
    parser.add_argument("--limit", type=int, default=None, help="只跑前N条，先小规模验证再跑全量")
    parser.add_argument("--workers", type=int, default=6, help="并发线程数")
    parser.add_argument("--retries", type=int, default=2, help="单次调用失败后的重试次数")
    parser.add_argument("--backoff", type=float, default=1.5, help="重试之间的基础退避秒数")
    parser.add_argument("--throttle", type=float, default=0.0, help="每次调用完成后额外等待的秒数，用于避免限流")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    queries = [c.query for c in cases]

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    call_fns = {}
    if args.mode in {"naive", "both"}:
        call_fns["llm_v1_naive"] = route_with_llm_naive
    if args.mode in {"structured", "both"}:
        call_fns["llm_v2_structured"] = route_with_llm_structured

    lines = [
        "# 真实 LLM 路由评测报告",
        "",
        f"> 数据集：{args.dataset or 'data/eval_cases.jsonl'}，样本数：{len(cases)}，并发数：{args.workers}，"
        "模型与网关来自 .env（MODEL_NAME / OPENAI_BASE_URL）。",
        "",
    ]
    for name, call_fn in call_fns.items():
        print(f"[{name}] 并发调用中...")
        results, records = run_concurrent(call_fn, queries, args.workers, args.retries, args.backoff, args.throttle)
        router = make_lookup_router(results)
        report = evaluate(router, cases)

        (output_dir / f"{args.prefix}{name}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        fields = ["id", "query", "gold_intents", "pred_intents", "gold_tools", "pred_tools",
                   "gold_clarify", "pred_clarify", "gold_mode", "pred_mode",
                   "expected_slots", "predicted_slots", "tags"]
        with (output_dir / f"{args.prefix}{name}_failures.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report["failures"])

        with (output_dir / f"{args.prefix}{name}_predictions.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        error_count = sum(1 for r in records if r["error"])
        lines += [
            f"## {name}",
            f"- 样本数：{report['case_count']}（去重后实际调用 {len(records)} 次，失败/异常 {error_count} 次）",
            f"- 意图集合完全匹配：{report['intent_set_exact']:.1%}",
            f"- 意图 micro-F1：{report['intent_micro_f1']:.1%}",
            f"- 工具集合完全匹配：{report['tool_set_exact']:.1%}",
            f"- 工具 micro-F1：{report['tool_micro_f1']:.1%}",
            f"- 澄清判断准确率：{report['clarification_accuracy']:.1%}",
            f"- 执行模式准确率：{report['plan_mode_accuracy']:.1%}",
            f"- 必需槽位准确率：{report['required_slot_accuracy']:.1%}",
            f"- 端到端用例成功率：{report['end_to_end_case_success']:.1%}",
            "",
        ]

    (output_dir / f"{args.prefix}eval_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
