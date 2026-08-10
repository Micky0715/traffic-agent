from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from src.models import Intent, IntentPlan, ToolResult
from src.router_v3 import route_v3
from src.tools import execute_task
from src.validator import validate_and_repair


def _execute_independent(plan: IntentPlan) -> List[ToolResult]:
    executable = [t for t in plan.subtasks if t.tool.value != "none" and not t.depends_on]
    results: List[ToolResult] = []
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(executable)))) as pool:
        futures = {pool.submit(execute_task, task): task for task in executable}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _extract_numeric_values(data: Any) -> List[float]:
    values: List[float] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (int, float)) and key not in {"page"}:
                values.append(float(value))
            elif isinstance(value, (dict, list)):
                values.extend(_extract_numeric_values(value))
    elif isinstance(data, list):
        for item in data:
            values.extend(_extract_numeric_values(item))
    return values


def _condition_satisfied(task, results: List[ToolResult]) -> bool:
    if not task.condition:
        return True
    threshold_raw = task.slots.get("threshold")
    if threshold_raw is None:
        return False
    import re
    match = re.search(r"\d+(?:\.\d+)?", str(threshold_raw))
    if not match:
        return False
    threshold = float(match.group())
    upstream = [r for r in results if r.task_id in task.depends_on and r.ok]
    values = []
    for result in upstream:
        values.extend(_extract_numeric_values(result.data))
    return any(value > threshold for value in values)


def _execute_dependent(plan: IntentPlan, results: List[ToolResult]) -> List[ToolResult]:
    completed = {r.task_id for r in results if r.ok}
    pending = [t for t in plan.subtasks if t.tool.value != "none" and t.depends_on]
    for task in pending:
        if all(dep in completed for dep in task.depends_on) and _condition_satisfied(task, results):
            results.append(execute_task(task))
    return results


def synthesize(plan: IntentPlan, results: List[ToolResult]) -> str:
    if plan.needs_clarification:
        return plan.clarification_question or "请补充查询条件。"

    by_task: Dict[str, ToolResult] = {r.task_id: r for r in results}
    lines = []
    for task in plan.subtasks:
        if task.tool.value == "none":
            continue
        result = by_task.get(task.id)
        if not result:
            lines.append(f"[{task.id}] 未执行：依赖条件未满足。")
        elif not result.ok:
            lines.append(f"[{task.id}] 查询失败：{result.error}。")
        else:
            lines.append(f"[{task.id}] {task.intent.value} 查询结果：{result.data}")

    if any(t.intent == Intent.FAULT_DIAGNOSIS for t in plan.subtasks):
        ok_results = [r for r in results if r.ok and r.tool.value != "none"]
        if ok_results:
            lines.append("处置建议：先核对实时告警与规范要求的一致性，再按设备手册执行停机、复位或检修；建议部分不作为工具事实。")
        else:
            lines.append("当前证据不足，暂不输出确定性故障结论。")
    return "\n".join(lines) if lines else "该问题不需要外部工具。"


def run_pipeline(query: str) -> Dict[str, Any]:
    plan = validate_and_repair(route_v3(query))
    if plan.needs_clarification:
        return {"plan": plan.model_dump(mode="json"), "results": [], "answer": synthesize(plan, [])}
    results = _execute_independent(plan)
    results = _execute_dependent(plan, results)
    return {
        "plan": plan.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in results],
        "answer": synthesize(plan, results),
    }
