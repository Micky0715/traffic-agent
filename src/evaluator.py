from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

from src.models import EvaluationCase, IntentPlan


ROOT = Path(__file__).resolve().parents[1]


def load_cases(path: str | None = None) -> List[EvaluationCase]:
    rows = []
    case_path = Path(path) if path else (ROOT / "data" / "eval_cases.jsonl")
    with case_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(EvaluationCase.model_validate_json(line))
    return rows


def _set_f1(pred: Set[str], gold: Set[str]) -> Tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)


def _flatten_leaf_values(value) -> List[str]:
    """Collect leaf primitives from arbitrarily nested slot structures.

    The rule-based routers (V1-V3) emit flat slots like {"asset_id": "A12风机"}.
    A real structured tool-calling prompt nests the same information, e.g.
    {"filters": {"asset_id": "A12风机"}, "time_range": {"type": "relative",
    "value": "最近3天"}}. Both must be scoreable against the same expected_slots,
    so slot matching checks whether the expected value appears anywhere in the
    task's slots, not only under an exact matching top-level key.
    """
    if isinstance(value, dict):
        leaves: List[str] = []
        for v in value.values():
            leaves.extend(_flatten_leaf_values(v))
        return leaves
    if isinstance(value, list):
        leaves = []
        for v in value:
            leaves.extend(_flatten_leaf_values(v))
        return leaves
    if value is None:
        return []
    return [str(value)]


def evaluate(router: Callable[[str], IntentPlan], cases: Iterable[EvaluationCase]) -> Dict:
    cases = list(cases)
    intent_tp = intent_fp = intent_fn = 0
    tool_tp = tool_fp = tool_fn = 0
    exact_intent = exact_tool = clarify_ok = mode_ok = slot_ok_count = all_ok = 0
    failures = []
    tag_stats: Dict[str, Counter] = {}

    for case in cases:
        plan = router(case.query)
        pred_intents = {t.intent.value for t in plan.subtasks}
        pred_tools = {t.tool.value for t in plan.subtasks if t.tool.value != "none"}
        gold_intents = {x.value for x in case.gold_intents}
        gold_tools = {x.value for x in case.gold_tools if x.value != "none"}

        i_tp, i_fp, i_fn = _set_f1(pred_intents, gold_intents)
        t_tp, t_fp, t_fn = _set_f1(pred_tools, gold_tools)
        intent_tp += i_tp; intent_fp += i_fp; intent_fn += i_fn
        tool_tp += t_tp; tool_fp += t_fp; tool_fn += t_fn

        intent_exact = pred_intents == gold_intents
        tool_exact = pred_tools == gold_tools
        clarification_exact = plan.needs_clarification == case.needs_clarification
        mode_exact = plan.plan_mode == case.plan_mode
        predicted_slots = {}
        all_leaf_values: List[str] = []
        for task in plan.subtasks:
            for key, value in task.slots.items():
                predicted_slots.setdefault(key, []).append(value)
            all_leaf_values.extend(_flatten_leaf_values(task.slots))
        norm_values = {v.upper() for v in all_leaf_values}
        slot_exact = True
        for key, expected in case.expected_slots.items():
            if isinstance(expected, list):
                slot_exact = slot_exact and {str(x).upper() for x in expected}.issubset(norm_values)
            else:
                slot_exact = slot_exact and str(expected).upper() in norm_values
        success = intent_exact and tool_exact and clarification_exact and mode_exact and slot_exact

        exact_intent += int(intent_exact)
        exact_tool += int(tool_exact)
        clarify_ok += int(clarification_exact)
        mode_ok += int(mode_exact)
        slot_ok_count += int(slot_exact)
        all_ok += int(success)

        for tag in case.tags:
            stat = tag_stats.setdefault(tag, Counter(total=0, success=0))
            stat["total"] += 1
            stat["success"] += int(success)

        if not success:
            failures.append({
                "id": case.id,
                "query": case.query,
                "gold_intents": sorted(gold_intents),
                "pred_intents": sorted(pred_intents),
                "gold_tools": sorted(gold_tools),
                "pred_tools": sorted(pred_tools),
                "gold_clarify": case.needs_clarification,
                "pred_clarify": plan.needs_clarification,
                "gold_mode": case.plan_mode,
                "pred_mode": plan.plan_mode,
                "expected_slots": case.expected_slots,
                "predicted_slots": predicted_slots,
                "tags": case.tags,
            })

    n = len(cases)
    intent_f1 = 2 * intent_tp / max(1, 2 * intent_tp + intent_fp + intent_fn)
    tool_f1 = 2 * tool_tp / max(1, 2 * tool_tp + tool_fp + tool_fn)
    return {
        "case_count": n,
        "intent_set_exact": exact_intent / n,
        "intent_micro_f1": intent_f1,
        "tool_set_exact": exact_tool / n,
        "tool_micro_f1": tool_f1,
        "clarification_accuracy": clarify_ok / n,
        "plan_mode_accuracy": mode_ok / n,
        "required_slot_accuracy": slot_ok_count / n,
        "end_to_end_case_success": all_ok / n,
        "failures": failures,
        "tag_success": {tag: {"success": c["success"], "total": c["total"], "rate": c["success"] / c["total"]} for tag, c in sorted(tag_stats.items())},
    }
