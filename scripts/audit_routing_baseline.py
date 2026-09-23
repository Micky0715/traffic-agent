"""Phase-1 audit of the intent -> tool routing chain.

Read-only. Computes what the current gold schema actually supports, marks the
rest uncomputable rather than inventing it, and classifies end-to-end failures
into the ten categories from the task spec.

No API calls: the LLM side is read from prediction files already on disk.
Nothing here reads a case id or a gold label to make a routing decision — gold
is used only for scoring, after the fact.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluator import load_cases  # noqa: E402
from src.router_v1 import route_v1  # noqa: E402
from src.router_v2 import route_v2  # noqa: E402
from src.router_v3 import route_v3  # noqa: E402
from src.validator import validate_and_repair  # noqa: E402

OUT = ROOT / "outputs" / "routing_phase1_audit.json"

DATASETS = {
    "base46": ROOT / "data" / "eval_cases.jsonl",
    "challenge75": ROOT / "data" / "challenge_cases.jsonl",
}
LLM_PREDICTIONS = {
    "base46": ROOT / "outputs" / "llm_llm_v2_structured_predictions.jsonl",
    "challenge75": ROOT / "outputs" / "llm_challenge_v2_llm_v2_structured_predictions.jsonl",
}

ENTITY_SLOTS = {"asset_id", "regulation_code", "drawing_no"}
TIME_SLOTS = {"time_range"}


def norm(value) -> set:
    if isinstance(value, list):
        return {str(v).upper() for v in value}
    return {str(value).upper()}


def leaf_values(value) -> list:
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(leaf_values(v))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(leaf_values(v))
        return out
    return [] if value is None else [str(value)]


def plan_view(plan) -> dict:
    """Everything the audit needs from a plan, as plain data."""
    subtasks = plan.subtasks if hasattr(plan, "subtasks") else plan.get("subtasks", [])
    out = {"intents": [], "tools": [], "slot_values": set(), "slots_by_intent": defaultdict(set),
           "subtask_count": 0, "depends_on": 0}
    for task in subtasks:
        if hasattr(task, "intent"):
            intent = task.intent.value
            tool = task.tool.value
            slots = task.slots
            depends = task.depends_on
        else:
            intent = task.get("intent")
            tool = task.get("tool")
            slots = task.get("slots") or {}
            depends = task.get("depends_on") or []
        out["intents"].append(intent)
        out["tools"].append(tool)
        values = {v.upper() for v in leaf_values(slots)}
        out["slot_values"] |= values
        out["slots_by_intent"][intent] |= values
        out["subtask_count"] += 1
        out["depends_on"] += len(depends or [])
    if hasattr(plan, "needs_clarification"):
        out["clarify"] = plan.needs_clarification
        out["plan_mode"] = plan.plan_mode
    else:
        out["clarify"] = plan.get("needs_clarification", False)
        out["plan_mode"] = plan.get("plan_mode")
    return out


def score(view: dict, case) -> dict:
    gold_intents = {x.value for x in case.gold_intents}
    gold_tools = {x.value for x in case.gold_tools if x.value != "none"}
    pred_intents = set(view["intents"])
    pred_tools = {t for t in view["tools"] if t != "none"}

    slot_ok = True
    missing_slot_keys = []
    for key, expected in case.expected_slots.items():
        if not norm(expected).issubset(view["slot_values"]):
            slot_ok = False
            missing_slot_keys.append(key)

    return {
        "gold_intents": sorted(gold_intents), "pred_intents": sorted(pred_intents),
        "gold_tools": sorted(gold_tools), "pred_tools": sorted(pred_tools),
        "intent_exact": pred_intents == gold_intents,
        "tool_exact": pred_tools == gold_tools,
        "clarify_exact": view["clarify"] == case.needs_clarification,
        "mode_exact": view["plan_mode"] == case.plan_mode,
        "slot_exact": slot_ok,
        "missing_slot_keys": missing_slot_keys,
        "gold_clarify": case.needs_clarification, "pred_clarify": view["clarify"],
        "gold_mode": case.plan_mode, "pred_mode": view["plan_mode"],
        "gold_intent_count": len(gold_intents), "pred_subtask_count": view["subtask_count"],
        "over_routing": bool(pred_tools - gold_tools),
        "under_routing": bool(gold_tools - pred_tools),
        "success": (pred_intents == gold_intents and pred_tools == gold_tools
                    and view["clarify"] == case.needs_clarification
                    and view["plan_mode"] == case.plan_mode and slot_ok),
    }


def classify_failure(s: dict, case) -> list:
    """The ten failure categories from the spec. A case may hit several."""
    labels = []
    gold_i, pred_i = set(s["gold_intents"]), set(s["pred_intents"])

    if s["intent_exact"] and not s["tool_exact"]:
        labels.append("1_intent_ok_tool_wrong")
    if s["intent_exact"] and s["tool_exact"] and not s["slot_exact"]:
        labels.append("2_intent_tool_ok_args_wrong")
    if pred_i < gold_i:
        labels.append("3_multi_intent_under_split")
    if pred_i > gold_i:
        labels.append("4_single_intent_over_split")

    entity_keys = [k for k in s["missing_slot_keys"] if k in ENTITY_SLOTS]
    time_keys = [k for k in s["missing_slot_keys"] if k in TIME_SLOTS]
    if entity_keys:
        labels.append("5_entity_missing")
    if s["gold_clarify"] and not s["pred_clarify"]:
        labels.append("6_ambiguity_not_clarified")
    if time_keys:
        labels.append("7_time_range_lost")
    if not s["mode_exact"] and (s["gold_mode"] in {"conditional", "serial"}
                                or s["pred_mode"] in {"conditional", "serial"}):
        labels.append("8_dependency_wrong")
    return labels


def load_llm_predictions(path: Path) -> dict:
    """query -> plan dict, plus an error marker. Keyed on query because the
    prediction files predate case ids being written out."""
    out = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["query"]] = row
    return out


report = {
    "note": ("Phase-1 audit. Rule routers re-run live; LLM side read from saved "
             "prediction files (no API calls). Gold is used only for scoring."),
    "datasets": {},
}

for dataset_name, path in DATASETS.items():
    cases = load_cases(str(path))
    llm_predictions = load_llm_predictions(LLM_PREDICTIONS[dataset_name])
    routers = {
        "v1_single_label": route_v1,
        "v2_multi_intent": lambda q: validate_and_repair(route_v2(q)),
        "v3_guarded": lambda q: validate_and_repair(route_v3(q)),
    }

    dataset_block = {"case_count": len(cases), "routers": {},
                     "llm_predictions_available": len(llm_predictions)}

    for router_name, router in routers.items():
        rows = []
        for case in cases:
            view = plan_view(router(case.query))
            s = score(view, case)
            s["id"] = case.id
            s["query"] = case.query
            s["tags"] = case.tags
            s["failure_labels"] = [] if s["success"] else classify_failure(s, case)
            rows.append(s)

        n = len(rows)
        single = [r for r in rows if r["gold_intent_count"] == 1]
        multi = [r for r in rows if r["gold_intent_count"] > 1]
        gold_clarify = [r for r in rows if r["gold_clarify"]]
        pred_clarify = [r for r in rows if r["pred_clarify"]]
        clarify_tp = sum(1 for r in rows if r["gold_clarify"] and r["pred_clarify"])

        metrics = {
            "intent_set_exact_match": sum(r["intent_exact"] for r in rows) / n,
            "single_intent_accuracy": (sum(r["intent_exact"] for r in single) / len(single)
                                       if single else None),
            "multi_intent_accuracy": (sum(r["intent_exact"] for r in multi) / len(multi)
                                      if multi else None),
            "subtask_count_accuracy": "UNCOMPUTABLE: gold has no subtask count; "
                                      "gold_intents is a set and cannot express two "
                                      "subtasks sharing one intent (see the 'same_intent' tag)",
            "entity_precision_recall_f1": "UNCOMPUTABLE: no entity-level gold "
                                          "(no entity_type, no span, no normalized value); "
                                          "expected_slots is a partial value dict only",
            "required_slot_completeness": "PROXY ONLY: gold marks no slot as required; "
                                          "expected_slots coverage reported instead",
            "expected_slot_coverage_proxy": sum(r["slot_exact"] for r in rows) / n,
            "tool_selection_exact_match": sum(r["tool_exact"] for r in rows) / n,
            "tool_argument_exact_match": "UNCOMPUTABLE: no gold tool arguments; "
                                         "expected_slots is matched value-wise against any "
                                         "subtask, so it cannot verify arguments per tool call",
            "clarification_precision": (clarify_tp / len(pred_clarify)
                                        if pred_clarify else None),
            "clarification_recall": (clarify_tp / len(gold_clarify)
                                     if gold_clarify else None),
            "over_routing_rate": sum(r["over_routing"] for r in rows) / n,
            "under_routing_rate": sum(r["under_routing"] for r in rows) / n,
            "end_to_end_route_success": sum(r["success"] for r in rows) / n,
        }

        failure_counts = Counter(
            label for r in rows for label in r["failure_labels"])
        failure_examples = {}
        for label in failure_counts:
            example = next(r for r in rows if label in r["failure_labels"])
            failure_examples[label] = {
                "case_id": example["id"], "query": example["query"],
                "gold_intents": example["gold_intents"], "pred_intents": example["pred_intents"],
                "gold_tools": example["gold_tools"], "pred_tools": example["pred_tools"],
                "gold_mode": example["gold_mode"], "pred_mode": example["pred_mode"],
                "missing_slot_keys": example["missing_slot_keys"],
            }
        unlabelled = [r["id"] for r in rows if not r["success"] and not r["failure_labels"]]

        dataset_block["routers"][router_name] = {
            "metrics": metrics,
            "failure_counts": dict(failure_counts),
            "failure_case_ids": {
                label: [r["id"] for r in rows if label in r["failure_labels"]]
                for label in sorted(failure_counts)},
            "failure_examples": failure_examples,
            "failed_but_unlabelled_case_ids": unlabelled,
            "rows": rows,
        }

    # ---- rule (v3) vs saved LLM structured predictions -------------------
    if llm_predictions:
        v3_rows = {r["id"]: r for r in dataset_block["routers"]["v3_guarded"]["rows"]}
        conflicts, hallucinated, llm_errors = [], [], []
        for case in cases:
            record = llm_predictions.get(case.query)
            if record is None:
                continue
            if record.get("error"):
                llm_errors.append({"case_id": case.id, "error": record["error"][:200]})
                continue
            llm_view = plan_view(record["plan"])
            rule = v3_rows[case.id]
            llm_intents = set(llm_view["intents"])
            rule_intents = set(rule["pred_intents"])
            gold_intents = set(rule["gold_intents"])

            if llm_intents != rule_intents:
                conflicts.append({
                    "case_id": case.id, "query": case.query,
                    "rule_intents": sorted(rule_intents), "llm_intents": sorted(llm_intents),
                    "gold_intents": sorted(gold_intents),
                    "rule_correct": rule_intents == gold_intents,
                    "llm_correct": llm_intents == gold_intents,
                })
            invented = llm_intents - gold_intents
            if invented:
                hallucinated.append({
                    "case_id": case.id, "query": case.query,
                    "invented_intents": sorted(invented),
                    "gold_intents": sorted(gold_intents),
                    "llm_subtask_count": llm_view["subtask_count"],
                })

        dataset_block["rule_vs_llm"] = {
            "compared": len([c for c in cases if c.query in llm_predictions]),
            "9_rule_llm_conflict_count": len(conflicts),
            "9_rule_llm_conflicts": conflicts,
            "10_llm_hallucinated_task_count": len(hallucinated),
            "10_llm_hallucinated_tasks": hallucinated,
            "llm_call_errors": llm_errors,
            "conflict_won_by": {
                "rule_only_correct": sum(1 for c in conflicts
                                         if c["rule_correct"] and not c["llm_correct"]),
                "llm_only_correct": sum(1 for c in conflicts
                                        if c["llm_correct"] and not c["rule_correct"]),
                "both_wrong": sum(1 for c in conflicts
                                  if not c["rule_correct"] and not c["llm_correct"]),
            },
        }

    report["datasets"][dataset_name] = dataset_block

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

for dataset_name, block in report["datasets"].items():
    print(f"\n===== {dataset_name}  (n={block['case_count']}) =====")
    for router_name, data in block["routers"].items():
        m = data["metrics"]
        print(f"  --- {router_name}")
        for key in ["intent_set_exact_match", "single_intent_accuracy", "multi_intent_accuracy",
                    "tool_selection_exact_match", "clarification_precision",
                    "clarification_recall", "over_routing_rate", "under_routing_rate",
                    "expected_slot_coverage_proxy", "end_to_end_route_success"]:
            value = m[key]
            shown = "—" if value is None else f"{value:.1%}"
            print(f"      {key:<34} {shown}")
        print(f"      failures by category: {data['failure_counts']}")
        if data["failed_but_unlabelled_case_ids"]:
            print(f"      unlabelled failures : {data['failed_but_unlabelled_case_ids']}")
    if "rule_vs_llm" in block:
        r = block["rule_vs_llm"]
        print(f"  --- rule(v3) vs saved LLM structured  (compared {r['compared']})")
        print(f"      9_rule_llm_conflict        {r['9_rule_llm_conflict_count']}")
        print(f"      10_llm_hallucinated_task   {r['10_llm_hallucinated_task_count']}")
        print(f"      conflict won by            {r['conflict_won_by']}")
        print(f"      llm call errors            {len(r['llm_call_errors'])}")

print(f"\nsaved -> {OUT}")
