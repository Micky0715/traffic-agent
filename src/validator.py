from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from src.models import Intent, IntentPlan, ToolName


ALLOWED_TOOL_BY_INTENT: Dict[Intent, List[ToolName]] = {
    Intent.REGULATION_LOOKUP: [ToolName.HYBRID_SEARCH],
    Intent.EQUIPMENT_STATUS: [ToolName.NL2API_QUERY],
    Intent.WORK_ORDER_QUERY: [ToolName.NL2API_QUERY],
    Intent.METRIC_QUERY: [ToolName.NL2API_QUERY],
    Intent.DRAWING_LOOKUP: [ToolName.DRAWING_SEARCH],
    Intent.FAULT_DIAGNOSIS: [ToolName.NONE],
    Intent.GENERAL_QA: [ToolName.NONE],
    Intent.UNSUPPORTED: [ToolName.NONE],
}


def validate_and_repair(plan: IntentPlan) -> IntentPlan:
    repaired = deepcopy(plan)
    ids = {task.id for task in repaired.subtasks}
    for task in repaired.subtasks:
        allowed = ALLOWED_TOOL_BY_INTENT[task.intent]
        if task.tool not in allowed:
            task.reason += f"；validator修复工具映射 {task.tool.value}->{allowed[0].value}"
            task.tool = allowed[0]
        task.depends_on = [x for x in task.depends_on if x in ids and x != task.id]

    # Do not execute any external tool while critical slots are missing.
    if any(task.missing_slots for task in repaired.subtasks):
        repaired.needs_clarification = True
        repaired.plan_mode = "clarify"
        if not repaired.clarification_question:
            repaired.clarification_question = "查询条件不完整，请补充设备、时间范围或图号。"
    return IntentPlan.model_validate(repaired.model_dump())
