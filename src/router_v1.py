from __future__ import annotations

from src.extractors import extract_slots
from src.models import Intent, IntentPlan, SubTask, ToolName
from src.normalize import normalize_text


def route_v1(query: str) -> IntentPlan:
    """Naive single-label router used to reproduce common bad cases.

    It intentionally returns only one intent/tool and therefore under-calls on
    multi-intent queries.
    """
    q = normalize_text(query)
    slots = extract_slots(q)

    if any(k in q for k in ["图纸", "图号", "原理图", "接线图"]):
        intent, tool = Intent.DRAWING_LOOKUP, ToolName.DRAWING_SEARCH
    elif any(k in q for k in ["规范", "条文", "标准", "依据", "要求", "JTG ", "GB ", "TB "]):
        intent, tool = Intent.REGULATION_LOOKUP, ToolName.HYBRID_SEARCH
    elif any(k in q for k in ["工单", "维修单", "派单", "闭环"]):
        intent, tool = Intent.WORK_ORDER_QUERY, ToolName.NL2API_QUERY
    elif any(k in q for k in ["多少", "次数", "趋势", "完成率", "平均值", "统计"]):
        intent, tool = Intent.METRIC_QUERY, ToolName.NL2API_QUERY
    elif any(k in q for k in ["状态", "运行", "在线", "离线", "告警", "报警"]):
        intent, tool = Intent.EQUIPMENT_STATUS, ToolName.NL2API_QUERY
    elif any(k in q for k in ["原因", "怎么处理", "处理建议", "排查"]):
        intent, tool = Intent.FAULT_DIAGNOSIS, ToolName.NONE
    else:
        intent, tool = Intent.GENERAL_QA, ToolName.NONE

    task = SubTask(
        id="T1",
        intent=intent,
        tool=tool,
        query=q,
        slots=slots,
        confidence=0.72,
        reason="v1按关键词优先级只选择一个标签",
    )
    return IntentPlan(
        normalized_query=q,
        user_goal=q,
        plan_mode="single" if tool != ToolName.NONE else "none",
        subtasks=[task],
    )
