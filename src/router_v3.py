from __future__ import annotations

import re

from src.models import Intent, IntentPlan, SubTask, ToolName
from src.normalize import normalize_text
from src.router_v2 import route_v2


TOOL_PROHIBITION = [
    "不要访问设备数据库",
    "不访问设备数据库",
    "不要调用设备接口",
    "不要调用API",
    "不要查询内部数据库",
]

NEGATED_SPAN = re.compile(
    r"(?:不要|别|无需|不需要)\s*(?:查|查询|检索|打开|调用|提供|给)?\s*"
    r"(?:规范|图纸|处理建议|处置建议|设备数据库|接口)[^，。；]*"
)
QUOTED = re.compile(r"[“\"]([^”\"]+)[”\"]")
ABSOLUTE_RANGE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*(?:至|到|~|～|-)\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
RELATIVE_RANGE = re.compile(r"(?:近|最近|过去)\s*(?:\d+|[一二两三四五六七八九十])\s*(?:小时|天|周|月|年)")


def _unsupported(query: str, reason: str) -> IntentPlan:
    return IntentPlan(
        normalized_query=normalize_text(query),
        user_goal=query,
        plan_mode="none",
        subtasks=[SubTask(
            id="T1",
            intent=Intent.UNSUPPORTED,
            tool=ToolName.NONE,
            query=query,
            confidence=0.99,
            reason=reason,
        )],
        answer_constraints=["说明当前系统无法在用户禁止必要数据源的情况下完成该查询"],
    )


def route_v3(query: str) -> IntentPlan:
    q = normalize_text(query)
    if any(x in q for x in TOOL_PROHIBITION):
        return _unsupported(q, "用户禁止访问完成该任务所必需的数据源")

    has_relative = bool(RELATIVE_RANGE.search(q))
    has_absolute = bool(ABSOLUTE_RANGE.search(q))

    # Remove explicitly negated intents before classification.
    sanitized = NEGATED_SPAN.sub("", q)
    # Quoted words are treated as search terms or UI text, not as independent intent signals.
    quoted_terms = QUOTED.findall(sanitized)
    classification_text = QUOTED.sub("关键词", sanitized)

    plan = route_v2(classification_text)
    plan.normalized_query = q
    plan.user_goal = q

    if has_relative and has_absolute:
        plan.needs_clarification = True
        plan.plan_mode = "clarify"
        plan.clarification_question = "你同时给出了相对时间和绝对日期，请确认以哪一个时间范围为准。"
        for task in plan.subtasks:
            if task.intent in {Intent.METRIC_QUERY, Intent.WORK_ORDER_QUERY}:
                if "time_range_conflict" not in task.missing_slots:
                    task.missing_slots.append("time_range_conflict")

    # Restore original wording for static search so quoted keywords are not lost.
    if quoted_terms:
        for task in plan.subtasks:
            if task.intent == Intent.REGULATION_LOOKUP:
                task.query = q

    return IntentPlan.model_validate(plan.model_dump())
