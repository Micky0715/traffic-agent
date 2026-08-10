from __future__ import annotations

import re
from typing import List, Tuple

from src.extractors import ASSET_PATTERN, extract_slots, has_time_signal
from src.models import Intent, IntentPlan, SubTask, ToolName
from src.normalize import normalize_text


REG_WORDS = ["规范", "规程", "条文", "标准", "依据", "要求", "限值", "JTG ", "GB ", "TB ", "CJJ ", "OPS-"]
DRAWING_WORDS = ["图纸", "图号", "原理图", "接线图", "设备布置图", "剖面图"]
WORKORDER_WORDS = ["工单", "维修单", "维修记录", "派单", "闭环", "处理记录", "检修记录"]
METRIC_WORDS = ["多少", "几次", "次数", "趋势", "完成率", "平均值", "统计", "汇总", "排名"]
STATUS_WORDS = ["状态", "运行", "在线", "离线", "告警", "报警", "电流", "温度", "振动值", "压力"]
DIAG_WORDS = ["原因", "怎么处理", "处理建议", "排查", "定位故障", "诊断", "处置建议", "判断是否", "是否需要停机", "异常设备"]
PRONOUNS = ["它", "这个设备", "该设备", "这台设备", "那个设备"]

UNSAFE_WRITE_WORDS = ["下发控制", "远程启动", "远程停机", "打开风机", "关闭风机"]
WRITE_VERBS = ["删除", "修改", "关闭", "新建", "创建"]
WRITE_OBJECTS = ["工单", "告警", "维修单", "记录"]
DRAWING_PRONOUNS = ["这个图纸", "该图纸", "这张图", "那张图"]


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def _split_clauses(q: str) -> List[str]:
    # Only split on discourse-level connectors. Do not split simple noun phrases
    # such as “风机设备规范和检修要求”.
    pattern = r"(?:；|。|，(?=再|然后|同时|并且|顺便|另外|还要|以及)|\b再\b|然后|同时|并且|顺便|另外|还要)"
    parts = [p.strip(" ，；。") for p in re.split(pattern, q) if p.strip(" ，；。")]
    return parts or [q]


def _classify_clause(clause: str) -> List[Tuple[Intent, ToolName, str]]:
    found: List[Tuple[Intent, ToolName, str]] = []
    if _contains_any(clause, REG_WORDS):
        found.append((Intent.REGULATION_LOOKUP, ToolName.HYBRID_SEARCH, "出现规范/条文/标准/编号类信号"))
    if _contains_any(clause, DRAWING_WORDS):
        found.append((Intent.DRAWING_LOOKUP, ToolName.DRAWING_SEARCH, "出现图纸/图号/接线图类信号"))
    if _contains_any(clause, WORKORDER_WORDS):
        found.append((Intent.WORK_ORDER_QUERY, ToolName.NL2API_QUERY, "出现工单/维修记录类信号"))
    has_metric = _contains_any(clause, METRIC_WORDS) or bool(re.search(r"\d+(?:\.\d+)?\s*次", clause))
    has_explicit_status = _contains_any(clause, ["状态", "运行", "在线", "离线", "当前温度", "当前电流", "当前压力", "当前振动值"])
    if has_metric:
        found.append((Intent.METRIC_QUERY, ToolName.NL2API_QUERY, "出现统计/次数/趋势类信号"))
    if _contains_any(clause, STATUS_WORDS) and (not has_metric or has_explicit_status):
        found.append((Intent.EQUIPMENT_STATUS, ToolName.NL2API_QUERY, "出现设备实时/运行状态类信号"))
    if _contains_any(clause, DIAG_WORDS):
        found.append((Intent.FAULT_DIAGNOSIS, ToolName.NONE, "用户要求原因分析或处置建议"))
    return found


def _deduplicate(tasks: List[SubTask]) -> List[SubTask]:
    result: List[SubTask] = []
    seen = set()
    for task in tasks:
        key = (task.intent, task.tool, task.query)
        if key not in seen:
            seen.add(key)
            result.append(task)
    return result


def route_v2(query: str) -> IntentPlan:
    q = normalize_text(query)
    if _contains_any(q, UNSAFE_WRITE_WORDS) or (_contains_any(q, WRITE_VERBS) and _contains_any(q, WRITE_OBJECTS)):
        return IntentPlan(
            normalized_query=q,
            user_goal=q,
            plan_mode="none",
            subtasks=[SubTask(
                id="T1",
                intent=Intent.UNSUPPORTED,
                tool=ToolName.NONE,
                query=q,
                confidence=0.98,
                reason="当前工具仅允许只读查询，不执行删除、修改或设备控制操作",
            )],
            answer_constraints=["说明当前系统只支持只读查询，并引导用户走人工审批流程"],
        )
    conditional = bool(re.search(r"(?:如果|若).{0,40}(?:则|再|就)|当.{0,30}时.{0,20}(?:再|则|就)", q))
    clauses = _split_clauses(q)
    global_slots = extract_slots(q)
    tasks: List[SubTask] = []

    for clause in clauses:
        labels = _classify_clause(clause)
        # A clause can itself contain multiple true intents, e.g. “查规范并统计告警次数”.
        for intent, tool, reason in labels:
            slots = extract_slots(clause)
            tasks.append(
                SubTask(
                    id="",
                    intent=intent,
                    tool=tool,
                    query=clause,
                    entities={k: v for k, v in slots.items() if k in {"asset_id", "regulation_code", "drawing_no"}},
                    slots=slots,
                    confidence=0.9 if tool != ToolName.NONE else 0.84,
                    reason=reason,
                )
            )

    # Propagate only fields relevant to the task type. Avoid leaking a regulation
    # code into an API task or a time range into a static regulation query.
    for task in tasks:
        relevant = []
        if task.intent in {Intent.EQUIPMENT_STATUS, Intent.WORK_ORDER_QUERY, Intent.METRIC_QUERY}:
            relevant = ["asset_id", "time_range", "threshold"]
        elif task.intent == Intent.REGULATION_LOOKUP:
            relevant = ["regulation_code"]
        elif task.intent == Intent.DRAWING_LOOKUP:
            relevant = ["drawing_no", "asset_id"]
        for key in relevant:
            if key not in task.slots and key in global_slots:
                task.slots[key] = global_slots[key]
                if key in {"asset_id", "regulation_code", "drawing_no"}:
                    task.entities[key] = global_slots[key]

    if not tasks:
        tasks.append(
            SubTask(
                id="",
                intent=Intent.GENERAL_QA,
                tool=ToolName.NONE,
                query=q,
                confidence=0.75,
                reason="未发现需要外部工具的明确业务信号",
            )
        )

    tasks = _deduplicate(tasks)
    for i, task in enumerate(tasks, start=1):
        task.id = f"T{i}"

    # Diagnose is a synthesis task and depends on all evidence-producing tasks.
    evidence_ids = [t.id for t in tasks if t.tool != ToolName.NONE]
    for task in tasks:
        if task.intent == Intent.FAULT_DIAGNOSIS:
            task.depends_on = evidence_ids.copy()

    # Conditional query: the later regulation/diagnosis task depends on first API task.
    if conditional:
        first_api = next((t.id for t in tasks if t.tool == ToolName.NL2API_QUERY), None)
        if first_api:
            for task in tasks:
                if task.id != first_api and task.intent in {Intent.REGULATION_LOOKUP, Intent.FAULT_DIAGNOSIS}:
                    task.depends_on = [first_api]
                    task.condition = "仅当上游结果满足用户给定条件时执行"

    missing = []
    has_asset = bool(ASSET_PATTERN.search(q))
    pronoun_only = _contains_any(q, PRONOUNS) and not has_asset
    for task in tasks:
        if task.intent in {Intent.EQUIPMENT_STATUS, Intent.WORK_ORDER_QUERY, Intent.METRIC_QUERY}:
            if pronoun_only or "asset_id" not in task.slots:
                task.missing_slots.append("asset_id")
            if task.intent in {Intent.METRIC_QUERY, Intent.WORK_ORDER_QUERY} and not has_time_signal(task.query):
                task.missing_slots.append("time_range")
        if task.intent == Intent.DRAWING_LOOKUP:
            if not any(k in task.slots for k in ["drawing_no", "asset_id"]):
                # Keywords can still locate drawings; only ask when the clause is fully generic.
                if task.query.strip() in {"查图纸", "看看图纸", "查询图纸"} or _contains_any(task.query, DRAWING_PRONOUNS):
                    task.missing_slots.append("drawing_no_or_asset_id")
        missing.extend(task.missing_slots)

    needs_clarification = bool(missing)
    clarification = None
    if needs_clarification:
        unique = list(dict.fromkeys(missing))
        labels = {
            "asset_id": "具体设备或设备编号",
            "time_range": "查询时间范围",
            "drawing_no_or_asset_id": "图号或对应设备编号",
        }
        clarification = "请补充" + "、".join(labels.get(x, x) for x in unique) + "，我再执行查询。"

    tool_tasks = [t for t in tasks if t.tool != ToolName.NONE]
    diagnosis_tasks = [t for t in tasks if t.intent == Intent.FAULT_DIAGNOSIS]
    if needs_clarification:
        mode = "clarify"
    elif conditional:
        mode = "conditional"
    elif len(tool_tasks) == 0:
        mode = "none"
    elif len(tool_tasks) == 1 and not diagnosis_tasks:
        mode = "single"
    elif len(tool_tasks) > 1 and diagnosis_tasks:
        mode = "parallel_then_synthesize"
    elif len(tool_tasks) > 1:
        mode = "parallel"
    else:
        mode = "serial"

    return IntentPlan(
        normalized_query=q,
        user_goal=q,
        needs_clarification=needs_clarification,
        clarification_question=clarification,
        plan_mode=mode,
        subtasks=tasks,
        answer_constraints=[
            "不得编造工具未返回的事实或数值",
            "规范结论必须附来源编号或文档片段",
            "实时数据与经验性建议分开表达",
            "部分工具失败时明确说明缺失范围",
        ],
    )
