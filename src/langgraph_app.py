from __future__ import annotations

"""Optional LangGraph implementation.

Install requirements, then run:
    python -m src.langgraph_app "查JTG D81-2017要求，再统计A12风机最近3天告警次数"
"""

import sys
from typing import Any, Dict, List, TypedDict

from src.models import IntentPlan, ToolResult
from src.pipeline import _execute_dependent, _execute_independent, synthesize
from src.router_v3 import route_v3
from src.validator import validate_and_repair

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 requirements.txt 中的 langgraph 与 langchain-core") from exc


class AgentState(TypedDict, total=False):
    query: str
    plan: Dict[str, Any]
    results: List[Dict[str, Any]]
    answer: str


def route_node(state: AgentState) -> AgentState:
    plan = validate_and_repair(route_v3(state["query"]))
    return {"plan": plan.model_dump(mode="json")}


def branch_after_route(state: AgentState) -> str:
    plan = IntentPlan.model_validate(state["plan"])
    return "clarify" if plan.needs_clarification else "execute"


def clarify_node(state: AgentState) -> AgentState:
    plan = IntentPlan.model_validate(state["plan"])
    return {"answer": plan.clarification_question or "请补充查询条件。", "results": []}


def execute_node(state: AgentState) -> AgentState:
    plan = IntentPlan.model_validate(state["plan"])
    results = _execute_independent(plan)
    results = _execute_dependent(plan, results)
    return {"results": [r.model_dump(mode="json") for r in results]}


def synthesize_node(state: AgentState) -> AgentState:
    plan = IntentPlan.model_validate(state["plan"])
    results = [ToolResult.model_validate(x) for x in state.get("results", [])]
    return {"answer": synthesize(plan, results)}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("route", route_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("execute", execute_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", branch_after_route, {"clarify": "clarify", "execute": "execute"})
    builder.add_edge("clarify", END)
    builder.add_edge("execute", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "查JTG D81-2017照明要求，再统计A12风机最近3天告警次数，并给处理建议"
    graph = build_graph()
    print(graph.invoke({"query": query}))
