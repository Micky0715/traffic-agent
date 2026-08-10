from __future__ import annotations

import json
from typing import List

from src.llm_router import _call_json
from src.models import IntentPlan, SufficiencyJudgment, ToolResult


def _build_context(query: str, plan: IntentPlan, results: List[ToolResult], extra_context: str = "") -> str:
    payload = {
        "user_query": query,
        "subtasks": [t.model_dump(mode="json") for t in plan.subtasks],
        "tool_results": [r.model_dump(mode="json") for r in results],
    }
    if extra_context:
        # Not part of the original prompt's documented input shape. Used to test
        # whether retry/attempt history changes the model's next_action choice
        # even though prompts/03_result_sufficiency.md never asked for it.
        payload["retry_history_note"] = extra_context
    return json.dumps(payload, ensure_ascii=False)


def judge_sufficiency(
    query: str, plan: IntentPlan, results: List[ToolResult], extra_context: str = ""
) -> SufficiencyJudgment:
    """Real call to prompts/03_result_sufficiency.md.

    This prompt existed in the repo but nothing wired it up: run_pipeline() in
    pipeline.py only checks "did any tool succeed" for fault_diagnosis tasks.
    This is the actual multi-dimensional judgment the resume's 'confidence
    threshold + sufficiency check' claim describes.
    """
    context = _build_context(query, plan, results, extra_context)
    data = _call_json("03_result_sufficiency.md", context)
    return SufficiencyJudgment.model_validate(data)
