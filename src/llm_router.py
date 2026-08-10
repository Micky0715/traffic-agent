from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from src.models import Intent, IntentPlan, SubTask, ToolName


ROOT = Path(__file__).resolve().parents[1]

# Mirrors validator.ALLOWED_TOOL_BY_INTENT. Kept local so the naive V1 baseline
# does not depend on the repaired/structured code path it is being compared against.
NAIVE_INTENT_TO_TOOL = {
    Intent.REGULATION_LOOKUP: ToolName.HYBRID_SEARCH,
    Intent.EQUIPMENT_STATUS: ToolName.NL2API_QUERY,
    Intent.WORK_ORDER_QUERY: ToolName.NL2API_QUERY,
    Intent.METRIC_QUERY: ToolName.NL2API_QUERY,
    Intent.DRAWING_LOOKUP: ToolName.DRAWING_SEARCH,
    Intent.FAULT_DIAGNOSIS: ToolName.NONE,
    Intent.GENERAL_QA: ToolName.NONE,
    Intent.UNSUPPORTED: ToolName.NONE,
}


def _client() -> OpenAI:
    # override=True: .env must win over a stale OPENAI_API_KEY already sitting
    # in the shell environment (e.g. from an unrelated tool's config), otherwise
    # requests silently authenticate with the wrong key.
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)


def _call_json(prompt_file: str, query: str) -> dict:
    """Call an OpenAI-compatible model in JSON mode and return the parsed object.

    Set OPENAI_BASE_URL / MODEL_NAME in .env to point at Qwen, DeepSeek or an
    internal gateway instead of OpenAI directly.
    """
    client = _client()
    prompt = (ROOT / "prompts" / prompt_file).read_text(encoding="utf-8")
    model = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=float(os.getenv("ROUTER_TEMPERATURE", "0")),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("model returned empty content")
    return json.loads(content)


def route_with_llm_naive(query: str) -> IntentPlan:
    """Reproduce the resume's 'raw Function Calling, one label only' baseline for real.

    Uses prompts/00_intent_router_v1_bad.md, which forces the model to output a
    single intent. This is the actual failure mode being compared against, not a
    simulation of it, so its mistakes on multi-intent queries are genuine bad cases.
    """
    data = _call_json("00_intent_router_v1_bad.md", query)
    intent = Intent(data["intent"])
    tool = NAIVE_INTENT_TO_TOOL[intent]
    return IntentPlan(
        normalized_query=query,
        user_goal=query,
        plan_mode="single" if tool != ToolName.NONE else "none",
        subtasks=[SubTask(
            id="T1",
            intent=intent,
            tool=tool,
            query=query,
            confidence=float(data.get("confidence", 0.0)),
            reason="v1_naive_single_label",
        )],
        answer_constraints=[],
    )


def route_with_llm_structured(query: str) -> IntentPlan:
    """The structured multi-intent planner (prompts/01_intent_router_v2.md)."""
    data = _call_json("01_intent_router_v2.md", query)
    return IntentPlan.model_validate(data)


# Backward-compatible name used by run_llm_case.py.
route_with_llm = route_with_llm_structured
