"""Tests for the legacy <-> unified adapter.

The most important assertions here are about what the adapter REFUSES to do.
The old format does not record query spans, tool candidates or entity
provenance, and the cheapest way to make the new contract look complete would
be to fill those in from something plausible. Every such shortcut is asserted
against, because a fabricated provenance field is worse than a null: it reads
as a measurement.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.models import Intent, IntentPlan, SubTask, ToolName
from src.routing import (
    ReasonCode, SchemaAdaptationError, legacy_request_id, to_legacy, to_unified,
)

ROOT = Path(__file__).resolve().parents[1]


def legacy_task(task_id="T1", intent=Intent.REGULATION_LOOKUP,
                tool=ToolName.HYBRID_SEARCH, **kwargs) -> SubTask:
    base = {"query": "查规范", "slots": {}, "missing_slots": [], "depends_on": [],
            "confidence": 0.9, "reason": ""}
    base.update(kwargs)
    return SubTask(id=task_id, intent=intent, tool=tool, **base)


def legacy_plan(subtasks, plan_mode="single", **kwargs) -> IntentPlan:
    base = {"normalized_query": "查规范", "user_goal": "查规范",
            "needs_clarification": False, "answer_constraints": []}
    base.update(kwargs)
    return IntentPlan(plan_mode=plan_mode, subtasks=subtasks, **base)


def convert(plan, query="查规范", strategy="rule_v3", **kwargs):
    return to_unified(plan, original_query=query, source_strategy=strategy, **kwargs)


# --------------------------------------------------------------------------
# 1-4. decisions
# --------------------------------------------------------------------------

def test_single_executable_task_converts_to_execute():
    unified = convert(legacy_plan([legacy_task()]))
    assert unified.decision == "execute"
    assert unified.subtasks[0].decision == "execute"
    assert unified.subtasks[0].selected_tool == "hybrid_search"


def test_needs_clarification_converts_to_clarify():
    plan = legacy_plan(
        [legacy_task(intent=Intent.METRIC_QUERY, tool=ToolName.NL2API_QUERY,
                     missing_slots=["asset_id"])],
        plan_mode="clarify", needs_clarification=True,
        clarification_question="请补充设备编号。")
    unified = convert(plan)
    assert unified.decision == "clarify"
    assert unified.subtasks[0].decision == "clarify"
    assert ReasonCode.REQUIRED_SLOT_MISSING in unified.subtasks[0].reason_codes
    assert ReasonCode.ENTITY_MISSING in unified.subtasks[0].reason_codes


def test_all_unsupported_converts_to_reject():
    plan = legacy_plan(
        [legacy_task(intent=Intent.UNSUPPORTED, tool=ToolName.NONE,
                     reason="当前工具仅允许只读查询，不执行删除、修改或设备控制操作")],
        plan_mode="none")
    unified = convert(plan)
    assert unified.decision == "reject"
    assert unified.subtasks[0].decision == "reject"
    assert ReasonCode.TOOL_PROHIBITED in unified.subtasks[0].reason_codes


def test_mixed_supported_and_unsupported_converts_to_partial():
    """The state the legacy format could not express. A query mixing a lookup
    with a forbidden write had to be forced entirely one way; neither answer is
    right."""
    plan = legacy_plan([
        legacy_task("T1", intent=Intent.EQUIPMENT_STATUS, tool=ToolName.NL2API_QUERY,
                    slots={"asset_id": "A12风机"}),
        legacy_task("T2", intent=Intent.UNSUPPORTED, tool=ToolName.NONE,
                    reason="当前工具仅允许只读查询，不执行删除、修改或设备控制操作"),
    ], plan_mode="parallel")
    unified = convert(plan, query="查询A12风机状态，同时删除历史记录。")

    assert unified.decision == "partial"
    assert [t.decision for t in unified.subtasks] == ["execute", "reject"]
    assert ReasonCode.PARTIAL_EXECUTION in unified.reason_codes


def test_empty_plan_is_an_error_not_a_fallback():
    """fallback is a deliberate routing outcome; the absence of one is a bug.
    Conflating them would let a structurally empty plan pass as a decision."""
    with pytest.raises(SchemaAdaptationError, match="empty plan"):
        convert(legacy_plan([]))


# --------------------------------------------------------------------------
# 5-8. what must not be fabricated
# --------------------------------------------------------------------------

def test_legacy_plan_mode_is_preserved():
    """The existing evaluation scores against plan_mode; losing it would break
    every historical comparison."""
    unified = convert(legacy_plan([legacy_task()], plan_mode="parallel_then_synthesize"))
    assert unified.legacy_plan_mode == "parallel_then_synthesize"


def test_query_span_is_none_for_legacy_data():
    """The old format never recorded which words a subtask came from. Using the
    whole query as every subtask's span would make span-based metrics look
    computable when they are not."""
    unified = convert(legacy_plan([legacy_task(), legacy_task("T2")], plan_mode="parallel"),
                      query="查规范，再查图纸")
    assert all(t.query_span is None for t in unified.subtasks)
    assert ReasonCode.QUERY_SPAN_UNAVAILABLE_LEGACY in unified.reason_codes


def test_entity_source_is_legacy_not_regex():
    """A legacy plan cannot say whether a value came from a pattern or a model."""
    unified = convert(legacy_plan(
        [legacy_task(intent=Intent.EQUIPMENT_STATUS, tool=ToolName.NL2API_QUERY,
                     slots={"asset_id": "A12风机"})]))
    entity = unified.subtasks[0].entities[0]
    assert entity.source == "legacy"
    assert ReasonCode.ENTITY_SOURCE_UNAVAILABLE_LEGACY in unified.reason_codes


def test_tool_candidates_are_not_fabricated_from_the_selected_tool():
    """Presenting the one chosen tool as a scored candidate would invent a
    selection process that never ran."""
    unified = convert(legacy_plan([legacy_task()]))
    assert unified.subtasks[0].tool_candidates == []
    assert unified.subtasks[0].selected_tool == "hybrid_search"
    assert ReasonCode.TOOL_CANDIDATES_UNAVAILABLE_LEGACY in unified.reason_codes


def test_model_info_and_prompt_version_stay_none_when_unknown():
    unified = convert(legacy_plan([legacy_task()]))
    assert unified.model_info is None and unified.prompt_version is None
    assert ReasonCode.MODEL_INFO_UNAVAILABLE_LEGACY in unified.reason_codes
    assert ReasonCode.PROMPT_VERSION_UNAVAILABLE_LEGACY in unified.reason_codes


# --------------------------------------------------------------------------
# 9-10. reason text handling
# --------------------------------------------------------------------------

def test_original_chinese_reason_text_is_kept():
    text = "当前工具仅允许只读查询，不执行删除、修改或设备控制操作"
    unified = convert(legacy_plan(
        [legacy_task(intent=Intent.UNSUPPORTED, tool=ToolName.NONE, reason=text)],
        plan_mode="none"))
    assert text in unified.subtasks[0].reason_texts


def test_unmappable_reason_produces_reason_unmapped_without_losing_text():
    text = "某个从未见过的解释文本"
    unified = convert(legacy_plan([legacy_task(reason=text)]))
    assert ReasonCode.REASON_UNMAPPED in unified.subtasks[0].reason_codes
    assert unified.subtasks[0].reason_texts == [text]


def test_multi_part_reason_is_split_and_both_parts_kept():
    unified = convert(legacy_plan([legacy_task(
        reason="出现规范/条文/标准/编号类信号；validator修复工具映射 none->hybrid_search")]))
    assert len(unified.subtasks[0].reason_texts) == 2
    assert ReasonCode.TOOL_NOT_ALLOWED_FOR_INTENT in unified.subtasks[0].reason_codes


# --------------------------------------------------------------------------
# 11-18. strict validation
# --------------------------------------------------------------------------

def test_duplicate_subtask_id_raises():
    """The legacy model rejects duplicates at construction, so this reproduces
    what arrives from a deserialized prediction file, where the constructor's
    check was bypassed."""
    plan = legacy_plan([legacy_task("T1"), legacy_task("T2")], plan_mode="parallel")
    object.__setattr__(plan.subtasks[1], "id", "T1")
    with pytest.raises(SchemaAdaptationError, match="duplicate subtask_id"):
        convert(plan)


def test_dependency_on_a_missing_task_raises():
    plan = legacy_plan([legacy_task("T1")], plan_mode="serial")
    plan.subtasks[0].depends_on = ["T9"]
    with pytest.raises(SchemaAdaptationError, match="depends on unknown"):
        convert(plan)


def test_dependency_cycle_raises():
    plan = legacy_plan([legacy_task("T1"), legacy_task("T2")], plan_mode="serial")
    plan.subtasks[0].depends_on = ["T2"]
    plan.subtasks[1].depends_on = ["T1"]
    with pytest.raises(SchemaAdaptationError, match="dependency cycle"):
        convert(plan)


def test_unknown_intent_raises_and_is_not_downgraded_to_general_qa():
    plan = legacy_plan([legacy_task()])
    object.__setattr__(plan.subtasks[0], "intent", "totally_new_intent")
    with pytest.raises(SchemaAdaptationError, match="unknown intent"):
        convert(plan)


def test_tool_not_allowed_for_intent_raises():
    plan = legacy_plan([legacy_task(intent=Intent.REGULATION_LOOKUP,
                                    tool=ToolName.DRAWING_SEARCH)])
    with pytest.raises(SchemaAdaptationError, match="not allowed for intent"):
        convert(plan)


def test_unknown_plan_mode_raises():
    plan = legacy_plan([legacy_task()])
    object.__setattr__(plan, "plan_mode", "some_new_mode")
    with pytest.raises(SchemaAdaptationError, match="unknown legacy plan_mode"):
        convert(plan)


def test_missing_config_raises():
    with pytest.raises(SchemaAdaptationError, match="routing config not found"):
        convert(legacy_plan([legacy_task()]), config_path="configs/does_not_exist.yaml")


# --------------------------------------------------------------------------
# 19-20. purity and round trip
# --------------------------------------------------------------------------

def test_input_plan_is_not_mutated():
    plan = legacy_plan([legacy_task(slots={"asset_id": "A12风机"})])
    before = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    convert(plan)
    after = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    assert before == after


def test_round_trip_preserves_the_fields_the_old_evaluator_reads():
    original = legacy_plan([
        legacy_task("T1", intent=Intent.REGULATION_LOOKUP, tool=ToolName.HYBRID_SEARCH,
                    slots={"regulation_code": "JTG D81-2017"}),
        legacy_task("T2", intent=Intent.EQUIPMENT_STATUS, tool=ToolName.NL2API_QUERY,
                    slots={"asset_id": "A12风机"}),
    ], plan_mode="parallel")
    restored = to_legacy(convert(original, query="查规范，再查状态"))

    assert [t.intent for t in restored.subtasks] == [t.intent for t in original.subtasks]
    assert [t.tool for t in restored.subtasks] == [t.tool for t in original.subtasks]
    assert [t.slots for t in restored.subtasks] == [t.slots for t in original.subtasks]
    assert restored.plan_mode == original.plan_mode
    assert restored.needs_clarification == original.needs_clarification


def test_round_trip_preserves_clarification():
    original = legacy_plan(
        [legacy_task(intent=Intent.METRIC_QUERY, tool=ToolName.NL2API_QUERY,
                     missing_slots=["asset_id"])],
        plan_mode="clarify", needs_clarification=True, clarification_question="补充设备。")
    restored = to_legacy(convert(original))
    assert restored.needs_clarification is True
    assert restored.clarification_question == "补充设备。"
    assert restored.plan_mode == "clarify"


def test_to_legacy_refuses_a_plan_that_never_had_a_legacy_mode():
    """A natively-unified plan has no legacy mode, and inventing one would
    change how the old evaluator scores it."""
    unified = convert(legacy_plan([legacy_task()]))
    stripped = unified.model_copy(update={"legacy_plan_mode": None})
    with pytest.raises(SchemaAdaptationError, match="legacy_plan_mode is unset"):
        to_legacy(stripped)


# --------------------------------------------------------------------------
# 21-22. batch conversion of the frozen predictions
# --------------------------------------------------------------------------

def test_legacy_request_id_is_deterministic():
    """A UUID here would make two conversions of the same frozen file
    undiffable."""
    assert legacy_request_id("base46", "A01") == "legacy:base46:A01"
    assert legacy_request_id("base46", "A01") == legacy_request_id("base46", "A01")


@pytest.mark.parametrize("name", [
    "llm_llm_v2_structured_predictions.unified.jsonl",
    "llm_challenge_v2_llm_v1_naive_predictions.unified.jsonl",
])
def test_converted_prediction_files_exist_and_parse(name):
    path = ROOT / "outputs" / "unified_routing_predictions" / name
    if not path.exists():
        pytest.skip("run scripts/convert_legacy_routing_predictions.py first")
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    assert rows
    for row in rows:
        assert row["dataset_name"] and row["case_id"]
        plan = row["plan"]
        assert plan["request_id"].startswith("legacy:")
        assert plan["model_info"] is None
        assert all(t["query_span"] is None for t in plan["subtasks"])


def test_batch_conversion_makes_no_api_calls(monkeypatch):
    """Conversion reads files already on disk. A network call here would mean a
    report could quietly cost money and change between runs."""
    import src.routing.adapter as adapter_module

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("adapter attempted a network call")

    monkeypatch.setattr("openai.OpenAI", explode, raising=False)
    unified = convert(legacy_plan([legacy_task()]),
                      request_id=legacy_request_id("base46", "A01"))
    assert unified.request_id == "legacy:base46:A01"
    assert adapter_module.SchemaAdaptationError is SchemaAdaptationError
