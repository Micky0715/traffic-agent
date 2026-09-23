"""Tests for the unified routing contract itself.

These check the invariants the schema enforces on its own, independent of any
conversion: dependency integrity, span consistency, and the fact that a plan
cannot claim a decision its subtasks contradict.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.routing.reason_codes import UNAVAILABLE_CODES, ReasonCode, parse
from src.routing.schema import (
    ModelInfo, ResolvedEntity, Span, ToolCandidate, UnifiedIntentPlan,
    UnifiedSubTask,
)


def task(subtask_id: str, **kwargs) -> UnifiedSubTask:
    base = {"intent": "regulation_lookup", "decision": "execute",
            "selected_tool": "hybrid_search"}
    base.update(kwargs)
    return UnifiedSubTask(subtask_id=subtask_id, **base)


def plan(subtasks, **kwargs) -> UnifiedIntentPlan:
    base = {"request_id": "rt:test", "taxonomy_version": "t1", "decision": "execute",
            "original_query": "q", "normalized_query": "q"}
    base.update(kwargs)
    return UnifiedIntentPlan(subtasks=subtasks, **base)


# --------------------------------------------------------------------------
# Span
# --------------------------------------------------------------------------

def test_span_text_must_match_its_own_length():
    """Stored redundantly so an offset computed on a normalized string and
    applied to a raw one is caught, rather than silently pointing at the wrong
    words."""
    Span(start=2, end=5, text="abc")
    with pytest.raises(ValidationError):
        Span(start=2, end=5, text="abcd")


def test_span_rejects_inverted_or_negative_ranges():
    with pytest.raises(ValidationError):
        Span(start=5, end=2, text="")
    with pytest.raises(ValidationError):
        Span(start=-1, end=2, text="ab")


# --------------------------------------------------------------------------
# ResolvedEntity
# --------------------------------------------------------------------------

def test_resolved_entity_needs_a_normalized_value():
    """Claiming an entity is resolved while carrying no value is the shape of a
    silent gap — the status says "usable", the content says otherwise."""
    with pytest.raises(ValidationError):
        ResolvedEntity(entity_type="asset", raw_text="A12风机", resolution_status="resolved")


def test_unresolved_entity_may_have_no_value():
    entity = ResolvedEntity(entity_type="asset", raw_text="", resolution_status="unresolved")
    assert entity.normalized_value is None
    assert entity.source == "legacy"  # default is the honest one, not "regex"


def test_ambiguous_entity_carries_its_candidates():
    entity = ResolvedEntity(entity_type="asset", raw_text="那台设备",
                            resolution_status="ambiguous",
                            candidates=["A12风机", "A13风机"])
    assert entity.candidates == ["A12风机", "A13风机"]


# --------------------------------------------------------------------------
# plan integrity
# --------------------------------------------------------------------------

def test_duplicate_subtask_id_is_rejected():
    with pytest.raises(ValidationError, match="duplicate subtask_id"):
        plan([task("T1"), task("T1")])


def test_dependency_on_a_missing_subtask_is_rejected():
    with pytest.raises(ValidationError, match="depends on unknown"):
        plan([task("T1", depends_on=["T9"])])


def test_self_dependency_is_rejected():
    with pytest.raises(ValidationError, match="depends on itself"):
        plan([task("T1", depends_on=["T1"])])


def test_dependency_cycle_is_rejected():
    """The legacy executor would simply never run a cycle's members and report
    nothing about it."""
    with pytest.raises(ValidationError, match="dependency cycle"):
        plan([task("T1", depends_on=["T2"]), task("T2", depends_on=["T1"])])


def test_longer_dependency_cycle_is_rejected():
    with pytest.raises(ValidationError, match="dependency cycle"):
        plan([task("T1", depends_on=["T3"]), task("T2", depends_on=["T1"]),
              task("T3", depends_on=["T2"])])


def test_valid_dependency_chain_is_accepted():
    built = plan([task("T1"), task("T2", depends_on=["T1"]),
                  task("T3", depends_on=["T1", "T2"])])
    assert len(built.subtasks) == 3


def test_clarify_plan_without_a_question_is_rejected():
    with pytest.raises(ValidationError, match="clarification_question"):
        plan([task("T1", decision="clarify")], decision="clarify")


def test_partial_is_a_plan_level_decision_only():
    """A single subtask cannot be 'partial' — partial describes subtasks that
    disagree with each other."""
    with pytest.raises(ValidationError):
        task("T1", decision="partial")


def test_decision_counts_reports_the_mix():
    built = plan([task("T1", decision="execute"),
                  task("T2", decision="reject", intent="unsupported", selected_tool=None)],
                 decision="partial")
    assert built.decision_counts() == {"execute": 1, "reject": 1}


# --------------------------------------------------------------------------
# reason codes
# --------------------------------------------------------------------------

def test_reason_codes_are_aggregatable_enum_members():
    assert ReasonCode.PARTIAL_EXECUTION.value == "PARTIAL_EXECUTION"
    assert parse("TOOL_PROHIBITED") is ReasonCode.TOOL_PROHIBITED


def test_unknown_reason_code_is_rejected():
    """A typo in configs/routing.yaml must not create a category nobody counts."""
    with pytest.raises(ValueError, match="unknown reason code"):
        parse("NOT_A_REAL_CODE")


def test_unavailable_codes_are_separable_from_events():
    """Reports need to tell 'this happened' apart from 'we never recorded it'."""
    assert ReasonCode.MODEL_INFO_UNAVAILABLE_LEGACY in UNAVAILABLE_CODES
    assert ReasonCode.TOOL_PROHIBITED not in UNAVAILABLE_CODES


# --------------------------------------------------------------------------
# optional provenance
# --------------------------------------------------------------------------

def test_model_info_fields_are_all_optional():
    """Legacy records carry none of it, and a guessed model name reads as a
    measurement."""
    info = ModelInfo()
    assert (info.provider, info.model_name, info.temperature) == (None, None, None)


def test_tool_candidate_score_may_be_absent():
    candidate = ToolCandidate(tool_name="hybrid_search")
    assert candidate.score is None
    assert candidate.reason_codes == []
