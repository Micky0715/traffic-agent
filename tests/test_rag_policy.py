"""Required-evidence policy and the abstention closed loop.

SYNTHETIC TEST FIXTURES only. No model, tool or service is called.
"""
from __future__ import annotations

import pytest

from src.rag.config import EvidencePolicyConfig, load_rag_config
from src.rag.policy import (
    DECISIONS, ROUTING_DECISION_MAP, EvidenceBundle, FieldEvidence,
    RemediationPlan, RequiredEvidencePolicy,
)
from src.vision.schemas import DocumentChunk

SYNTHETIC_TEST_FIXTURE = True


@pytest.fixture
def policy() -> RequiredEvidencePolicy:
    return RequiredEvidencePolicy(load_rag_config().evidence_policy)


def field(name, value, **kwargs):
    base = {"source": "vlm", "document_id": "drawing_001", "page": 3,
            "bbox": [120, 340, 930, 410], "entity_id": "A16",
            "chunk_id": "drawing_field:doc:abc", "confidence": 0.9,
            "validation_status": "valid"}
    base.update(kwargs)
    return FieldEvidence(field_name=name, value=value, **base)


def drawing_bundle(**overrides) -> EvidenceBundle:
    payload = {
        "task_type": "drawing_field_query", "entity_id": "A16",
        "requested_fields": ["功率"], "fields": [field("功率", "45kW")],
    }
    payload.update(overrides)
    return EvidenceBundle(**payload)


# --------------------------------------------------------------------------
# 1-3. the straightforward outcomes
# --------------------------------------------------------------------------

def test_complete_evidence_executes(policy):
    decision = policy.evaluate(drawing_bundle())
    assert decision.decision == "execute"
    assert decision.answerable_fields == {"功率": "45kW"}
    assert decision.evidence_refs[0]["page"] == 3


def test_missing_user_parameter_asks_rather_than_retrying(policy):
    """A parameter only the user has is the one thing retrying cannot fetch."""
    decision = policy.evaluate(EvidenceBundle(
        task_type="equipment_metric_query", entity_id=None, metric_name="告警次数",
        time_range="最近3天", requested_fields=["告警次数"],
        fields=[field("告警次数", "12", source="tool")]))
    assert decision.decision == "clarify"
    assert "USER_PARAMETER_MISSING" in decision.reason_codes
    assert decision.remediation == ["ask_user"]


def test_some_fields_reliable_yields_partial(policy):
    decision = policy.evaluate(drawing_bundle(
        requested_fields=["功率", "控制柜编号"],
        fields=[field("功率", "45kW")]))
    assert decision.decision in {"partial", "fallback"}
    assert decision.answerable_fields == {"功率": "45kW"}
    assert "控制柜编号" in decision.missing_fields


# --------------------------------------------------------------------------
# 4-5. conflicts and invalid values
# --------------------------------------------------------------------------

def test_conflict_on_a_non_high_risk_field_abstains(policy):
    decision = policy.evaluate(drawing_bundle(
        fields=[field("功率", "45kW", conflict_with="55kW")]))
    assert decision.decision == "abstain"
    assert "UNRESOLVED_FIELD_CONFLICT" in decision.reason_codes
    assert decision.conflicts[0]["resolution"] == "unresolved"


def test_high_risk_field_conflict_goes_to_a_person(policy):
    """A wrong cabinet number dispatches somebody to the wrong equipment, so it
    is never silently abstained either — a human has to look."""
    decision = policy.evaluate(drawing_bundle(
        requested_fields=["控制柜编号"],
        fields=[field("控制柜编号", "FAN-CAB-16", conflict_with="审核专用章")]))
    assert decision.decision == "human_review"
    assert "HIGH_RISK_FIELD_CONFLICT" in decision.reason_codes


def test_neither_source_is_preferred_in_a_conflict(policy):
    """Defaulting to the VLM was falsified by this repo's own adversarial run,
    where OCR was right and the VLM misread a digit."""
    decision = policy.evaluate(drawing_bundle(
        fields=[field("功率", "45kW", source="ocr", conflict_with="55kW")]))
    conflict = decision.conflicts[0]
    assert conflict["value"] == "45kW" and conflict["conflicting_value"] == "55kW"
    assert conflict["resolution"] == "unresolved"


def test_invalid_field_value_is_not_answered(policy):
    decision = policy.evaluate(drawing_bundle(
        fields=[field("功率", "严禁外传", validation_status="invalid")]))
    assert decision.decision in {"fallback", "abstain", "partial"}
    assert "功率" not in decision.answerable_fields


# --------------------------------------------------------------------------
# 6. bounded remediation
# --------------------------------------------------------------------------

def test_remediation_is_capped(policy):
    plan = RemediationPlan(max_rounds=2, actions=["hybrid_retrieval",
                                                  "parent_expansion",
                                                  "local_vlm_reidentify"])
    plan.record(plan.next_action())
    plan.record(plan.next_action())
    assert plan.exhausted and plan.next_action() is None


def test_exhausted_remediation_abstains_rather_than_looping(policy):
    decision = policy.evaluate(drawing_bundle(
        requested_fields=["控制柜编号"], fields=[], remediation_exhausted=True))
    assert decision.decision == "abstain"
    assert "REMEDIATION_EXHAUSTED" in decision.reason_codes


def test_exhausted_remediation_still_returns_what_is_supported(policy):
    decision = policy.evaluate(drawing_bundle(
        requested_fields=["功率", "控制柜编号"], fields=[field("功率", "45kW")],
        remediation_exhausted=True))
    assert decision.decision == "partial"
    assert decision.answerable_fields == {"功率": "45kW"}


def test_vlm_reidentification_is_planned_but_marked_not_invoked(policy):
    decision = policy.evaluate(drawing_bundle(
        requested_fields=["控制柜编号"], fields=[]))
    assert any("not_invoked" in action for action in decision.remediation)


# --------------------------------------------------------------------------
# 7-9. the traps
# --------------------------------------------------------------------------

def test_a_missing_non_required_field_does_not_block_everything(policy):
    """One absent extra must not veto the fields that are fully supported."""
    decision = policy.evaluate(drawing_bundle(
        requested_fields=["功率", "备注"],
        fields=[field("功率", "45kW"), field("备注", None)]))
    assert decision.answerable_fields == {"功率": "45kW"}
    assert decision.decision != "reject"


def test_high_ocr_confidence_with_a_missing_field_does_not_execute(policy):
    """The silent-miss case: confidence describes the characters that WERE
    returned and says nothing about the ones that were not."""
    decision = policy.evaluate(drawing_bundle(
        requested_fields=["功率", "控制柜编号"],
        fields=[field("功率", "45kW", confidence=0.999)]))
    assert decision.decision != "execute"
    assert "控制柜编号" in decision.missing_fields


def test_a_value_with_no_traceable_source_is_not_decision_ready(policy):
    """A string appearing somewhere is not evidence: without a page it cannot
    be cited and cannot be checked."""
    decision = policy.evaluate(drawing_bundle(
        fields=[field("功率", "45kW", document_id=None, page=None)]))
    assert decision.decision != "execute"
    assert "source_traceable" in decision.unmet_requirements


def test_unconfigured_task_type_raises_instead_of_guessing(policy):
    with pytest.raises(ValueError, match="no required-evidence policy"):
        policy.evaluate(EvidenceBundle(task_type="some_new_task"))


def test_clause_lookup_requires_a_clause_chunk(policy):
    without = policy.evaluate(EvidenceBundle(
        task_type="regulation_lookup", requested_fields=[],
        fields=[field("条文", "应急照明…", document_id="reg1", page=1)],
        retrieved_chunks=[]))
    assert "clause_located" in without.unmet_requirements

    with_clause = policy.evaluate(EvidenceBundle(
        task_type="regulation_lookup", requested_fields=[],
        fields=[field("条文", "应急照明…", document_id="reg1", page=1)],
        retrieved_chunks=[DocumentChunk(
            text="4.2.1 应急照明…", parent_id="reg1", page_number=1,
            document_id="reg1", content_type="clause")]))
    assert with_clause.decision == "execute"


# --------------------------------------------------------------------------
# vocabulary compatibility
# --------------------------------------------------------------------------

def test_every_decision_maps_onto_the_routing_contract():
    """abstain and human_review have no routing equivalent, so they surface as
    fallback there rather than inventing a conflicting enum value."""
    from src.routing.schema import UnifiedIntentPlan  # noqa: F401
    allowed = {"execute", "clarify", "reject", "fallback", "partial"}
    assert set(ROUTING_DECISION_MAP.values()) <= allowed
    assert set(DECISIONS) == set(ROUTING_DECISION_MAP)
    assert ROUTING_DECISION_MAP["abstain"] == "fallback"
    assert ROUTING_DECISION_MAP["human_review"] == "fallback"


def test_refusal_explains_itself_rather_than_citing_low_confidence(policy):
    decision = policy.evaluate(drawing_bundle(
        requested_fields=["功率", "控制柜编号"], fields=[field("功率", "45kW")]))
    payload = decision.to_dict()
    assert payload["missing_fields"] == ["控制柜编号"]
    assert payload["reason_codes"]
    assert payload["evidence_refs"][0]["bbox"] == [120, 340, 930, 410]
    assert "confidence" not in str(payload["reason_codes"]).lower()
