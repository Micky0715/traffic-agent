"""The second decision pass.

    first RequiredEvidencePolicy -> fallback -> review requests ->
    VisionReviewExecutor -> fuse -> validate -> second RequiredEvidencePolicy

The rules about what the second pass may conclude are the substance here:

  * a VLM value that is legal, attributable and uncontested can lift abstain
    to partial or execute;
  * a conflict can NEVER produce execute, whichever side looks better;
  * an unreadable reply leaves the decision where it was;
  * a high-risk field standing on a single VLM reading goes to partial or
    human_review by configuration, not to execute.

And exactly one review round. `FallbackLedger` already caps this; the cap is
used rather than re-implemented so there is one place that decides when the
system stops paying to look again.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from src.multimodal.config import MultimodalReviewConfig
from src.multimodal.schemas import ReviewOutcome, VisionReviewResult
from src.rag.policy import Decision, EvidenceBundle, RequiredEvidencePolicy
from src.rag.policy import FieldEvidence as PolicyFieldEvidence
from src.vision.schemas import FieldEvidence as VisionFieldEvidence
from src.vision.schemas import FieldEvidenceSet

REVIEW_ACTION = "local_vision_review"


def _label(entity_id: Optional[str], field_name: str) -> str:
    return f"{entity_id}.{field_name}" if entity_id else field_name


def to_policy_fields(
    evidence_set: FieldEvidenceSet,
    *,
    document_id: str,
    page: int,
    chunk_id: Optional[str] = None,
) -> List[PolicyFieldEvidence]:
    """Cross from the vision evidence model to the policy's own.

    A conflicted field is handed over with `conflict_with` set, so the policy
    sees the disagreement itself rather than a value that happens to be one
    side of one. Which side gets to be `value` does not matter: the policy
    treats anything in conflict as not answerable.
    """
    conflicted: Dict[Tuple[Optional[str], str], str] = {}
    for conflict in evidence_set.conflicts:
        conflicted[(conflict.entity_id, conflict.field_name)] = conflict.vlm_value

    out: List[PolicyFieldEvidence] = []
    seen: set = set()
    for item in evidence_set.evidence:
        key = (item.entity_id, item.field_name)
        if key in seen:
            continue
        seen.add(key)
        out.append(PolicyFieldEvidence(
            field_name=item.field_name,
            value=item.raw_value,
            source=item.source,
            document_id=document_id,
            page=page,
            bbox=list(item.bbox) or None,
            entity_id=item.entity_id,
            chunk_id=chunk_id,
            confidence=item.confidence,
            validation_status=item.validation_status,
            conflict_with=conflicted.get(key),
        ))
    return out


def _sources_by_field(evidence_set: FieldEvidenceSet) -> Dict[Tuple, set]:
    out: Dict[Tuple, set] = {}
    for item in evidence_set.evidence:
        out.setdefault((item.entity_id, item.field_name), set()).add(item.source)
    return out


def apply_single_source_guard(
    decision: Decision,
    evidence_set: FieldEvidenceSet,
    cfg: MultimodalReviewConfig,
) -> Tuple[Decision, List[str]]:
    """Stop a high-risk field from reaching execute on one VLM reading alone.

    Nothing corroborates it: OCR did not produce that field, so "two sources
    agree" is unavailable and the model's own certainty is not a substitute.
    Downgrade target is configured, because how much a project will stake on an
    uncorroborated machine reading is a project decision.
    """
    sources = _sources_by_field(evidence_set)
    flagged = [
        _label(entity_id, field_name)
        for (entity_id, field_name), srcs in sorted(
            sources.items(), key=lambda kv: (kv[0][0] or "", kv[0][1]))
        if srcs == {"vlm"} and any(
            field_name.endswith(h) or h in field_name for h in cfg.high_risk_fields)
    ]
    if not flagged or decision.decision != "execute":
        return decision, flagged

    target = cfg.single_source_vlm_high_risk_decision
    downgraded = replace(
        decision,
        decision=target,
        reason_codes=sorted(set(list(decision.reason_codes) +
                                ["single_source_vlm_high_risk_field"])),
        routing_decision="partial" if target == "partial" else "fallback",
    )
    return downgraded, flagged


def run_second_pass(
    *,
    policy: RequiredEvidencePolicy,
    bundle_before: EvidenceBundle,
    decision_before: Decision,
    fused: FieldEvidenceSet,
    review_results: Sequence[VisionReviewResult],
    cfg: MultimodalReviewConfig,
    document_id: str,
    page: int = 1,
    chunk_id: Optional[str] = None,
) -> ReviewOutcome:
    """Re-decide on the merged evidence and record what moved."""
    before_fields = {(f.entity_id, f.field_name) for f in bundle_before.fields}
    before_conflicts = set()
    for f in bundle_before.fields:
        if f.in_conflict:
            before_conflicts.add(_label(f.entity_id, f.field_name))

    policy_fields = to_policy_fields(
        fused, document_id=document_id, page=page, chunk_id=chunk_id)

    bundle_after = replace(
        bundle_before,
        fields=policy_fields,
        remediation_rounds=bundle_before.remediation_rounds + 1,
        # One review round, and the ledger says so. A second pass that could
        # request another review is a loop with a bill attached.
        remediation_exhausted=True,
    )
    decision_after = policy.evaluate(bundle_after)
    decision_after, single_source = apply_single_source_guard(
        decision_after, fused, cfg)

    recovered = [
        _label(item.entity_id, item.field_name)
        for item in fused.evidence
        if item.source == "vlm"
        and (item.entity_id, item.field_name) not in before_fields
    ]
    after_conflicts = {_label(c.entity_id, c.field_name) for c in fused.conflicts}

    return ReviewOutcome(
        document_id=document_id,
        decision_before_review=decision_before.decision,
        decision_after_review=decision_after.decision,
        fields_recovered=sorted(set(recovered)),
        conflicts_created=sorted(after_conflicts - before_conflicts),
        conflicts_resolved=sorted(before_conflicts - after_conflicts),
        still_missing_fields=sorted(decision_after.missing_fields)
        if hasattr(decision_after, "missing_fields") else [],
        single_source_vlm_fields=single_source,
        review_results=list(review_results),
        notes={
            "routing_decision_before": decision_before.routing_decision,
            "routing_decision_after": decision_after.routing_decision,
            "reason_codes_after": sorted(decision_after.reason_codes),
            "review_rounds": 1,
            "review_rounds_capped_at": 1,
        },
    )
