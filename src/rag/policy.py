"""Required-evidence policy and the decision it produces.

There is deliberately no single blended sufficiency score. "0.78 overall" does
not say whether the missing fraction was a nice-to-have parameter or the device
identifier, and those two are not interchangeable: one is a slightly thinner
answer, the other dispatches somebody to the wrong machine. So the policy is a
checklist per task type, and the decision names exactly which requirement
failed.

Decision values reuse the routing contract's vocabulary
(execute / clarify / reject / fallback / partial) and add two this layer needs:

  abstain      — evidence exists but cannot be trusted enough to answer, and no
                 remediation is left. Distinct from `reject`, which means the
                 request itself is not permitted.
  human_review — high-risk field in unresolved conflict, or remediation
                 exhausted. Distinct from abstain in that a person is expected
                 to act.

The mapping to the routing enum is recorded in `ROUTING_DECISION_MAP` rather
than forcing either vocabulary to pretend the other's distinctions do not
exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.rag.config import EvidencePolicyConfig
from src.vision.schemas import DocumentChunk

# How this layer's decisions map onto src.routing.schema.PlanDecision.
# abstain and human_review both surface as `fallback` there — the routing
# contract has no vocabulary for "answerable but not trustworthy", so the finer
# value is kept here and the coarser one is what crosses the boundary.
ROUTING_DECISION_MAP = {
    "execute": "execute",
    "partial": "partial",
    "clarify": "clarify",
    "reject": "reject",
    "fallback": "fallback",
    "abstain": "fallback",
    "human_review": "fallback",
}

DECISIONS = tuple(ROUTING_DECISION_MAP)


@dataclass
class FieldEvidence:
    """One field the system believes it knows, and why it believes it."""

    field_name: str
    value: Optional[str]
    source: str                      # ocr | vlm | table_parser | tool | ledger
    document_id: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    entity_id: Optional[str] = None
    chunk_id: Optional[str] = None
    confidence: float = 0.0
    validation_status: str = "unknown"      # valid | invalid | unknown
    conflict_with: Optional[str] = None     # the other source's value, verbatim

    @property
    def traceable(self) -> bool:
        """A value you cannot point at on a page is not usable as evidence: no
        reviewer can check it and no answer can cite it."""
        return self.document_id is not None and self.page is not None

    @property
    def in_conflict(self) -> bool:
        return self.conflict_with is not None


@dataclass
class EvidenceBundle:
    task_type: str
    entity_id: Optional[str] = None
    metric_name: Optional[str] = None
    time_range: Optional[str] = None
    tool_available: bool = True
    fields: List[FieldEvidence] = field(default_factory=list)
    requested_fields: List[str] = field(default_factory=list)
    retrieved_chunks: List[DocumentChunk] = field(default_factory=list)
    remediation_rounds: int = 0
    remediation_exhausted: bool = False

    def by_name(self, name: str) -> Optional[FieldEvidence]:
        return next((f for f in self.fields if f.field_name == name), None)


@dataclass
class Decision:
    decision: str
    reason_codes: List[str] = field(default_factory=list)
    answerable_fields: Dict[str, str] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    unmet_requirements: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    remediation: List[str] = field(default_factory=list)
    routing_decision: str = "fallback"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "routing_decision": self.routing_decision,
            "answerable_fields": self.answerable_fields,
            "missing_fields": self.missing_fields,
            "conflicts": self.conflicts,
            "unmet_requirements": self.unmet_requirements,
            "reason_codes": self.reason_codes,
            "evidence_refs": self.evidence_refs,
            "remediation": self.remediation,
        }


class RequiredEvidencePolicy:
    """Checklist per task type. Each requirement is a named, separately
    reportable condition — never folded into one number."""

    def __init__(self, cfg: EvidencePolicyConfig):
        self.cfg = cfg

    # -- individual requirements ------------------------------------------
    def _check(self, name: str, bundle: EvidenceBundle) -> bool:
        if name == "entity_resolved":
            return bool(bundle.entity_id)
        if name == "metric_named":
            return bool(bundle.metric_name)
        if name == "time_range_satisfied":
            return bool(bundle.time_range)
        if name == "tool_available":
            return bundle.tool_available
        if name == "target_field_present":
            return any(f.value for f in bundle.fields
                       if f.field_name in (bundle.requested_fields or []))
        if name == "source_traceable":
            relevant = [f for f in bundle.fields
                        if not bundle.requested_fields
                        or f.field_name in bundle.requested_fields]
            return bool(relevant) and all(f.traceable for f in relevant)
        if name == "field_value_valid":
            return not any(f.validation_status == "invalid" for f in bundle.fields
                           if f.field_name in (bundle.requested_fields or []))
        if name == "no_unresolved_conflict":
            return not any(f.in_conflict for f in bundle.fields)
        if name == "clause_located":
            return any(c.content_type == "clause" for c in bundle.retrieved_chunks)
        raise ValueError(f"unknown evidence requirement {name!r}")

    def evaluate(self, bundle: EvidenceBundle) -> Decision:
        requirements = self.cfg.required_evidence.get(bundle.task_type)
        if requirements is None:
            raise ValueError(
                f"no required-evidence policy configured for task type "
                f"{bundle.task_type!r}. Refusing to fall back to a default: an "
                "unconfigured task silently answered under someone else's rules "
                "is exactly what this policy exists to prevent.")

        unmet = [name for name in requirements if not self._check(name, bundle)]

        answerable, missing, conflicts, refs = {}, [], [], []
        wanted = bundle.requested_fields or [f.field_name for f in bundle.fields]
        for name in wanted:
            evidence = bundle.by_name(name)
            if evidence is None or not evidence.value:
                missing.append(name)
                continue
            if evidence.in_conflict:
                conflicts.append({
                    "field_name": name, "value": evidence.value,
                    "conflicting_value": evidence.conflict_with,
                    "source": evidence.source,
                    # Neither side is picked. Preferring the VLM by default was
                    # falsified in this repo's own adversarial run, where OCR
                    # was right and the VLM misread a digit.
                    "resolution": "unresolved",
                })
                continue
            if evidence.validation_status == "invalid":
                missing.append(name)
                continue
            if evidence.confidence < self.cfg.min_field_confidence:
                missing.append(name)
                continue
            answerable[name] = evidence.value
            if evidence.traceable:
                refs.append({"document_id": evidence.document_id, "page": evidence.page,
                             "bbox": evidence.bbox, "chunk_id": evidence.chunk_id,
                             "source": evidence.source})

        return self._decide(bundle, unmet, answerable, missing, conflicts, refs)

    # -- the decision itself ----------------------------------------------
    def _decide(self, bundle, unmet, answerable, missing, conflicts, refs) -> Decision:
        reason_codes: List[str] = []
        remediation: List[str] = []

        high_risk_conflicts = [c for c in conflicts
                               if c["field_name"] in self.cfg.high_risk_fields]

        def build(decision: str) -> Decision:
            return Decision(
                decision=decision, reason_codes=reason_codes,
                answerable_fields=answerable, missing_fields=missing,
                conflicts=conflicts, unmet_requirements=unmet,
                evidence_refs=refs, remediation=remediation,
                routing_decision=ROUTING_DECISION_MAP[decision])

        # A high-risk field in unresolved conflict is never executed and never
        # quietly abstained: somebody has to look at it.
        if high_risk_conflicts:
            reason_codes.append("HIGH_RISK_FIELD_CONFLICT")
            return build("human_review")

        if bundle.remediation_exhausted or \
                bundle.remediation_rounds >= self.cfg.max_remediation_rounds:
            reason_codes.append("REMEDIATION_EXHAUSTED")
            # Partial evidence is still worth handing over; nothing at all is not.
            return build("partial" if answerable else "abstain")

        if conflicts:
            reason_codes.append("UNRESOLVED_FIELD_CONFLICT")
            return build("abstain")

        if not unmet and not missing:
            return build("execute")

        # A user-supplied parameter is the only thing a machine cannot fetch
        # for itself, so it is asked for rather than retried.
        if "entity_resolved" in unmet or "metric_named" in unmet or \
                "time_range_satisfied" in unmet:
            reason_codes.append("USER_PARAMETER_MISSING")
            remediation.append("ask_user")
            return build("clarify")

        if "tool_available" in unmet:
            reason_codes.append("TOOL_UNAVAILABLE")
            return build("abstain")

        if unmet:
            reason_codes.append("REQUIRED_EVIDENCE_MISSING")
            remediation.extend(["hybrid_retrieval", "parent_expansion"])
            if "target_field_present" in unmet or "source_traceable" in unmet:
                # Recorded as an intent; this round never calls a paid model.
                remediation.append("local_vlm_reidentify:not_invoked")
            return build("fallback")

        # Requirements met, some requested fields still missing: answer what is
        # supported and say what is not.
        reason_codes.append("PARTIAL_FIELD_COVERAGE")
        return build("partial" if answerable else "abstain")


@dataclass
class RemediationPlan:
    """What may be attempted, and how many times.

    A bounded loop is the point. An agent that keeps retrying a page it cannot
    read burns money and still cannot read it, and without a cap the failure
    mode is an infinite loop rather than a refusal.
    """

    max_rounds: int
    actions: List[str] = field(default_factory=list)
    performed: List[str] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return len(self.performed) >= self.max_rounds

    def next_action(self) -> Optional[str]:
        if self.exhausted:
            return None
        for action in self.actions:
            if action not in self.performed:
                return action
        return None

    def record(self, action: str) -> None:
        self.performed.append(action)
