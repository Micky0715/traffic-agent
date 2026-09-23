"""Unified routing data contract.

Named UnifiedIntentPlan rather than IntentPlan so it cannot be confused with
the legacy type in src/models.py — both exist during the migration and a
shared name would make every import ambiguous.

Three things the legacy format could not express, and why each matters:

  decision as a first-class field. The old plan encodes its outcome across
  `needs_clarification` plus a 7-value `plan_mode`, and a refusal is smuggled
  in as an *intent* (UNSUPPORTED). There was no way to say "half of this
  request is executable and half is not", so a query mixing a lookup with a
  delete had to be forced entirely one way. `partial` is that missing state.

  reason codes. The old `reason` is free Chinese text, which cannot be counted
  or grouped. Codes make causes aggregatable; the original text is preserved
  in `reason_texts` rather than replaced.

  provenance. Nothing recorded which strategy, model, prompt version or
  taxonomy produced a plan, so two results could never be told apart after the
  fact. Optional fields here are genuinely optional — for legacy data they stay
  None and carry an *_UNAVAILABLE_LEGACY code, which is a statement that the
  value was never recorded, not a default.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.routing.reason_codes import ReasonCode

PlanDecision = Literal["execute", "clarify", "reject", "fallback", "partial"]
# A subtask is never "partial" — partial describes a plan whose subtasks
# disagree, so it cannot apply to a single subtask.
SubTaskDecision = Literal["execute", "clarify", "reject", "fallback"]
SourceStrategy = Literal["rule_v1", "rule_v2", "rule_v3", "llm", "legacy_unknown"]
EntitySource = Literal["regex", "llm", "ledger", "user", "legacy"]
ResolutionStatus = Literal["resolved", "ambiguous", "unresolved"]


class Span(BaseModel):
    """A half-open character range in the original query, plus its text.

    `text` is stored redundantly on purpose: the adapter verifies it against
    the original query, which catches an offset computed on a normalized string
    and applied to a raw one. Without the check, a span silently points at the
    wrong words and every later span-based metric is quietly wrong.
    """

    start: int
    end: int
    text: str

    @model_validator(mode="after")
    def validate_range(self) -> "Span":
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span range [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"span text length {len(self.text)} does not match range "
                f"[{self.start}, {self.end})")
        return self


class ModelInfo(BaseModel):
    """What produced a plan. Every field is optional because legacy records
    carry none of it — and a guessed model name is worse than a null, since it
    reads as a measurement."""

    provider: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    temperature: Optional[float] = None
    request_id: Optional[str] = None


class ToolCandidate(BaseModel):
    """A tool that was considered, with why.

    Legacy plans expose only the tool that was chosen, with no candidate set
    and no scores. Those convert to an EMPTY candidate list plus
    TOOL_CANDIDATES_UNAVAILABLE_LEGACY — dressing the single chosen tool up as
    a scored candidate would invent a selection process that never ran.
    """

    tool_name: str
    score: Optional[float] = None
    reason_codes: List[ReasonCode] = Field(default_factory=list)
    source: Optional[str] = None


class ResolvedEntity(BaseModel):
    """One entity mention and what became of it.

    `source` says which mechanism produced it. Legacy data cannot distinguish a
    regex extraction from an LLM one, so it converts to "legacy" — writing
    "regex" for everything would be a provenance claim the data does not
    support.
    """

    entity_type: str
    raw_text: str
    normalized_value: Optional[str] = None
    source: EntitySource = "legacy"
    span: Optional[Span] = None
    resolution_status: ResolutionStatus = "resolved"
    candidates: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> "ResolvedEntity":
        if self.resolution_status == "resolved" and self.normalized_value is None:
            raise ValueError(
                f"entity {self.entity_type!r} is marked resolved but has no "
                "normalized_value")
        return self


class UnifiedSubTask(BaseModel):
    subtask_id: str
    # None for anything converted from a legacy plan: the old format never
    # recorded which part of the query a subtask came from, and reconstructing
    # it by searching for keywords would be a guess dressed as provenance.
    query_span: Optional[Span] = None
    intent: str
    decision: SubTaskDecision = "execute"
    entities: List[ResolvedEntity] = Field(default_factory=list)
    slots: Dict[str, Any] = Field(default_factory=dict)
    required_slots: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    tool_candidates: List[ToolCandidate] = Field(default_factory=list)
    selected_tool: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    parallel_group: Optional[str] = None
    reason_codes: List[ReasonCode] = Field(default_factory=list)
    reason_texts: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class UnifiedIntentPlan(BaseModel):
    request_id: str
    taxonomy_version: str
    decision: PlanDecision
    original_query: str
    normalized_query: str = ""
    subtasks: List[UnifiedSubTask] = Field(default_factory=list)
    reason_codes: List[ReasonCode] = Field(default_factory=list)
    reason_texts: List[str] = Field(default_factory=list)
    model_info: Optional[ModelInfo] = None
    prompt_version: Optional[str] = None
    # The legacy 7-value plan_mode, carried through untouched. The old
    # evaluation scores against it, so losing it would break every historical
    # comparison; it is kept as legacy metadata rather than promoted, because
    # parallel_group / depends_on express the same thing structurally.
    legacy_plan_mode: Optional[str] = None
    source_strategy: SourceStrategy = "legacy_unknown"
    clarification_question: Optional[str] = None
    answer_constraints: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan(self) -> "UnifiedIntentPlan":
        ids = [task.subtask_id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate subtask_id: {duplicates}")

        known = set(ids)
        for task in self.subtasks:
            unknown = [d for d in task.depends_on if d not in known]
            if unknown:
                raise ValueError(
                    f"subtask {task.subtask_id!r} depends on unknown subtask(s) {unknown}")
            if task.subtask_id in task.depends_on:
                raise ValueError(f"subtask {task.subtask_id!r} depends on itself")

        _assert_acyclic(self.subtasks)

        if self.clarification_question is None and self.decision == "clarify":
            # A clarify decision with nothing to ask is unusable downstream.
            raise ValueError("decision is 'clarify' but clarification_question is missing")
        return self

    def decision_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task in self.subtasks:
            counts[task.decision] = counts.get(task.decision, 0) + 1
        return counts


def _assert_acyclic(subtasks: List[UnifiedSubTask]) -> None:
    """A dependency cycle makes the plan unexecutable, and the legacy executor
    would simply never run the cycle's members while reporting nothing. Caught
    here instead."""
    graph = {t.subtask_id: list(t.depends_on) for t in subtasks}
    state: Dict[str, int] = {}  # 0 = unvisited, 1 = on stack, 2 = done

    def visit(node: str, stack: List[str]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cycle = stack[stack.index(node):] + [node]
            raise ValueError(f"dependency cycle: {' -> '.join(cycle)}")
        state[node] = 1
        for parent in graph.get(node, []):
            visit(parent, stack + [node])
        state[node] = 2

    for node in graph:
        visit(node, [])
