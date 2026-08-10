from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Intent(str, Enum):
    REGULATION_LOOKUP = "regulation_lookup"
    EQUIPMENT_STATUS = "equipment_status"
    WORK_ORDER_QUERY = "work_order_query"
    METRIC_QUERY = "metric_query"
    DRAWING_LOOKUP = "drawing_lookup"
    FAULT_DIAGNOSIS = "fault_diagnosis"
    GENERAL_QA = "general_qa"
    UNSUPPORTED = "unsupported"


class ToolName(str, Enum):
    HYBRID_SEARCH = "hybrid_search"
    NL2API_QUERY = "nl2api_query"
    DRAWING_SEARCH = "drawing_search"
    NONE = "none"


class SubTask(BaseModel):
    id: str
    intent: Intent
    tool: ToolName
    query: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    slots: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    condition: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class IntentPlan(BaseModel):
    normalized_query: str
    user_goal: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    plan_mode: Literal[
        "none",
        "single",
        "parallel",
        "serial",
        "conditional",
        "parallel_then_synthesize",
        "clarify",
    ] = "none"
    subtasks: List[SubTask] = Field(default_factory=list)
    answer_constraints: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan(self) -> "IntentPlan":
        ids = [task.id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask id must be unique")
        valid_ids = set(ids)
        for task in self.subtasks:
            unknown = set(task.depends_on) - valid_ids
            if unknown:
                raise ValueError(f"unknown dependencies: {unknown}")
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("clarification_question is required")
        return self


class ToolResult(BaseModel):
    task_id: str
    tool: ToolName
    ok: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: int = 0
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationCase(BaseModel):
    id: str
    query: str
    gold_intents: List[Intent]
    gold_tools: List[ToolName]
    needs_clarification: bool
    plan_mode: str
    expected_slots: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class SufficiencyScores(BaseModel):
    coverage: float = Field(ge=0.0, le=1.0)
    entity_match: float = Field(ge=0.0, le=1.0)
    time_match: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    actionability: float = Field(ge=0.0, le=1.0)


class SufficiencyJudgment(BaseModel):
    sufficient: bool
    scores: SufficiencyScores
    missing_evidence: List[str] = Field(default_factory=list)
    next_action: Literal["retry", "clarify", "partial_answer", "answer", "refuse"]
    retry_plan: List[Any] = Field(default_factory=list)


class SufficiencyCase(BaseModel):
    id: str
    description: str
    query: str
    subtasks: List[SubTask]
    tool_results: List[ToolResult]
    gold_sufficient: bool
    gold_next_action: List[str]
    tags: List[str] = Field(default_factory=list)
    extra_context: str = ""


class DrawingExtraction(BaseModel):
    drawing_no: str = ""
    title: str = ""
    page: str = ""
    asset_id: str = ""
    key_fields: Dict[str, str] = Field(default_factory=dict)
    low_confidence_fields: List[str] = Field(default_factory=list)
    notes: str = ""
