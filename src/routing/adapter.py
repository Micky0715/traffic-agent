"""Two-way conversion between the legacy IntentPlan and UnifiedIntentPlan.

The adapter is a bypass, not a replacement: every existing entry point keeps
working on the legacy type during phase 1.

Its job is to be honest about what the old format does not contain. Three
temptations are refused explicitly, because each would turn an absence into a
fabricated fact:

  the whole query is NOT used as every subtask's span — the old format never
  recorded which words a subtask came from, so query_span stays None;
  the chosen tool is NOT presented as a scored candidate — the old format
  records no candidate set, so tool_candidates stays empty;
  entity provenance is NOT asserted as "regex" — a legacy plan cannot say
  whether a value came from a pattern or a model, so the source is "legacy".

Each refusal carries a reason code, so a null can always be distinguished from
a value that was measured and happened to be empty.
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.models import Intent, IntentPlan, SubTask, ToolName
from src.routing.reason_codes import ReasonCode, parse as parse_reason_code
from src.routing.schema import (
    ModelInfo, ResolvedEntity, Span, SubTaskDecision, UnifiedIntentPlan,
    UnifiedSubTask,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "routing.yaml"


class SchemaAdaptationError(Exception):
    """Raised when a plan cannot be represented faithfully.

    Deliberately not recoverable by returning something. An adapter that
    answered a malformed plan with an empty plan, or quietly rewrote an
    unrecognised intent to general_qa, would convert a loud data problem into a
    silent wrong answer — and this repo has already paid for one of those.
    """


@lru_cache(maxsize=4)
def load_routing_config(path: str | None = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise SchemaAdaptationError(f"routing config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in ("taxonomy_version", "intents", "plan_mode_rules"):
        if key not in config:
            raise SchemaAdaptationError(f"routing config missing required key {key!r}")
    return config


def _intent_spec(config: Dict[str, Any], intent: str) -> Dict[str, Any]:
    spec = config["intents"].get(intent)
    if spec is None:
        raise SchemaAdaptationError(
            f"unknown intent {intent!r}: not declared in routing config "
            f"(taxonomy_version={config['taxonomy_version']})")
    return spec


def _map_reason_text(config: Dict[str, Any], text: str) -> List[ReasonCode]:
    codes: List[ReasonCode] = []
    for rule in config.get("legacy_reason_patterns") or []:
        if rule["match"] in text:
            codes.append(parse_reason_code(rule["code"]))
    return codes or [ReasonCode.REASON_UNMAPPED]


def _split_reason_texts(reason: str) -> List[str]:
    """The rule routers append repair notes onto one string with '；'."""
    if not reason:
        return []
    return [part.strip() for part in reason.split("；") if part.strip()]


def _subtask_decision(intent: str, missing_slots: List[str],
                      plan_needs_clarification: bool,
                      config: Dict[str, Any]) -> SubTaskDecision:
    """One subtask's own outcome.

    Deriving this per subtask is what makes `partial` possible: the legacy
    format could only say something about the whole request, so a query mixing
    a lookup with a forbidden write had to be forced entirely one way.
    """
    if intent in set(config.get("reject_intents") or []):
        return "reject"
    if missing_slots:
        return "clarify"
    if plan_needs_clarification:
        # The legacy validator escalates the whole plan when ANY subtask is
        # short a slot. Subtasks that are not themselves missing anything are
        # still blocked by that decision, so they are reported as clarify.
        return "clarify"
    return "execute"


def _plan_decision(subtask_decisions: List[SubTaskDecision]) -> str:
    if not subtask_decisions:
        raise SchemaAdaptationError(
            "empty plan: a plan with no subtasks cannot be represented. "
            "An empty plan is not 'fallback' — fallback is a deliberate "
            "routing outcome, not the absence of one.")
    distinct = set(subtask_decisions)
    if len(distinct) == 1:
        return distinct.pop()
    return "partial"


def _build_entities(slots: Dict[str, Any], missing_slots: List[str],
                    config: Dict[str, Any]) -> List[ResolvedEntity]:
    entity_types = config.get("slot_entity_types") or {}
    entity_slots = set(config.get("entity_slots") or [])
    entities: List[ResolvedEntity] = []

    for key, value in slots.items():
        entity_type = entity_types.get(key)
        if entity_type is None or key not in entity_slots:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item in (None, ""):
                continue
            entities.append(ResolvedEntity(
                entity_type=entity_type,
                raw_text=str(item),
                normalized_value=str(item),
                # A legacy plan cannot say whether this came from a regex or a
                # model. "legacy" is the truthful answer; "regex" would be a
                # provenance claim the data does not support.
                source="legacy",
                span=None,
                resolution_status="resolved",
            ))

    for key in missing_slots:
        entity_type = entity_types.get(key)
        if entity_type is None or key not in entity_slots:
            continue
        entities.append(ResolvedEntity(
            entity_type=entity_type, raw_text="", normalized_value=None,
            source="legacy", span=None, resolution_status="unresolved",
        ))
    return entities


def to_unified(
    legacy_plan: IntentPlan,
    *,
    original_query: str,
    source_strategy: str,
    taxonomy_version: Optional[str] = None,
    request_id: Optional[str] = None,
    model_info: Optional[ModelInfo] = None,
    prompt_version: Optional[str] = None,
    config_path: str | None = None,
) -> UnifiedIntentPlan:
    """Convert a legacy plan. The input object is never mutated."""
    if legacy_plan is None:
        raise SchemaAdaptationError("legacy_plan is None")

    config = load_routing_config(config_path)
    plan = deepcopy(legacy_plan)
    taxonomy = taxonomy_version or config["taxonomy_version"]

    if not plan.subtasks:
        raise SchemaAdaptationError(
            "empty plan: legacy plan has no subtasks. Refusing to emit a "
            "placeholder plan or downgrade to general_qa.")

    mode_rules = config["plan_mode_rules"]
    mode_rule = mode_rules.get(plan.plan_mode)
    if mode_rule is None:
        raise SchemaAdaptationError(
            f"unknown legacy plan_mode {plan.plan_mode!r}: no conversion rule in config")

    plan_reason_codes: List[ReasonCode] = [
        ReasonCode.QUERY_SPAN_UNAVAILABLE_LEGACY,
        ReasonCode.TOOL_CANDIDATES_UNAVAILABLE_LEGACY,
        ReasonCode.ENTITY_SOURCE_UNAVAILABLE_LEGACY,
    ]
    if model_info is None:
        plan_reason_codes.append(ReasonCode.MODEL_INFO_UNAVAILABLE_LEGACY)
    if prompt_version is None:
        plan_reason_codes.append(ReasonCode.PROMPT_VERSION_UNAVAILABLE_LEGACY)

    subtasks: List[UnifiedSubTask] = []
    decisions: List[SubTaskDecision] = []

    for task in plan.subtasks:
        intent_value = task.intent.value if hasattr(task.intent, "value") else str(task.intent)
        tool_value = task.tool.value if hasattr(task.tool, "value") else str(task.tool)
        spec = _intent_spec(config, intent_value)

        allowed = set(spec.get("allowed_tools") or [])
        if tool_value not in allowed and tool_value != "none":
            raise SchemaAdaptationError(
                f"tool {tool_value!r} is not allowed for intent {intent_value!r} "
                f"(allowed: {sorted(allowed)})")
        if tool_value not in {t.value for t in ToolName}:
            raise SchemaAdaptationError(f"unknown tool {tool_value!r}")

        decision = _subtask_decision(
            intent_value, list(task.missing_slots), plan.needs_clarification, config)
        decisions.append(decision)

        codes: List[ReasonCode] = []
        texts = _split_reason_texts(task.reason or "")
        for text in texts:
            codes.extend(_map_reason_text(config, text))
        for slot in task.missing_slots:
            codes.append(ReasonCode.REQUIRED_SLOT_MISSING)
            if slot in set(config.get("entity_slots") or []):
                codes.append(ReasonCode.ENTITY_MISSING)
            if slot == "time_range":
                codes.append(ReasonCode.TIME_RANGE_MISSING)
        if decision == "reject":
            codes.append(ReasonCode.TOOL_PROHIBITED)

        selected_tool = tool_value if tool_value != "none" else None
        if decision == "execute" and spec.get("requires_tool") and selected_tool is None:
            raise SchemaAdaptationError(
                f"subtask {task.id!r} with intent {intent_value!r} is executable and the "
                "intent requires a tool, but no tool was selected")

        parallel_group = mode_rule.get("parallel_group") if not task.depends_on else None

        subtasks.append(UnifiedSubTask(
            subtask_id=task.id,
            query_span=None,  # see module docstring
            intent=intent_value,
            decision=decision,
            entities=_build_entities(dict(task.slots), list(task.missing_slots), config),
            slots=dict(task.slots),
            required_slots=list(spec.get("required_slots") or []),
            missing_slots=list(task.missing_slots),
            tool_candidates=[],  # see module docstring
            selected_tool=selected_tool,
            depends_on=list(task.depends_on),
            parallel_group=parallel_group,
            reason_codes=_dedupe(codes),
            reason_texts=texts,
            confidence=task.confidence,
        ))

    plan_decision = _plan_decision(decisions)
    if plan_decision == "partial":
        plan_reason_codes.append(ReasonCode.PARTIAL_EXECUTION)

    try:
        return UnifiedIntentPlan(
            request_id=request_id or f"rt:{uuid.uuid4()}",
            taxonomy_version=taxonomy,
            decision=plan_decision,
            original_query=original_query,
            normalized_query=plan.normalized_query,
            subtasks=subtasks,
            reason_codes=_dedupe(plan_reason_codes),
            reason_texts=[],
            model_info=model_info,
            prompt_version=prompt_version,
            legacy_plan_mode=plan.plan_mode,
            source_strategy=source_strategy,
            clarification_question=plan.clarification_question,
            answer_constraints=list(plan.answer_constraints),
        )
    except ValueError as exc:
        raise SchemaAdaptationError(f"unified plan failed validation: {exc}") from exc


def to_legacy(unified_plan: UnifiedIntentPlan) -> IntentPlan:
    """Convert back, restoring exactly the fields the existing evaluation reads.

    Lossy by nature — query spans, reason codes, per-subtask decisions and tool
    candidates have no legacy home. That loss is one-directional and expected;
    what must not happen is a field the old evaluator reads coming back
    different, so intents, tools, slots, plan_mode and needs_clarification are
    restored verbatim.
    """
    if unified_plan is None:
        raise SchemaAdaptationError("unified_plan is None")
    if not unified_plan.subtasks:
        raise SchemaAdaptationError("empty plan: cannot convert back to legacy")

    plan_mode = unified_plan.legacy_plan_mode
    if plan_mode is None:
        raise SchemaAdaptationError(
            "cannot convert back to legacy: legacy_plan_mode is unset. A plan "
            "produced natively in the unified format has no legacy mode, and "
            "inventing one would change how the old evaluator scores it.")

    tasks: List[SubTask] = []
    for task in unified_plan.subtasks:
        try:
            intent = Intent(task.intent)
        except ValueError as exc:
            raise SchemaAdaptationError(f"unknown intent {task.intent!r}") from exc
        try:
            tool = ToolName(task.selected_tool) if task.selected_tool else ToolName.NONE
        except ValueError as exc:
            raise SchemaAdaptationError(f"unknown tool {task.selected_tool!r}") from exc

        tasks.append(SubTask(
            id=task.subtask_id,
            intent=intent,
            tool=tool,
            query=task.query_span.text if task.query_span else unified_plan.normalized_query,
            entities={},
            slots=dict(task.slots),
            missing_slots=list(task.missing_slots),
            depends_on=list(task.depends_on),
            confidence=task.confidence,
            reason="；".join(task.reason_texts),
        ))

    needs_clarification = unified_plan.decision in {"clarify", "partial"} and any(
        t.decision == "clarify" for t in unified_plan.subtasks)

    return IntentPlan(
        normalized_query=unified_plan.normalized_query,
        user_goal=unified_plan.original_query,
        needs_clarification=needs_clarification,
        clarification_question=unified_plan.clarification_question,
        plan_mode=plan_mode,
        subtasks=tasks,
        answer_constraints=list(unified_plan.answer_constraints),
    )


def legacy_request_id(dataset_name: str, case_id: str) -> str:
    """Deterministic id for batch conversion of historical records.

    A UUID would make every re-run of the converter produce different ids, so
    two conversions of the same frozen file could never be diffed.
    """
    return f"legacy:{dataset_name}:{case_id}"


def _dedupe(codes: List[ReasonCode]) -> List[ReasonCode]:
    seen = set()
    out = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out
