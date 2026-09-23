"""Unified routing data contract and legacy adapter.

Phase 1 of the routing rework: this package defines the contract and converts
between it and the legacy plan. It changes no routing behaviour — router_v1/v2/
v3, llm_router, pipeline, validator and evaluator are untouched, and every
existing entry point keeps operating on the legacy type.
"""
from src.routing.adapter import (
    SchemaAdaptationError, legacy_request_id, load_routing_config, to_legacy,
    to_unified,
)
from src.routing.reason_codes import UNAVAILABLE_CODES, ReasonCode
from src.routing.schema import (
    ModelInfo, ResolvedEntity, Span, ToolCandidate, UnifiedIntentPlan,
    UnifiedSubTask,
)

__all__ = [
    "SchemaAdaptationError", "legacy_request_id", "load_routing_config",
    "to_legacy", "to_unified", "ReasonCode", "UNAVAILABLE_CODES",
    "ModelInfo", "ResolvedEntity", "Span", "ToolCandidate",
    "UnifiedIntentPlan", "UnifiedSubTask",
]
