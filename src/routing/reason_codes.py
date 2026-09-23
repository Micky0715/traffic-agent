"""Aggregatable reason codes for routing decisions and schema adaptation.

The legacy plans carry a free-text Chinese `reason` ("v1按关键词优先级只选择一个标签").
That is readable and completely unaggregatable: you cannot count how often a
particular cause fired, group failures by it, or alert on it. These codes exist
so causes can be counted; the original text is never discarded, it travels
alongside in `reason_texts`.

Codes ending in `_UNAVAILABLE_LEGACY` mark information the old format simply
never recorded. They are not failures — they are the honest statement that a
field is unknown, and they exist so that a null cannot later be mistaken for a
measured value.
"""
from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    # --- adaptation / provenance -------------------------------------------
    SCHEMA_ADAPTATION_FAILED = "SCHEMA_ADAPTATION_FAILED"
    REASON_UNMAPPED = "REASON_UNMAPPED"
    QUERY_SPAN_UNAVAILABLE_LEGACY = "QUERY_SPAN_UNAVAILABLE_LEGACY"
    MODEL_INFO_UNAVAILABLE_LEGACY = "MODEL_INFO_UNAVAILABLE_LEGACY"
    PROMPT_VERSION_UNAVAILABLE_LEGACY = "PROMPT_VERSION_UNAVAILABLE_LEGACY"
    TOOL_CANDIDATES_UNAVAILABLE_LEGACY = "TOOL_CANDIDATES_UNAVAILABLE_LEGACY"
    ENTITY_SOURCE_UNAVAILABLE_LEGACY = "ENTITY_SOURCE_UNAVAILABLE_LEGACY"

    # --- entity / slot resolution ------------------------------------------
    ENTITY_MISSING = "ENTITY_MISSING"
    ENTITY_AMBIGUOUS = "ENTITY_AMBIGUOUS"
    REQUIRED_SLOT_MISSING = "REQUIRED_SLOT_MISSING"
    TIME_RANGE_MISSING = "TIME_RANGE_MISSING"

    # --- tool policy --------------------------------------------------------
    TOOL_PROHIBITED = "TOOL_PROHIBITED"
    TOOL_NOT_ALLOWED_FOR_INTENT = "TOOL_NOT_ALLOWED_FOR_INTENT"

    # --- planning -----------------------------------------------------------
    RULE_LLM_DISAGREEMENT = "RULE_LLM_DISAGREEMENT"
    DEPENDENCY_INVALID = "DEPENDENCY_INVALID"
    PARTIAL_EXECUTION = "PARTIAL_EXECUTION"


# Codes that record an unknown rather than an event. Useful for reports that
# want to separate "this happened" from "we never recorded this".
UNAVAILABLE_CODES = frozenset({
    ReasonCode.QUERY_SPAN_UNAVAILABLE_LEGACY,
    ReasonCode.MODEL_INFO_UNAVAILABLE_LEGACY,
    ReasonCode.PROMPT_VERSION_UNAVAILABLE_LEGACY,
    ReasonCode.TOOL_CANDIDATES_UNAVAILABLE_LEGACY,
    ReasonCode.ENTITY_SOURCE_UNAVAILABLE_LEGACY,
})


def parse(value: str) -> ReasonCode:
    """Strict lookup — an unrecognised code is a configuration error, not a
    value to pass through. Silently accepting unknown codes would let a typo in
    configs/routing.yaml produce a category nobody ever counts."""
    try:
        return ReasonCode(value)
    except ValueError as exc:
        raise ValueError(f"unknown reason code {value!r}") from exc
