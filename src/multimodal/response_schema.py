"""Validating what the model sent back.

A model reply is untrusted input. It is checked against a fixed shape before
any of it becomes evidence, because a reply that is merely *shaped* like an
answer will otherwise flow straight into the evidence set and from there into a
decision.

One repair attempt is allowed — stripping a markdown fence and re-parsing —
and no more. Repeated coaxing until something parses is how a refusal
("I can't read this") gets turned into a value.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from src.multimodal.schemas import VisionFieldResult

SCHEMA_VERSION = "vision_review_v1"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class SchemaValidationError(ValueError):
    pass


def extract_json(text: str) -> Dict[str, Any]:
    """Parse the reply, with exactly one repair attempt for a markdown fence."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _FENCE.search(text or "")
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"reply is not JSON: {exc}") from exc
    # Last resort: the outermost brace pair. Still a single attempt.
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"reply is not JSON: {exc}") from exc
    raise SchemaValidationError("reply contains no JSON object")


def validate_response(
    payload: Dict[str, Any], expected_fields: List[str],
) -> Tuple[List[VisionFieldResult], List[str]]:
    """Check the parsed reply and return (fields, unreadable_reasons).

    Rejects, rather than trims, a reply that answers about a field nobody
    asked for: that means the model was reading something other than the
    region it was given, and the rest of its answer is not trustworthy either.
    """
    if not isinstance(payload, dict):
        raise SchemaValidationError("top level is not an object")
    if not isinstance(payload.get("fields"), list):
        raise SchemaValidationError("`fields` missing or not a list")

    expected = set(expected_fields)
    out: List[VisionFieldResult] = []
    for index, raw in enumerate(payload["fields"]):
        if not isinstance(raw, dict):
            raise SchemaValidationError(f"fields[{index}] is not an object")
        name = raw.get("canonical_field_name")
        if not isinstance(name, str) or not name:
            raise SchemaValidationError(
                f"fields[{index}].canonical_field_name missing")
        if expected and name not in expected:
            raise SchemaValidationError(
                f"fields[{index}] reports {name!r}, which was not requested "
                f"({sorted(expected)})")

        value = raw.get("raw_value")
        if value is not None and not isinstance(value, str):
            raise SchemaValidationError(
                f"fields[{index}].raw_value must be a string or null")
        readable = raw.get("readable")
        if not isinstance(readable, bool):
            raise SchemaValidationError(
                f"fields[{index}].readable must be a boolean")
        # "readable with no value" is incoherent; treating it as readable would
        # let an empty string through as a recovered field.
        if readable and not value:
            raise SchemaValidationError(
                f"fields[{index}] claims readable=true with no raw_value")

        bbox = raw.get("evidence_bbox_in_crop") or []
        if not isinstance(bbox, list) or (bbox and len(bbox) != 4):
            raise SchemaValidationError(
                f"fields[{index}].evidence_bbox_in_crop must be [] or 4 numbers")

        out.append(VisionFieldResult(
            canonical_field_name=name,
            raw_name=raw.get("raw_name") if isinstance(raw.get("raw_name"), str) else None,
            raw_value=value,
            readable=readable,
            evidence_bbox_in_crop=[float(v) for v in bbox],
            notes=raw.get("notes") if isinstance(raw.get("notes"), str) else None,
        ))

    reasons = payload.get("unreadable_reasons") or []
    if not isinstance(reasons, list):
        raise SchemaValidationError("`unreadable_reasons` is not a list")
    return out, [str(r) for r in reasons]
