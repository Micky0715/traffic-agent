"""Checks whether a field's VALUE has a plausible shape for that field.

The gap this closes: field completeness asks "was anything read into this
slot", and a stamp that lands on top of a cabinet number leaves a slot that is
occupied but wrong. Label present, value present, completeness 1.0, OCR
confidence 0.986 — and the value reads as Chinese stamp text rather than an
equipment code. Every signal in the previous round says the page is fine.

Two hard rules govern everything here:

  A value is never repaired. Nothing in this module rewrites, trims into
  shape, or regex-corrects a business value. Normalization is limited to
  transformations that cannot change meaning (see normalize_value), and both
  the raw and normalized forms are kept.

  Absence of a rule is not evidence of a problem. A field with no configured
  rule returns "unknown" and is excluded from the ratio denominator entirely.
  Treating unwritten rules as failures would manufacture exactly the kind of
  signal this repo is trying not to manufacture.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence

from src.vision.config import (
    DrawingTypeConfig, ValueRuleDefinition, ValueValidationConfig,
)
from src.vision.schemas import (
    FieldPair, InvalidFieldDetail, ValueValidationResult,
)


def normalize_value(
    raw: str, hyphen_variants: Sequence[str], collapse_spaces_around_hyphen: bool = True
) -> str:
    """Non-semantic cleanup only, applied before any rule is evaluated.

    NFKC folds full-width forms onto their ASCII equivalents, which is what
    lets a code typed or scanned in full-width characters compare equal to the
    same code in ASCII. Dash variants are unified separately because NFKC does
    NOT map en/em dashes or the minus sign onto the hyphen, and OCR routinely
    returns them in place of one. Whitespace runs collapse to a single space
    and the ends are trimmed.

    Deliberately NOT done here: deleting spaces inside a code, uppercasing,
    stripping stray characters, or coercing anything toward a pattern. Each of
    those is a guess about business content, and a guess that succeeds hides
    the very error this module is looking for.
    """
    text = unicodedata.normalize("NFKC", raw)
    for variant in hyphen_variants:
        text = text.replace(variant, "-")
    text = re.sub(r"\s+", " ", text).strip()
    if collapse_spaces_around_hyphen:
        # OCR inserts spacing around a hyphen it can see perfectly well; "M -
        # 13" and "M-13" are the same code, so removing that spacing cannot
        # change which piece of equipment is named. Left configurable because
        # it is the one normalization step here that is a judgement about
        # codes rather than a pure Unicode fold.
        text = re.sub(r"\s*-\s*", "-", text)
    return text


def _rules_for_field(
    field_name: str,
    cfg: ValueValidationConfig,
    drawing_type_spec: Optional[DrawingTypeConfig],
) -> List[str]:
    """Rule ids that apply to this field, type-specific config winning.

    Naming-prefix rules live in a drawing type's overrides rather than in the
    global defaults on purpose: a prefix convention belongs to a specific
    project's equipment-numbering scheme, and this repo has no such scheme
    document to cite. The override hook exists and is exercised by tests; it
    ships empty rather than shipping a prefix inferred from this dataset.
    """
    if drawing_type_spec is not None:
        override = drawing_type_spec.field_value_rule_overrides.get(field_name)
        if override is not None:
            return list(override)
    return list(cfg.field_value_rules.get(field_name, []))


def _evaluate_rule(rule: ValueRuleDefinition, value: str) -> bool:
    return re.search(rule.pattern, value) is not None


def validate_field_values(
    pairs: Sequence[FieldPair],
    cfg: ValueValidationConfig,
    drawing_type_spec: Optional[DrawingTypeConfig] = None,
) -> ValueValidationResult:
    """Validate every recovered label->value pair on one page.

    A field is checked only when a rule exists for it. checked_fields is
    therefore the ratio's denominator, and unknown_fields sits outside it —
    see the module docstring for why that distinction is load-bearing.
    """
    checked: List[str] = []
    valid: List[str] = []
    invalid: List[str] = []
    unknown: List[str] = []
    details: List[InvalidFieldDetail] = []

    for pair in pairs:
        label = _qualified_name(pair)
        rule_ids = _rules_for_field(pair.field_name, cfg, drawing_type_spec)
        if not rule_ids:
            unknown.append(label)
            continue

        normalized = normalize_value(
            pair.raw_value, cfg.hyphen_variants, cfg.collapse_spaces_around_hyphen)
        checked.append(label)
        broken = None
        for rule_id in rule_ids:
            rule = cfg.value_rule_definitions.get(rule_id)
            if rule is None:  # configured but undefined: report, do not guess
                broken = ValueRuleDefinition(
                    pattern="", reason=f"rule {rule_id} is referenced but not defined",
                    origin="configuration_error")
                break
            if not _evaluate_rule(rule, normalized):
                broken = rule
                break

        if broken is None:
            valid.append(label)
        else:
            invalid.append(label)
            details.append(InvalidFieldDetail(
                field_name=pair.field_name,
                entity_id=pair.entity_id,
                observed_value=pair.raw_value,
                normalized_value=normalized,
                violated_rule=next(
                    (rid for rid in rule_ids if cfg.value_rule_definitions.get(rid) is broken),
                    rule_ids[0]),
                rule_pattern=broken.pattern,
                rule_origin=broken.origin,
                reason=broken.reason,
            ))

    ratio: Optional[float] = None
    if checked:
        ratio = len(valid) / len(checked)

    return ValueValidationResult(
        checked_fields=checked, valid_fields=valid, invalid_fields=invalid,
        unknown_fields=unknown, invalid_details=details, valid_ratio=ratio,
    )


def _qualified_name(pair: FieldPair) -> str:
    return f"{pair.entity_id}{pair.field_name}" if pair.entity_id else pair.field_name


def validity_status(result: ValueValidationResult, field_label: str) -> str:
    if field_label in result.invalid_fields:
        return "invalid"
    if field_label in result.valid_fields:
        return "valid"
    return "unknown"


def value_validity_below_threshold(
    result: ValueValidationResult, cfg: ValueValidationConfig
) -> bool:
    """Whether the page-level validity RATE is low enough to act on.

    Gated on a minimum number of checked fields. With one or two checkable
    fields the ratio jumps between 0.0, 0.5 and 1.0 on a single reading, and a
    threshold over that is noise, not measurement. A field that is outright
    invalid still raises its own signal independently of this rate — see
    route_ocr_page.
    """
    if result.valid_ratio is None:
        return False
    if len(result.checked_fields) < cfg.min_checked_fields:
        return False
    return result.valid_ratio < cfg.min_valid_ratio


def field_pairs_from_completeness(
    pairs: Sequence[FieldPair], keep_empty: bool = False
) -> List[FieldPair]:
    """Pairs worth validating: those that actually carry a value.

    An empty value is a completeness problem, already reported as a missing
    field or an isolated label, and running shape rules over "" would
    double-count one fault as two independent signals.
    """
    if keep_empty:
        return list(pairs)
    return [p for p in pairs if p.raw_value.strip()]
