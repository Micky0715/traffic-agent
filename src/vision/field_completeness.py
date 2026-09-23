"""Detects SILENT OCR misses — pages where recognition confidence is high but
whole fields were never read at all.

Why this module exists: OCR confidence answers "how sure am I about the
characters I returned", and says nothing about what it failed to return. On
FAN-A23-01 a watermark covers the value cells; PaddleOCR reads the watermark
text itself with 0.996 average confidence while 3 of 4 target fields are
missing entirely. Confidence-based routing waves that page straight through.

The four signals here are all about absence rather than uncertainty:
  field completeness    how many of the fields this KIND of drawing should
                        carry were actually read, label and value both
  isolated labels       a label was read but nothing sits where its value
                        belongs — the specific shape of occlusion damage
  watermark repetition  the same boilerplate string recurring across blocks,
                        which is what covering text looks like to OCR
  group cardinality     for multi-device drawings, whether the number of
                        devices can be trusted at all (see below)

Everything is computed from OCR output alone. Nothing in this module may read
gold/expected answers: routing decides whether a page needs help BEFORE
anyone knows whether it got the answer right, and wiring the answer into that
decision would make every downstream metric circular.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.vision.config import DrawingTypeConfig, FieldCompletenessConfig
from src.vision.schemas import (
    FieldCompleteness, FieldPair, OCRBlock, OCRResult, TableStructure,
)

# Field-combination fallback (priority 3): which labels hint at which type
# when neither the title nor the drawing number settled it.
_WIRING_HINT_LABELS = ("电机编号", "断路器编号", "控制柜编号")
_GROUP_HINT_SUFFIXES = ("功率", "风量", "流量", "扬程")


def _height(block: OCRBlock) -> float:
    return block.bbox[3] - block.bbox[1] if len(block.bbox) == 4 else 0.0


def _vertical_overlap_ratio(a: OCRBlock, b: OCRBlock) -> float:
    """Fraction of the SHORTER block's height that overlaps the other's y
    range. Using the shorter one means a tall watermark block cannot score a
    high overlap just by spanning the whole page."""
    if len(a.bbox) != 4 or len(b.bbox) != 4:
        return 0.0
    top, bottom = max(a.bbox[1], b.bbox[1]), min(a.bbox[3], b.bbox[3])
    if bottom <= top:
        return 0.0
    shorter = min(_height(a), _height(b))
    return (bottom - top) / shorter if shorter > 0 else 0.0


def find_value_block(
    label: OCRBlock, blocks: Sequence[OCRBlock], cfg: FieldCompletenessConfig
) -> Optional[OCRBlock]:
    """The block holding this label's value, or None if the value slot is empty.

    A value is printed on the same line as its label: to the right of it,
    vertically overlapping it, not absurdly far away, and — the part that
    matters — of comparable height. Without the height test, FAN-A23-01's
    watermark block (192px tall, spanning half the page) satisfies "to the
    right and overlapping" for every label on the page, and the occlusion
    becomes invisible.
    """
    if len(label.bbox) != 4:
        return None
    label_height = _height(label)
    best: Optional[Tuple[float, OCRBlock]] = None
    for block in blocks:
        if block is label or len(block.bbox) != 4:
            continue
        gap = block.bbox[0] - label.bbox[2]
        if gap < 0 or gap > cfg.label_value_max_gap_px:
            continue
        if label_height > 0 and _height(block) > label_height * cfg.label_value_height_tolerance:
            continue
        if _vertical_overlap_ratio(label, block) < cfg.label_value_min_vertical_overlap:
            continue
        if best is None or gap < best[0]:
            best = (gap, block)
    return best[1] if best else None


def watermark_repetition(text: str, ngram: int, min_repeats: int) -> float:
    """Share of the text made up of character n-grams repeated at least
    min_repeats times.

    Both parameters were forced by measurement, not chosen for elegance. The
    first version of this counted any n-gram appearing twice, with n=4, and
    fired on 6 of 7 development pages INCLUDING every clean one: normal
    drawings repeat text legitimately — the title is echoed in the 名称 field,
    and 电机编号/断路器编号/控制柜编号 share characters. "Contains repetition"
    does not distinguish a watermark from a title block.

    What does distinguish it is that a watermark TILES: one longish phrase
    stamped many times across the page. Requiring a longer n-gram repeated at
    least three times separates them cleanly on the development set —
    watermarked page 0.573, clean group table 0.055, everything else 0.000.
    A title echoed into a field appears exactly twice and never reaches the
    threshold.
    """
    compact = re.sub(r"\s+", "", text)
    if len(compact) < ngram * 2:
        return 0.0
    grams = [compact[i:i + ngram] for i in range(len(compact) - ngram + 1)]
    counts: Dict[str, int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    repeated = sum(count for count in counts.values() if count >= min_repeats)
    return repeated / len(grams)


def classify_drawing_type(
    ocr: OCRResult, drawing_types: Dict[str, DrawingTypeConfig]
) -> Tuple[str, str]:
    """Returns (type_name, how_it_was_decided).

    Priority is deliberate: a drawing's own title is the most trustworthy
    statement of what it is, the drawing-number prefix is next, and label
    combinations are a last resort. A FAN- prefix alone is NOT enough to call
    something a wiring diagram — fan wiring diagrams, fan parameter tables and
    fan group drawings all carry it, and they need different field sets.
    """
    texts = [b.text for b in ocr.blocks] or ([ocr.text] if ocr.text else [])
    blob = " ".join(texts)

    # 1. title keywords, in configured order (group types before wiring, since
    #    a group drawing's title legitimately contains both words)
    for name, spec in drawing_types.items():
        if any(keyword in blob for keyword in spec.title_keywords):
            return name, "title"

    # 2. explicit drawing-number prefix
    for name, spec in drawing_types.items():
        for prefix in spec.drawing_no_prefixes:
            if re.search(re.escape(prefix), blob, re.I):
                return name, "drawing_no"

    # 3. field combination
    repeated_metric_labels = [
        t for t in texts if any(t.endswith(suffix) for suffix in _GROUP_HINT_SUFFIXES)
    ]
    if len(repeated_metric_labels) >= 2:
        for name, spec in drawing_types.items():
            if spec.per_device:
                return name, "field_combination"
    if sum(1 for t in texts if t in _WIRING_HINT_LABELS) >= 2:
        for name, spec in drawing_types.items():
            if not spec.per_device and spec.required_labels:
                return name, "field_combination"

    return "unknown", "none"


def _evaluate_flat_type(
    ocr: OCRResult, spec: DrawingTypeConfig, cfg: FieldCompletenessConfig,
    pairs: Optional[List[FieldPair]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Single-device drawing: returns (found, missing, isolated_labels).

    A field counts as found only when BOTH its label and a value in the value
    slot were read. Label-only is recorded separately as an isolated label —
    that distinction is the whole point, because label-only is precisely what
    occlusion produces and what a confidence check cannot see.
    """
    by_text = {b.text.strip(): b for b in ocr.blocks}
    found: List[str] = []
    missing: List[str] = []
    isolated: List[str] = []

    def has_label_and_value(label_name: str) -> bool:
        block = by_text.get(label_name)
        if block is None:
            return False
        value = find_value_block(block, ocr.blocks, cfg)
        if value is None:
            isolated.append(label_name)
            return False
        if pairs is not None:
            # Recorded here rather than re-derived later so value validation
            # and completeness can never disagree about which block holds a
            # given field's value.
            pairs.append(FieldPair(
                entity_id=None, field_name=label_name, raw_value=value.text,
                bbox=list(value.bbox), confidence=value.confidence,
            ))
        return True

    for label_name in spec.required_labels:
        if has_label_and_value(label_name):
            found.append(label_name)
        else:
            missing.append(label_name)

    for group in spec.one_of_groups:
        satisfied = [name for name in group if has_label_and_value(name)]
        if satisfied:
            found.append(satisfied[0])
        else:
            missing.append("|".join(group))

    return found, missing, isolated


def device_count_from_table(
    table: Optional[TableStructure], identity_suffix: str
) -> Optional[int]:
    """Devices visible in the recovered table rows. DIAGNOSTIC ONLY — this is
    NOT an independent second source, and must never be treated as one.

    The text inside those rows was placed there by the same OCR pass whose
    output we are trying to check, so agreement proves nothing: if OCR missed
    a device row entirely, the row is missing from both the label scan and the
    table, and the two agree on a wrong number. Disagreement is still
    informative (something was recovered inconsistently), which is the only
    reason this survives.
    """
    if table is None or not table.rows:
        return None
    count = sum(1 for row in table.rows if identity_suffix in " ".join(row))
    return count or None


def _evaluate_per_device_type(
    ocr: OCRResult, spec: DrawingTypeConfig, cfg: FieldCompletenessConfig,
    table: Optional[TableStructure], min_table_confidence: float,
    pairs: Optional[List[FieldPair]] = None,
) -> Tuple[List[str], List[str], List[str], Optional[int], bool, List[str]]:
    """Multi-device drawing. Labels are per-device prefixed, so the check runs
    over required suffixes for each discovered device.

    The trap this has to avoid: if OCR reads one device perfectly and misses
    another device's row entirely, every device it CAN see looks complete and
    the page passes. Completeness over discovered devices is therefore not
    sufficient — the device count itself has to be corroborated, and when it
    cannot be, the page is marked cardinality-uncertain rather than quietly
    passed.
    """
    devices: Dict[str, List[str]] = {}
    for block in ocr.blocks:
        text = block.text.strip()
        for suffix in spec.required_label_suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                devices.setdefault(text[: -len(suffix)], []).append(suffix)

    found: List[str] = []
    missing: List[str] = []
    isolated: List[str] = []
    by_text = {b.text.strip(): b for b in ocr.blocks}

    for device, suffixes in sorted(devices.items()):
        for suffix in spec.required_label_suffixes:
            label_name = f"{device}{suffix}"
            if suffix not in suffixes:
                missing.append(label_name)
                continue
            block = by_text.get(label_name)
            value = find_value_block(block, ocr.blocks, cfg) if block is not None else None
            if block is not None and value is None:
                isolated.append(label_name)
                missing.append(label_name)
            else:
                found.append(label_name)
                if pairs is not None and value is not None:
                    # entity_id is what keeps two devices' different readings
                    # of the same field from being reported as a contradiction.
                    pairs.append(FieldPair(
                        entity_id=device, field_name=suffix, raw_value=value.text,
                        bbox=list(value.bbox), confidence=value.confidence,
                    ))

    reasons: List[str] = []
    uncertain = False
    device_count: Optional[int] = len(devices) or None

    if not devices:
        uncertain = True
        reasons.append("no_device_labels_recovered")
    else:
        # Corroborate the count against recovered table geometry. Without a
        # second source there is no way to tell "3 devices, all read" from
        # "4 devices, one row missed".
        # KNOWN LIMITATION, stated rather than papered over: this pipeline has
        # no source of device count that is independent of the OCR text. The
        # table rows are filled from the same OCR blocks, so a device row OCR
        # never saw is absent from both counts and they agree on a wrong
        # number. Until an OCR-independent count exists (grid-row geometry
        # scoped to the parameter-table region, or the equipment ledger), a
        # multi-device page is always cardinality-uncertain.
        #
        # The cost is explicit: every group drawing escalates to VLM, including
        # ones OCR read perfectly. That is the deliberate direction — an
        # unnoticed missing device is a wrong answer handed to someone
        # dispatching a fault, an extra VLM call is a few cents.
        uncertain = True
        reasons.append("device_count_not_independently_verifiable")
        table_count = device_count_from_table(table, spec.identity_label_suffix)
        if table_count is not None and table_count != len(devices):
            # Agreement proves nothing, but disagreement is real evidence that
            # recovery was inconsistent.
            reasons.append(
                f"device_count_disagreement_table={table_count}_labels={len(devices)}")
        if len({len(v) for v in devices.values()}) > 1:
            uncertain = True
            reasons.append("uneven_field_counts_across_devices")

    return found, missing, isolated, device_count, uncertain, reasons


def evaluate_field_completeness(
    ocr: OCRResult,
    drawing_types: Dict[str, DrawingTypeConfig],
    cfg: FieldCompletenessConfig,
    table: Optional[TableStructure] = None,
    task_required_fields: Optional[Sequence[str]] = None,
    min_table_confidence: float = 0.5,
) -> FieldCompleteness:
    """Compute all silent-miss signals for one page.

    task_required_fields, when supplied, REPLACES the drawing type's default
    field list: what the user actually asked for outranks what the drawing
    normally carries (ask "which page is this?" and a revision number will not
    do). It is left unset during evaluation runs — there is no user question
    there, and feeding expected answers in would make routing circular.
    """
    drawing_type, type_source = classify_drawing_type(ocr, drawing_types)
    repetition = watermark_repetition(ocr.text, cfg.watermark_ngram, cfg.watermark_min_repeats)

    if drawing_type == "unknown":
        return FieldCompleteness(
            drawing_type=drawing_type, type_source=type_source,
            completeness=0.0, watermark_repetition=repetition,
            reasons=["drawing_type_unknown"],
        )

    spec = drawing_types[drawing_type]
    reasons: List[str] = []
    device_count: Optional[int] = None
    uncertain = False
    pairs: List[FieldPair] = []

    if task_required_fields:
        effective = DrawingTypeConfig(required_labels=list(task_required_fields))
        found, missing, isolated = _evaluate_flat_type(ocr, effective, cfg, pairs)
        reasons.append("using_task_required_fields")
    elif spec.per_device:
        found, missing, isolated, device_count, uncertain, group_reasons = \
            _evaluate_per_device_type(ocr, spec, cfg, table, min_table_confidence, pairs)
        reasons.extend(group_reasons)
    else:
        found, missing, isolated = _evaluate_flat_type(ocr, spec, cfg, pairs)

    total = len(found) + len(missing)
    completeness = len(found) / total if total else 0.0

    if uncertain:
        reasons.append("group_cardinality_uncertain")
    if completeness < cfg.min_completeness:
        reasons.append("field_completeness_below_threshold")
    if len(isolated) > cfg.max_isolated_labels:
        reasons.append("isolated_labels_without_values")
    if repetition > cfg.max_watermark_repetition:
        reasons.append("repeated_boilerplate_text")

    return FieldCompleteness(
        drawing_type=drawing_type,
        type_source=type_source,
        found_fields=found,
        missing_fields=missing,
        isolated_labels=isolated,
        completeness=completeness,
        watermark_repetition=repetition,
        group_device_count=device_count,
        group_cardinality_uncertain=uncertain,
        reasons=reasons,
        field_pairs=pairs,
    )
