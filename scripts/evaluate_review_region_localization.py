"""Localisation metrics — only on human-reviewed records.

With no reviewed record the report is `not_evaluated`. Not 0%, not 100%, not
"pending": a number here would be read as a measurement, and there is nothing
to measure against. The system's own output is not an answer key.

Three recalls are kept apart because they answer different questions:

  candidate_oracle_recall  did ANY candidate hit the gold? — the ceiling a
                           better ranker could reach
  top1_localization_recall did the chosen candidate hit it? — what the
                           pipeline actually delivers today
  precise_localization_recall  same, counting only `precise` candidates —
                           the only grade allowed to feed an automatic answer

A full-page diagnostic trivially contains every gold box. It is excluded from
every localisation success count; letting it score would make "crop the whole
page" the best strategy.

Phase 1 (evaluator_version 2) ADDS, and changes nothing above:

  gold_coverage         = |pred ∩ gold| / |gold|   — was the target inside?
  prediction_precision  = |pred ∩ gold| / |pred|   — how much of the crop was it?

IoU says whether a box is TIGHT; coverage says whether the target was
INCLUDED. Neither replaces the other. Coverage alone is maximised by cropping
the whole page (coverage 1.0 on everything), which is exactly why it is always
reported beside IoU and prediction_precision, and why a contextual region with
coverage 1.0 still never counts as a precise localisation.

For fields a human judged unlocatable, one old number — whether the system
"declined" — is split into two that mean different things:

  unlocatable_auto_answer_false_positive_rate   SAFETY: the selected region
      was answer_eligible, so the region gate would have let it feed an answer
  unlocatable_review_trigger_rate               COST: any review region was
      produced, so a review would be spent on something no one can read

Phase 2 (evaluator_version 3) adds field ATTRIBUTION, from gold schema v2:

  confirmed   enters the formal IoU / coverage evaluation — the only records
              that do
  ambiguous   a readable value that may belong to several fields; outside
              the localisation denominator, and an auto-answer on it is a
              safety failure
  unassigned  a readable value that cannot be tied to this field; outside
              the denominator, and the system must block auto-answer
  unknown     every v1 record, and any unreadable value; counted and shown,
              never mixed into a formal accuracy denominator

The legacy and phase-1 numbers are still produced and still byte-for-byte what
they were. They are computed on gold_region_type and do not verify that the
located value belongs to the target field; phase 2 says so next to them.

    python scripts/evaluate_review_region_localization.py            # legacy default
    python scripts/evaluate_review_region_localization.py \\
        --gold data/review_region_gold_test.jsonl \\
        --output-json outputs/<new>.json --output-md outputs/<new>.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.region_gold import (  # noqa: E402
    AMBIGUOUS, ASSOCIATION_STATUSES, CONFIRMED, EXCLUDED, FULL_PAGE,
    HUMAN_REVIEWED, SCHEMA_V1, UNASSIGNED, UNKNOWN, UNLOCATABLE, UNREVIEWED,
    image_hash_matches, read_jsonl, status_summary, validate_record,
)
from src.vision.review_region_resolver import (  # noqa: E402
    CONTEXTUAL, DIAGNOSTIC, FULL_PAGE_DIAGNOSTIC, PRECISE, STRUCTURAL, iou,
)

GOLD = ROOT / "data" / "review_region_gold_unreviewed.jsonl"
PACK_MANIFEST = ROOT / "outputs" / "review_region_annotation_pack" / "manifest.json"
OUT_JSON = ROOT / "outputs" / "review_region_localization_eval.json"
OUT_MD = ROOT / "outputs" / "review_region_localization_eval.md"

THRESHOLDS = (0.3, 0.5, 0.75)
# Strategies that never count as a localisation success, however much they
# overlap: a box that contains everything hits everything.
NON_LOCALISING = {FULL_PAGE_DIAGNOSTIC}

EVALUATOR_VERSION = 2

# ---- phase 1 evaluation definitions ----------------------------------------
# These define what a VERDICT means. They are properties of the evaluation, not
# of the resolver, and changing them changes the meaning of every past report
# that used them — hence named constants recorded in each output.
COVERAGE_THRESHOLDS = (0.5, 0.8, 0.95)
PRECISE_MATCH_IOU = 0.75       # tight enough to call the box itself right
COVERED = 0.8                  # most of the target is inside the crop

PRECISE_MATCH = "precise_match"
COVERED_BUT_LOOSE = "covered_but_loose"
CONTEXT_ONLY = "context_only"
MISSED = "missed"
CORRECTLY_ABSTAINED = "correctly_abstained"
UNNECESSARY_REVIEW = "unnecessary_review"
UNSAFE_AUTO_ANSWER = "unsafe_auto_answer"
VERDICTS = (PRECISE_MATCH, COVERED_BUT_LOOSE, CONTEXT_ONLY, MISSED,
            CORRECTLY_ABSTAINED, UNNECESSARY_REVIEW, UNSAFE_AUTO_ANSWER)

TRUST_BUCKETS = (PRECISE, STRUCTURAL, CONTEXTUAL, DIAGNOSTIC, "unavailable")

# Reports that exist on disk and must never be written by an explicit
# --output-* path. The legacy default run still writes its own two files, as it
# always has; nothing else may.
PROTECTED_OUTPUTS = {
    (ROOT / "outputs" / "review_region_localization_eval.json").resolve(),
    (ROOT / "outputs" / "review_region_localization_eval.md").resolve(),
    (ROOT / "outputs" / "review_region_localization_eval_gold_test.json").resolve(),
    (ROOT / "outputs" / "review_region_localization_eval_gold_test.md").resolve(),
    (ROOT / "outputs" / "review_region_localization_eval_phase1_gold_test.json").resolve(),
    (ROOT / "outputs" / "review_region_localization_eval_phase1_gold_test.md").resolve(),
}


class EvaluationInputError(ValueError):
    """The run cannot proceed without guessing, so it does not."""


def load_predictions(path: Optional[Path] = None) -> Dict[str, List[dict]]:
    manifest_path = path or PACK_MANIFEST
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: Dict[str, List[dict]] = defaultdict(list)
    for page in manifest.get("page_details", []):
        for order, candidate in enumerate(page.get("candidates", [])):
            for field_name in candidate.get("target_fields", []):
                out[f"{page['document_id']}:p1:{field_name}"].append({
                    **candidate, "order": order,
                    "drawing_type": page.get("drawing_type")})
    return out


def not_evaluated(reason: str, summary: dict, extra: Optional[dict] = None) -> dict:
    return {
        "status": "not_evaluated",
        "reason": reason,
        "gold_status_summary": summary,
        "human_reviewed_count": summary.get(HUMAN_REVIEWED, 0),
        "metrics": None,
        "note": ("No metric is emitted. A 0% or 100% here would be read as a "
                 "measurement; there is nothing to measure against. The "
                 "system's own output is not an answer key."),
        **(extra or {}),
    }


def evaluate(records, predictions) -> dict:
    """Score the reviewed records. Never called when there are none.

    LEGACY (evaluator_version 1). Unchanged in phase 1; every number it
    produces is asserted identical to the pre-phase-1 report by a test.
    """
    rows = []
    for record in records:
        candidates = predictions.get(record.record_id, [])
        scoring = [c for c in candidates
                   if c["strategy"] not in NON_LOCALISING and c.get("bbox")]

        if record.gold_region_type == UNLOCATABLE:
            # The system is right here by declining, not by framing something.
            declined = (not scoring) or all(
                c["trust_level"] == DIAGNOSTIC for c in candidates)
            rows.append({"record_id": record.record_id, "kind": "unlocatable",
                         "system_declined": declined,
                         "candidate_count": len(candidates)})
            continue

        ious = [(iou(record.gold_bbox, c["bbox"]), c) for c in scoring]
        best = max(ious, key=lambda pair: pair[0]) if ious else (0.0, None)
        top1 = next((c for c in sorted(candidates, key=lambda c: c["order"])
                     if c["strategy"] not in NON_LOCALISING and c.get("bbox")), None)
        precise = [(v, c) for v, c in ious if c["trust_level"] == PRECISE]

        rows.append({
            "record_id": record.record_id,
            "kind": "locatable",
            "document_id": record.document_id,
            "target_field": record.target_field,
            "gold_region_type": record.gold_region_type,
            "drawing_type": (candidates[0].get("drawing_type")
                             if candidates else None),
            "candidate_count": len(candidates),
            "scoring_candidate_count": len(scoring),
            "best_iou": round(best[0], 4),
            "best_strategy": best[1]["strategy"] if best[1] else None,
            "best_trust": best[1]["trust_level"] if best[1] else None,
            "top1_iou": round(iou(record.gold_bbox, top1["bbox"]), 4) if top1 else 0.0,
            "top1_strategy": top1["strategy"] if top1 else None,
            "top1_trust": top1["trust_level"] if top1 else None,
            "precise_best_iou": round(max((v for v, _ in precise), default=0.0), 4),
            "area_ratio_top1": top1.get("area_ratio") if top1 else None,
        })

    locatable = [r for r in rows if r["kind"] == "locatable"]
    unlocatable = [r for r in rows if r["kind"] == "unlocatable"]
    denominator = len(locatable) or 1

    def recall(key: str, threshold: float) -> float:
        return round(sum(1 for r in locatable if r[key] >= threshold)
                     / denominator, 4)

    def block(subset: List[dict]) -> dict:
        n = len(subset) or 1
        return {
            "n": len(subset),
            "candidate_oracle_recall": {
                f"iou@{t}": round(sum(1 for r in subset if r["best_iou"] >= t) / n, 4)
                for t in THRESHOLDS},
            "top1_localization_recall": {
                f"iou@{t}": round(sum(1 for r in subset if r["top1_iou"] >= t) / n, 4)
                for t in THRESHOLDS},
            "mean_best_iou": round(
                statistics.mean([r["best_iou"] for r in subset]), 4) if subset else None,
        }

    areas = [r["area_ratio_top1"] for r in locatable if r["area_ratio_top1"]]
    return {
        "status": "evaluated",
        "human_reviewed_count": len(rows),
        "locatable_records": len(locatable),
        "unlocatable_records": len(unlocatable),
        "small_sample_warning": (
            f"{len(locatable)} locatable reviewed record(s). Every rate below "
            f"moves by {1 / denominator:.1%} per record; these are counts, not "
            f"performance estimates, and generalisation must not be inferred."),
        "metrics": {
            "all": block(locatable),
            "precise": {
                "n": sum(1 for r in locatable if r["top1_trust"] == PRECISE),
                "precise_localization_recall": {
                    f"iou@{t}": recall("precise_best_iou", t) for t in THRESHOLDS},
            },
            "structural": block([r for r in locatable
                                 if r["top1_trust"] == STRUCTURAL]),
            "contextual": block([r for r in locatable
                                 if r["top1_trust"] == CONTEXTUAL]),
            "diagnostic": {
                "n": sum(1 for r in locatable if r["top1_trust"] == DIAGNOSTIC),
                "note": ("full-page diagnostics are excluded from every "
                         "localisation success count"),
            },
            "by_strategy": {
                strategy: block([r for r in locatable
                                 if r["top1_strategy"] == strategy])
                for strategy in sorted({r["top1_strategy"] for r in locatable
                                        if r["top1_strategy"]})},
            "by_drawing_type": {
                kind: block([r for r in locatable if r["drawing_type"] == kind])
                for kind in sorted({r["drawing_type"] for r in locatable
                                    if r["drawing_type"]})},
            "by_gold_region_type": {
                kind: block([r for r in locatable if r["gold_region_type"] == kind])
                for kind in sorted({r["gold_region_type"] for r in locatable
                                    if r["gold_region_type"]})},
            "unlocatable_detection_accuracy": (
                round(sum(1 for r in unlocatable if r["system_declined"])
                      / len(unlocatable), 4) if unlocatable else None),
            "average_candidate_count": round(
                statistics.mean([r["candidate_count"] for r in rows]), 3) if rows else None,
            "area_ratio_p50": round(statistics.median(areas), 4) if areas else None,
            "area_ratio_p95": round(sorted(areas)[int(len(areas) * 0.95) - 1], 4)
            if len(areas) > 1 else (areas[0] if areas else None),
        },
        "records": rows,
    }


# ---------------------------------------------------------------------------
# phase 1 — coverage, verdicts, and the safety / cost split
# ---------------------------------------------------------------------------

def _area(box: Sequence[float]) -> float:
    if not box or len(box) != 4:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def gold_coverage(prediction: Sequence[float], gold: Sequence[float]) -> float:
    """|pred ∩ gold| / |gold|. 1.0 means the target was fully inside the crop.

    Says nothing about how much else was in the crop — see
    prediction_precision, which is why the two are always reported together.
    """
    area = _area(gold)
    return _intersection(prediction, gold) / area if area else 0.0


def prediction_precision(prediction: Sequence[float], gold: Sequence[float]) -> float:
    """|pred ∩ gold| / |pred|. Low means most of the crop was not the target."""
    area = _area(prediction)
    return _intersection(prediction, gold) / area if area else 0.0


def _selected(candidates: Sequence[dict], *, include_diagnostic: bool) -> Optional[dict]:
    """First candidate in the resolver's own order that has a box."""
    for candidate in sorted(candidates, key=lambda c: c["order"]):
        if not candidate.get("bbox"):
            continue
        if not include_diagnostic and candidate["strategy"] in NON_LOCALISING:
            continue
        return candidate
    return None


def classify_locatable(top1_trust: Optional[str], top1_iou: float,
                       top1_coverage: float) -> tuple:
    """Verdict for a field a human could locate. Returns (verdict, reason).

    precise_match is reserved for the precise grade. A structural or contextual
    region that happens to fit tightly is still not a grade the system allows
    to feed an automatic answer, and calling it a precise match would erase the
    distinction the grading exists to draw.
    """
    if top1_trust is None:
        return MISSED, "no candidate region"
    if top1_trust == PRECISE and top1_iou >= PRECISE_MATCH_IOU:
        return PRECISE_MATCH, f"precise grade, IoU >= {PRECISE_MATCH_IOU}"
    if top1_coverage >= COVERED:
        if top1_trust in (PRECISE, STRUCTURAL):
            why = ("precise grade, target inside but IoU below "
                   f"{PRECISE_MATCH_IOU}" if top1_trust == PRECISE else
                   "structural grade covers the target; grade is not precise")
            return COVERED_BUT_LOOSE, why
        return CONTEXT_ONLY, (f"{top1_trust} region contains the target; "
                              "not a field-level localisation")
    return MISSED, f"coverage {top1_coverage:.2f} < {COVERED}"


def classify_unlocatable(selected: Optional[dict], any_region: bool) -> tuple:
    """Verdict for a field a human could NOT reliably locate."""
    if selected is not None and selected.get("answer_eligible"):
        return UNSAFE_AUTO_ANSWER, ("selected region is answer_eligible on a "
                                    "field no human could locate")
    if any_region:
        return UNNECESSARY_REVIEW, ("a review region was produced for a field "
                                    "no human could locate; costs a review, "
                                    "cannot feed an answer")
    return CORRECTLY_ABSTAINED, "no region produced"


def _trust_bucket(candidate: Optional[dict]) -> str:
    if candidate is None or candidate.get("strategy") == "unavailable":
        return "unavailable"
    return candidate.get("trust_level", "unavailable")


def evaluate_phase1(records, predictions) -> dict:
    """Coverage, per-record verdicts and the safety/cost split.

    Uses the SAME top1 definition as the legacy evaluate() — first candidate in
    resolver order, full-page diagnostics excluded — so the two sets of numbers
    describe the same boxes.
    """
    rows = []
    for record in records:
        candidates = predictions.get(record.record_id, [])
        top1 = _selected(candidates, include_diagnostic=False)
        selected_any = _selected(candidates, include_diagnostic=True)
        scoring = [c for c in candidates
                   if c["strategy"] not in NON_LOCALISING and c.get("bbox")]

        base = {
            "record_id": record.record_id,
            "document_id": record.document_id,
            "target_field": record.target_field,
            "gold_region_type": record.gold_region_type,
            "gold_bbox": record.gold_bbox,
            "candidate_count": len(candidates),
        }

        if record.gold_region_type == UNLOCATABLE:
            verdict, reason = classify_unlocatable(
                selected_any, any(c.get("bbox") for c in candidates))
            rows.append({
                **base, "locatable": False,
                "selected_bbox": selected_any["bbox"] if selected_any else None,
                "selected_strategy": selected_any["strategy"] if selected_any else None,
                "selected_trust": _trust_bucket(selected_any),
                "answer_eligible": bool(selected_any and selected_any.get("answer_eligible")),
                "any_candidate_answer_eligible": any(
                    c.get("answer_eligible") for c in candidates),
                "verdict": verdict, "verdict_reason": reason,
            })
            continue

        gold = record.gold_bbox
        best_iou_pair = max(((iou(gold, c["bbox"]), c) for c in scoring),
                            key=lambda p: p[0], default=(0.0, None))
        best_cov_pair = max(((gold_coverage(c["bbox"], gold), c) for c in scoring),
                            key=lambda p: p[0], default=(0.0, None))
        top1_iou = iou(gold, top1["bbox"]) if top1 else 0.0
        top1_cov = gold_coverage(top1["bbox"], gold) if top1 else 0.0
        top1_prec = prediction_precision(top1["bbox"], gold) if top1 else 0.0

        # A page whose only candidate is the full-page diagnostic: it covers the
        # target by construction and is reported, never scored as local success.
        if top1 is None and selected_any is not None:
            verdict, reason = CONTEXT_ONLY, "only a full-page diagnostic exists"
            trust = DIAGNOSTIC
        else:
            trust = top1["trust_level"] if top1 else None
            verdict, reason = classify_locatable(trust, top1_iou, top1_cov)

        rows.append({
            **base, "locatable": True,
            "top1_bbox": top1["bbox"] if top1 else None,
            "top1_strategy": top1["strategy"] if top1 else None,
            "top1_trust": trust if trust else "unavailable",
            "answer_eligible": bool(top1 and top1.get("answer_eligible")),
            "top1_iou": round(top1_iou, 4),
            "top1_gold_coverage": round(top1_cov, 4),
            "top1_prediction_precision": round(top1_prec, 4),
            "best_iou_bbox": best_iou_pair[1]["bbox"] if best_iou_pair[1] else None,
            "best_iou": round(best_iou_pair[0], 4),
            "best_coverage_bbox": best_cov_pair[1]["bbox"] if best_cov_pair[1] else None,
            "best_gold_coverage": round(best_cov_pair[0], 4),
            "verdict": verdict, "verdict_reason": reason,
        })

    locatable = [r for r in rows if r["locatable"]]
    unlocatable = [r for r in rows if not r["locatable"]]

    def rate(subset, key, threshold):
        return {"count": sum(1 for r in subset if r[key] >= threshold),
                "n": len(subset),
                "rate": round(sum(1 for r in subset if r[key] >= threshold)
                              / len(subset), 4) if subset else None}

    def coverage_block(subset):
        return {
            "n": len(subset),
            "top1_gold_coverage_recall": {
                f"coverage@{t}": rate(subset, "top1_gold_coverage", t)
                for t in COVERAGE_THRESHOLDS},
            "top1_iou_recall": {
                f"iou@{t}": rate(subset, "top1_iou", t) for t in THRESHOLDS},
            "mean_top1_prediction_precision": round(statistics.mean(
                [r["top1_prediction_precision"] for r in subset]), 4) if subset else None,
        }

    n_unloc = len(unlocatable)
    unsafe = [r for r in unlocatable if r["verdict"] == UNSAFE_AUTO_ANSWER]
    triggered = [r for r in unlocatable if r["selected_trust"] != "unavailable"]

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "definitions": {
            "gold_coverage": "|pred ∩ gold| / |gold| — was the target inside the crop",
            "prediction_precision": "|pred ∩ gold| / |pred| — how much of the crop was the target",
            "iou": "|pred ∩ gold| / |pred ∪ gold| — how tight the crop is",
            "reading": ("IoU measures tightness; coverage measures inclusion. "
                        "Coverage alone is maximised by cropping the whole page, "
                        "so it is never reported without IoU and precision."),
            "precise_match_iou": PRECISE_MATCH_IOU,
            "covered_threshold": COVERED,
            "coverage_thresholds": list(COVERAGE_THRESHOLDS),
            "top1": ("first candidate in resolver order with a bbox, full-page "
                     "diagnostics excluded — identical to the legacy top1"),
            "contextual_never_precise": True,
            "diagnostic_never_local_success": True,
        },
        "coverage": {
            "all": coverage_block(locatable),
            "candidate_oracle_gold_coverage_recall": {
                f"coverage@{t}": rate(locatable, "best_gold_coverage", t)
                for t in COVERAGE_THRESHOLDS},
            "by_trust_level": {
                bucket: coverage_block([r for r in locatable
                                        if r["top1_trust"] == bucket])
                for bucket in TRUST_BUCKETS},
        },
        "unlocatable": {
            "n": n_unloc,
            "unlocatable_auto_answer_false_positive_rate": {
                "count": len(unsafe), "n": n_unloc,
                "rate": round(len(unsafe) / n_unloc, 4) if n_unloc else None,
                "kind": "SAFETY",
                "meaning": ("a human could not locate the field, yet the region "
                            "the system selected was answer_eligible"),
            },
            "unlocatable_review_trigger_rate": {
                "count": len(triggered), "n": n_unloc,
                "rate": round(len(triggered) / n_unloc, 4) if n_unloc else None,
                "kind": "COST",
                "meaning": ("a human could not locate the field, yet a review "
                            "region was produced for it"),
            },
            "selected_trust_distribution": {
                bucket: sum(1 for r in unlocatable if r["selected_trust"] == bucket)
                for bucket in TRUST_BUCKETS},
            "scope_note": ("answer_eligible measures the REGION GATE only. It is "
                           "not proof that an answer was emitted: value "
                           "validation and the evidence policy still run after it."),
        },
        "verdict_counts": {v: sum(1 for r in rows if r["verdict"] == v)
                           for v in VERDICTS},
        "records": rows,
    }


PHASE2_EVALUATOR_VERSION = 3

AUTO_ANSWER_ELIGIBLE = "auto_answer_eligible"
REVIEW_ONLY = "review_only"
NO_REGION = "no_region"
SYSTEM_OUTCOMES = (AUTO_ANSWER_ELIGIBLE, REVIEW_ONLY, NO_REGION)


def _system_outcome(selected: Optional[dict]) -> str:
    if selected is None:
        return NO_REGION
    return AUTO_ANSWER_ELIGIBLE if selected.get("answer_eligible") else REVIEW_ONLY


def evaluate_phase2(records, predictions) -> dict:
    """Attribution-aware evaluation. Formal localisation only on `confirmed`.

    Every v1 record is `unknown` here — including the ones a person clearly
    located — because v1 never asked whether the located value belongs to the
    target field. Promoting them would be inferring the answer the new schema
    exists to record.
    """
    by_status: Dict[str, list] = {status: [] for status in ASSOCIATION_STATUSES}
    for record in records:
        by_status[record.effective_association].append(record)

    confirmed = [r for r in by_status[CONFIRMED] if r.gold_bbox]
    if confirmed:
        formal = {"status": "evaluated",
                  "n": len(confirmed),
                  "legacy_on_confirmed": evaluate(confirmed, predictions)["metrics"],
                  "phase1_on_confirmed": evaluate_phase1(confirmed, predictions)}
    else:
        formal = {"status": "not_evaluated",
                  "reason": "no_confirmed_attribution_gold",
                  "n": 0,
                  "note": ("No record has association_status=confirmed. The "
                           "legacy and phase-1 numbers in this report are "
                           "computed on region type only and do not establish "
                           "that the located value belongs to the target field.")}

    def open_attribution(status: str) -> dict:
        rows = []
        for record in by_status[status]:
            candidates = predictions.get(record.record_id, [])
            selected = _selected(candidates, include_diagnostic=True)
            outcome = _system_outcome(selected)
            rows.append({
                "record_id": record.record_id,
                "candidate_fields": record.candidate_fields,
                "candidate_bbox": record.candidate_bbox,
                "system_outcome": outcome,
                "selected_strategy": selected["strategy"] if selected else None,
                "selected_trust": _trust_bucket(selected),
                # Did the review region at least include the visible candidate?
                # Diagnostic only: it does not make the attribution right.
                "candidate_coverage": (
                    round(gold_coverage(selected["bbox"], record.candidate_bbox), 4)
                    if selected and record.candidate_bbox else None),
            })
        unsafe = sum(1 for r in rows if r["system_outcome"] == AUTO_ANSWER_ELIGIBLE)
        return {
            "n": len(rows),
            "auto_answer_false_positive": {
                "count": unsafe, "n": len(rows),
                "rate": round(unsafe / len(rows), 4) if rows else None,
                "kind": "SAFETY"},
            "blocked_auto_answer": {
                "count": len(rows) - unsafe, "n": len(rows)},
            "records": rows,
        }

    unknown = by_status[UNKNOWN]
    distribution = {status: {outcome: 0 for outcome in SYSTEM_OUTCOMES}
                    for status in ASSOCIATION_STATUSES}
    for record in records:
        selected = _selected(predictions.get(record.record_id, []),
                             include_diagnostic=True)
        distribution[record.effective_association][_system_outcome(selected)] += 1

    return {
        "evaluator_version": PHASE2_EVALUATOR_VERSION,
        "counts": {status: len(by_status[status]) for status in ASSOCIATION_STATUSES},
        "schema_versions": dict(Counter(r.schema_version for r in records)),
        "formal_localization": formal,
        "ambiguous": {
            **open_attribution(AMBIGUOUS),
            "ambiguous_auto_answer_false_positive_rate":
                open_attribution(AMBIGUOUS)["auto_answer_false_positive"]},
        "unassigned": {
            **open_attribution(UNASSIGNED),
            "unassigned_auto_answer_false_positive_rate":
                open_attribution(UNASSIGNED)["auto_answer_false_positive"]},
        "unknown": {
            "n": len(unknown),
            "from_schema_v1": sum(1 for r in unknown if r.schema_version == SCHEMA_V1),
            "by_gold_region_type": dict(Counter(r.gold_region_type for r in unknown)),
            "note": ("counted and shown only; never in a formal accuracy "
                     "denominator. v1 records land here because v1 did not "
                     "record attribution, not because their annotation is wrong."),
        },
        "attribution_aware_decision_distribution": distribution,
        "supersession": {
            "still_valid": [
                "legacy IoU metrics and phase-1 coverage as statements about "
                "BOXES vs gold_region_type boxes",
                "phase-1 unlocatable safety/cost split (region gate behaviour)",
            ],
            "superseded": [
                "reading legacy/phase-1 localisation recall as FIELD localisation: "
                "it is only that once records are association_status=confirmed",
            ],
        },
    }


LEGACY_METRIC_NOTES = {
    "metrics.unlocatable_detection_accuracy": (
        "LEGACY, kept for compatibility. It scores 'the system declined' and "
        "counts ANY non-diagnostic region as a failure, so it merges a safety "
        "failure (an answer-eligible region) with a cost (a review region that "
        "cannot feed an answer). A value of 0/5 reads like five safety errors "
        "when, in this repo, it has been five unnecessary reviews. Use "
        "phase1.unlocatable.unlocatable_auto_answer_false_positive_rate (safety) "
        "and phase1.unlocatable.unlocatable_review_trigger_rate (cost) instead."),
}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _check_paths(gold: Path, output_json: Path, output_md: Path, *,
                 explicit_outputs: bool, overwrite: bool) -> None:
    if not gold.exists():
        # No fallback to the default gold: scoring the wrong file silently is
        # worse than not scoring.
        raise EvaluationInputError(f"gold file not found: {gold}")
    out_json, out_md = output_json.resolve(), output_md.resolve()
    if out_json == out_md:
        raise EvaluationInputError("--output-json and --output-md are the same path")
    if gold.resolve() in (out_json, out_md):
        raise EvaluationInputError("an output path is the gold input file")
    if not explicit_outputs:
        return
    for path in (out_json, out_md):
        if path in PROTECTED_OUTPUTS:
            raise EvaluationInputError(
                f"refusing to write a historical report: {path}")
        if path.exists() and not overwrite:
            raise EvaluationInputError(
                f"output already exists: {path} (choose a new path, or pass "
                f"--overwrite for a non-historical file)")


def build_payload(records, predictions) -> dict:
    """Everything the report contains, independent of where it is written."""
    summary = status_summary(records)
    reviewed = [r for r in records if r.label_status == HUMAN_REVIEWED]
    stale = [r.record_id for r in reviewed if not image_hash_matches(r, ROOT)]

    if stale:
        payload = not_evaluated(
            "gold_image_hash_mismatch", summary,
            {"stale_records": stale,
             "detail": ("these records were annotated on different pixels; "
                        "scoring them against the current files would compare "
                        "a box to a picture it was never drawn on")})
    elif not reviewed:
        payload = not_evaluated("no_human_reviewed_bbox_gold", summary, {
            "next_step": ("annotate data/review_region_gold_unreviewed.jsonl "
                          "using docs/review_region_annotation_guide.md, then "
                          "run scripts/validate_review_region_gold.py"),
            "metrics_unlocked_after_review": [
                "IoU", "IoU@0.3", "IoU@0.5", "IoU@0.75",
                "field_localization_recall", "precise_localization_recall",
                "candidate_oracle_recall", "top1_localization_recall",
                "unlocatable_detection_accuracy", "average_candidate_count",
                "area_ratio_p50", "area_ratio_p95"],
        })
    else:
        payload = evaluate(reviewed, predictions)
        payload["gold_status_summary"] = summary
        payload["phase1"] = evaluate_phase1(reviewed, predictions)
        payload["legacy_metric_notes"] = LEGACY_METRIC_NOTES
        payload["phase2"] = evaluate_phase2(reviewed, predictions)

    payload["real_vlm_calls"] = 0
    payload["excluded_from_denominator"] = {
        UNREVIEWED: summary.get(UNREVIEWED, 0),
        EXCLUDED: summary.get(EXCLUDED, 0),
    }
    return payload


def _fmt(block: dict) -> str:
    return f"{block['count']}/{block['n']}" if block["n"] else "—"


def render_markdown(payload: dict) -> str:
    lines = ["# 复核区域定位评测", ""]
    if payload["status"] == "not_evaluated":
        lines += [
            "## 状态：`not_evaluated`", "",
            f"- 原因：`{payload['reason']}`",
            f"- 人工审核记录数：**{payload['human_reviewed_count']}**",
            f"- Gold 状态：`{payload['gold_status_summary']}`", "",
            f"> {payload['note']}", "",
        ]
        if payload.get("next_step"):
            lines += ["## 下一步", "", payload["next_step"], "",
                      "## 人工审核后才能计算的指标", ""]
            lines += [f"- `{m}`" for m in payload["metrics_unlocked_after_review"]]
        if payload.get("stale_records"):
            lines += ["## 图片哈希不匹配，拒绝评测", ""]
            lines += [f"- `{r}`" for r in payload["stale_records"]]
        return "\n".join(lines)

    m = payload["metrics"]
    lines += [
        f"> {payload['small_sample_warning']}", "",
        f"- 人工审核记录：**{payload['human_reviewed_count']}**"
        f"（可定位 {payload['locatable_records']}，"
        f"不可定位 {payload['unlocatable_records']}）", "",
        "## 总体", "", "| 指标 | IoU@0.3 | IoU@0.5 | IoU@0.75 |", "|---|---|---|---|",
        "| candidate_oracle_recall | " + " | ".join(
            f"{m['all']['candidate_oracle_recall'][f'iou@{t}']:.1%}"
            for t in THRESHOLDS) + " |",
        "| top1_localization_recall | " + " | ".join(
            f"{m['all']['top1_localization_recall'][f'iou@{t}']:.1%}"
            for t in THRESHOLDS) + " |",
        "| precise_localization_recall | " + " | ".join(
            f"{m['precise']['precise_localization_recall'][f'iou@{t}']:.1%}"
            for t in THRESHOLDS) + " |", "",
        f"- unlocatable 判定准确率：{m['unlocatable_detection_accuracy']}",
        f"- 平均候选数：{m['average_candidate_count']}",
        f"- 面积占比 p50/p95：{m['area_ratio_p50']} / {m['area_ratio_p95']}", "",
        "> full-page diagnostic 不计入任何定位成功；"
        "unreviewed 与 excluded 不进入分母。",
    ]

    p = payload.get("phase1")
    if not p:
        return "\n".join(lines)

    cov = p["coverage"]
    unl = p["unlocatable"]
    lines += [
        "", "---", "", f"# Phase 1 补充（evaluator_version {p['evaluator_version']}）", "",
        "> **IoU 衡量框是否紧；Coverage 衡量是否把目标包含进去。** 两者同时报告，"
        "Coverage 不替代 IoU。只看 Coverage 时，整页裁剪能拿满分。", "",
        "## Gold Coverage", "",
        "| 口径 | Coverage@0.5 | Coverage@0.8 | Coverage@0.95 |", "|---|---|---|---|",
        "| top1 | " + " | ".join(_fmt(cov["all"]["top1_gold_coverage_recall"][f"coverage@{t}"])
                                for t in COVERAGE_THRESHOLDS) + " |",
        "| candidate oracle | " + " | ".join(
            _fmt(cov["candidate_oracle_gold_coverage_recall"][f"coverage@{t}"])
            for t in COVERAGE_THRESHOLDS) + " |", "",
        "## 按定位等级拆分（top1）", "",
        "| 等级 | n | Cov@0.8 | Cov@0.95 | IoU@0.5 | IoU@0.75 | 平均 precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for bucket in TRUST_BUCKETS:
        b = cov["by_trust_level"][bucket]
        if not b["n"]:
            lines.append(f"| {bucket} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {bucket} | {b['n']} | "
            f"{_fmt(b['top1_gold_coverage_recall']['coverage@0.8'])} | "
            f"{_fmt(b['top1_gold_coverage_recall']['coverage@0.95'])} | "
            f"{_fmt(b['top1_iou_recall']['iou@0.5'])} | "
            f"{_fmt(b['top1_iou_recall']['iou@0.75'])} | "
            f"{b['mean_top1_prediction_precision']} |")
    fp, trig = (unl["unlocatable_auto_answer_false_positive_rate"],
                unl["unlocatable_review_trigger_rate"])
    lines += [
        "", "> contextual 区域即使 Coverage=1 也**不计入** precise localization；"
        "full-page diagnostic 不计入局部定位成功。", "",
        "## 不可定位字段：安全与成本分开", "",
        "| 指标 | 类型 | 计数 | 含义 |", "|---|---|---|---|",
        f"| unlocatable_auto_answer_false_positive_rate | **安全** | "
        f"**{fp['count']}/{fp['n']}** | {fp['meaning']} |",
        f"| unlocatable_review_trigger_rate | 成本 | "
        f"**{trig['count']}/{trig['n']}** | {trig['meaning']} |", "",
        f"- 选中区域等级分布：`{unl['selected_trust_distribution']}`",
        f"- {unl['scope_note']}", "",
        f"> 旧指标 `unlocatable_detection_accuracy`（{m['unlocatable_detection_accuracy']}）"
        "保留为 **legacy**：它把安全失败和成本浪费混成一个数，0/5 看起来像五次安全错误。", "",
        "## 逐条", "",
        "| record | 可定位 | strategy | trust | IoU | Coverage | Precision | answer_eligible | 结论 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in p["records"]:
        if r["locatable"]:
            lines.append(
                f"| {r['record_id']} | 是 | `{r['top1_strategy']}` | {r['top1_trust']} | "
                f"{r['top1_iou']:.3f} | {r['top1_gold_coverage']:.2f} | "
                f"{r['top1_prediction_precision']:.2f} | "
                f"{'是' if r['answer_eligible'] else '否'} | **{r['verdict']}** |")
        else:
            lines.append(
                f"| {r['record_id']} | 否 | `{r['selected_strategy']}` | "
                f"{r['selected_trust']} | — | — | — | "
                f"{'是' if r['answer_eligible'] else '否'} | **{r['verdict']}** |")
    lines += ["", f"结论计数：`{p['verdict_counts']}`"]

    q = payload.get("phase2")
    if not q:
        return "\n".join(lines)
    formal = q["formal_localization"]
    amb = q["ambiguous"]["ambiguous_auto_answer_false_positive_rate"]
    una = q["unassigned"]["unassigned_auto_answer_false_positive_rate"]
    lines += [
        "", "---", "", f"# Phase 2 字段归属（evaluator_version {q['evaluator_version']}）", "",
        "> 只有 `association_status=confirmed` 的记录进入正式定位评测。"
        "v1 记录一律为 `unknown`：v1 从未记录值是否属于目标字段。", "",
        "| 归属状态 | 数量 |", "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in q["counts"].items()] + [
        "", f"- Schema 版本：`{q['schema_versions']}`",
        f"- 正式定位评测：**`{formal['status']}`**"
        + (f"（`{formal['reason']}`）" if formal["status"] != "evaluated" else
           f"（n={formal['n']}）"),
        f"- ambiguous 自动回答误放行：**{_fmt(amb)}**",
        f"- unassigned 自动回答误放行：**{_fmt(una)}**",
        f"- unknown：{q['unknown']['n']}（来自 v1：{q['unknown']['from_schema_v1']}）", "",
        "## 归属感知的决策分布", "",
        "| 归属状态 | 可自动回答 | 仅复核 | 无区域 |", "|---|---|---|---|",
    ] + [f"| {k} | {v['auto_answer_eligible']} | {v['review_only']} | {v['no_region']} |"
         for k, v in q["attribution_aware_decision_distribution"].items()] + [
        "", "## 口径变更", "",
        "- 仍然有效：" + "；".join(q["supersession"]["still_valid"]),
        "- 被取代：" + "；".join(q["supersession"]["superseded"]),
    ]
    return "\n".join(lines)


def run(*, gold: Path, output_json: Path, output_md: Path,
        explicit_outputs: bool = False, overwrite: bool = False,
        predictions_path: Optional[Path] = None) -> dict:
    """The one implementation. CLI and tests both call this."""
    _check_paths(gold, output_json, output_md,
                 explicit_outputs=explicit_outputs, overwrite=overwrite)
    records = read_jsonl(gold)
    if not records:
        raise EvaluationInputError(f"gold file has no records: {gold}")
    problems = [f"{r.record_id}: {p}" for r in records for p in validate_record(r)]
    if problems:
        raise EvaluationInputError(
            "gold file fails its schema; run scripts/validate_review_region_gold.py:\n  "
            + "\n  ".join(problems))

    payload = build_payload(records, load_predictions(predictions_path))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default=None,
                        help=f"gold JSONL (default: {GOLD.relative_to(ROOT)})")
    parser.add_argument("--output-json", default=None,
                        help=f"default: {OUT_JSON.relative_to(ROOT)}")
    parser.add_argument("--output-md", default=None,
                        help=f"default: {OUT_MD.relative_to(ROOT)}")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing NON-historical output")
    args = parser.parse_args(argv)

    explicit = args.output_json is not None or args.output_md is not None
    if explicit and (args.output_json is None or args.output_md is None):
        parser.error("--output-json and --output-md must be given together")

    gold = Path(args.gold) if args.gold else GOLD
    output_json = Path(args.output_json) if args.output_json else OUT_JSON
    output_md = Path(args.output_md) if args.output_md else OUT_MD
    if not gold.is_absolute():
        gold = ROOT / gold
    if not output_json.is_absolute():
        output_json = ROOT / output_json
    if not output_md.is_absolute():
        output_md = ROOT / output_md

    try:
        payload = run(gold=gold, output_json=output_json, output_md=output_md,
                      explicit_outputs=explicit, overwrite=args.overwrite)
    except EvaluationInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = payload["gold_status_summary"]
    print(f"status={payload['status']}")
    if payload["status"] == "not_evaluated":
        print(f"reason={payload['reason']}  "
              f"human_reviewed={payload['human_reviewed_count']}")
    print(f"gold: {summary}")
    print(f"-> {output_json}\n-> {output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
