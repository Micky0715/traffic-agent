"""Phase 1 localisation evaluator: coverage, verdicts, safety/cost split, CLI.

Geometry and classification tests are SYNTHETIC. Tests marked REAL read
data/review_region_gold_test.jsonl (human-reviewed by jdh07) and the recorded
resolver candidates; they check that the evaluator reports those boxes
faithfully, not that the resolver is good.

No network, no model, no paid API. No historical report is written: every
output goes to tmp_path, and the protected-path tests assert the historical
files' hashes are unchanged afterwards.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluate_review_region_localization as ev
from src.vision.region_gold import HUMAN_REVIEWED, RegionGoldRecord

ROOT = Path(__file__).resolve().parents[1]
GOLD_TEST = ROOT / "data" / "review_region_gold_test.jsonl"
OLD_GOLD_TEST_REPORT = ROOT / "outputs" / "review_region_localization_eval_gold_test.json"

GOLD = [100.0, 100.0, 200.0, 150.0]          # 100 x 50


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# geometry — SYNTHETIC
# --------------------------------------------------------------------------

def test_1_prediction_equal_to_gold():
    assert ev.gold_coverage(GOLD, GOLD) == 1.0
    assert ev.prediction_precision(GOLD, GOLD) == 1.0
    assert ev.iou(GOLD, GOLD) == 1.0


def test_2_prediction_fully_contains_gold():
    """Found and loose: every pixel of the target is in, but so is much else."""
    pred = [50.0, 50.0, 250.0, 200.0]
    assert ev.gold_coverage(pred, GOLD) == 1.0
    assert ev.prediction_precision(pred, GOLD) < 1.0
    assert ev.iou(GOLD, pred) < 1.0


def test_3_gold_fully_contains_prediction():
    """Tight but incomplete: the crop is all target, the target is not all in."""
    pred = [120.0, 110.0, 180.0, 140.0]
    assert ev.gold_coverage(pred, GOLD) < 1.0
    assert ev.prediction_precision(pred, GOLD) == 1.0


def test_4_partial_overlap():
    pred = [150.0, 100.0, 250.0, 150.0]       # right half of gold
    assert ev.gold_coverage(pred, GOLD) == pytest.approx(0.5)
    assert ev.prediction_precision(pred, GOLD) == pytest.approx(0.5)
    assert 0 < ev.iou(GOLD, pred) < 0.5


def test_5_disjoint():
    pred = [400.0, 400.0, 500.0, 450.0]
    assert ev.gold_coverage(pred, GOLD) == 0.0
    assert ev.prediction_precision(pred, GOLD) == 0.0
    assert ev.iou(GOLD, pred) == 0.0


def test_degenerate_boxes_do_not_divide_by_zero():
    assert ev.gold_coverage(GOLD, [1, 1, 1, 1]) == 0.0
    assert ev.prediction_precision([1, 1, 1, 1], GOLD) == 0.0


# --------------------------------------------------------------------------
# verdicts — SYNTHETIC, rules live in code
# --------------------------------------------------------------------------

def test_6_contextual_full_coverage_is_never_precise():
    """Coverage 1.0 from a table-sized crop is 'the answer is somewhere in here',
    not a located field."""
    verdict, _ = ev.classify_locatable(ev.CONTEXTUAL, 0.105, 1.0)
    assert verdict == ev.CONTEXT_ONLY
    assert verdict != ev.PRECISE_MATCH


def test_precise_and_tight_is_a_precise_match():
    assert ev.classify_locatable(ev.PRECISE, 0.80, 1.0)[0] == ev.PRECISE_MATCH


def test_precise_covering_but_loose():
    assert ev.classify_locatable(ev.PRECISE, 0.65, 1.0)[0] == ev.COVERED_BUT_LOOSE


def test_structural_never_reaches_precise_match_even_when_tight():
    """Its grade forbids auto-answer; a tight box does not change the grade."""
    verdict, reason = ev.classify_locatable(ev.STRUCTURAL, 0.95, 1.0)
    assert verdict == ev.COVERED_BUT_LOOSE
    assert "not precise" in reason


def test_low_coverage_is_missed_whatever_the_grade():
    assert ev.classify_locatable(ev.PRECISE, 0.2, 0.3)[0] == ev.MISSED
    assert ev.classify_locatable(ev.CONTEXTUAL, 0.0, 0.0)[0] == ev.MISSED
    assert ev.classify_locatable(None, 0.0, 0.0)[0] == ev.MISSED


def test_7_unlocatable_with_a_non_eligible_region_is_a_cost_not_a_safety_error():
    verdict, _ = ev.classify_unlocatable({"answer_eligible": False}, any_region=True)
    assert verdict == ev.UNNECESSARY_REVIEW


def test_8_unlocatable_with_an_eligible_selected_region_is_unsafe():
    verdict, _ = ev.classify_unlocatable({"answer_eligible": True}, any_region=True)
    assert verdict == ev.UNSAFE_AUTO_ANSWER


def test_unlocatable_with_no_region_is_a_correct_abstention():
    assert ev.classify_unlocatable(None, any_region=False)[0] == ev.CORRECTLY_ABSTAINED


def _record(record_id, gold_bbox, region_type="value_cell"):
    return RegionGoldRecord(
        record_id=record_id, document_id="D", page_no=1, image_path="x.png",
        image_sha256="0", target_field=record_id.split(":")[-1],
        label_status=HUMAN_REVIEWED, gold_bbox=gold_bbox,
        gold_region_type=region_type, field_visible=region_type != "unlocatable",
        value_visible=region_type != "unlocatable",
        unlocatable_reason="blur" if region_type == "unlocatable" else None,
        annotator="t", annotated_at="t")


def _cand(bbox, strategy, trust, eligible=False, order=0):
    return {"bbox": bbox, "strategy": strategy, "trust_level": trust,
            "answer_eligible": eligible, "order": order, "area_ratio": 0.1}


def test_9_a_full_page_diagnostic_is_not_a_local_success():
    """It covers the target by construction. Reported as context only, and the
    top1 coverage — which excludes diagnostics, like the legacy top1 — is 0."""
    rec = _record("D:p1:图号", GOLD)
    preds = {rec.record_id: [_cand([0, 0, 1000, 800], ev.FULL_PAGE_DIAGNOSTIC,
                                   ev.DIAGNOSTIC)]}
    p1 = ev.evaluate_phase1([rec], preds)
    row = p1["records"][0]
    assert row["verdict"] == ev.CONTEXT_ONLY
    assert row["top1_gold_coverage"] == 0.0
    assert p1["coverage"]["all"]["top1_gold_coverage_recall"]["coverage@0.5"]["count"] == 0
    legacy = ev.evaluate([rec], preds)
    assert legacy["metrics"]["all"]["top1_localization_recall"]["iou@0.3"] == 0.0


def test_the_safety_and_cost_rates_are_separate_numbers():
    """Two unlocatable fields, one with a harmless review region, one with an
    answer-eligible one. The old single number would score both as failures."""
    a = _record("D:p1:a", None, "unlocatable")
    b = _record("D:p1:b", None, "unlocatable")
    preds = {
        a.record_id: [_cand([0, 0, 50, 50], "table_or_title_block", ev.CONTEXTUAL)],
        b.record_id: [_cand([0, 0, 50, 50], "exact_label_right", ev.PRECISE,
                            eligible=True)],
    }
    unl = ev.evaluate_phase1([a, b], preds)["unlocatable"]
    assert unl["unlocatable_auto_answer_false_positive_rate"]["count"] == 1
    assert unl["unlocatable_review_trigger_rate"]["count"] == 2
    assert unl["unlocatable_auto_answer_false_positive_rate"]["kind"] == "SAFETY"
    assert unl["unlocatable_review_trigger_rate"]["kind"] == "COST"
    legacy = ev.evaluate([a, b], preds)["metrics"]["unlocatable_detection_accuracy"]
    assert legacy == 0.0          # the merged number that this split replaces


def test_the_legacy_unlocatable_metric_is_labelled_as_legacy():
    note = ev.LEGACY_METRIC_NOTES["metrics.unlocatable_detection_accuracy"]
    assert "LEGACY" in note and "safety" in note and "cost" in note


# --------------------------------------------------------------------------
# REAL — the pre-registered facts, checked, not tuned toward
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    return ev.run(gold=GOLD_TEST, output_json=out / "r.json",
                  output_md=out / "r.md", explicit_outputs=True)


def _row(payload, record_id):
    return next(r for r in payload["phase1"]["records"] if r["record_id"] == record_id)


def test_real_legacy_numbers_are_identical_to_the_pre_phase1_report(real_payload):
    """REAL. Every key of the report written by the old evaluator is equal in
    the new one; phase 1 only adds keys."""
    old = json.loads(OLD_GOLD_TEST_REPORT.read_text(encoding="utf-8"))
    for key, value in old.items():
        assert real_payload[key] == value, key
    added = set(real_payload) - set(old)
    # Phase 1 added these two; phase 2 (gold schema v2) adds "phase2".
    assert {"phase1", "legacy_metric_notes"} <= added
    assert added <= {"phase1", "legacy_metric_notes", "phase2"}


def test_real_a23_label_crops_found_the_field_but_are_loose(real_payload):
    """REAL. Pre-registered: IoU unchanged (~0.65), coverage 1.0."""
    for record_id, expected_iou in (("FAN-A23-01:p1:图号", 0.652),
                                    ("FAN-A23-01:p1:断路器编号", 0.638)):
        row = _row(real_payload, record_id)
        assert row["top1_iou"] == pytest.approx(expected_iou, abs=1e-3)
        assert row["top1_gold_coverage"] == 1.0
        assert row["verdict"] == ev.COVERED_BUT_LOOSE


def test_real_a23_contextual_regions_are_low_iou_high_coverage(real_payload):
    """REAL. Pre-registered."""
    for field_name in ("名称", "控制柜编号", "电机编号"):
        row = _row(real_payload, f"FAN-A23-01:p1:{field_name}")
        assert row["top1_iou"] < 0.2
        assert row["top1_gold_coverage"] == 1.0
        assert row["verdict"] == ev.CONTEXT_ONLY


def test_real_a24_name_is_a_genuine_miss(real_payload):
    """REAL. Pre-registered: the gold is the page's top title; the crop is the
    title-block table below it."""
    row = _row(real_payload, "FAN-A24-01:p1:名称")
    assert row["top1_gold_coverage"] == 0.0
    assert row["verdict"] == ev.MISSED


def test_real_unlocatable_fields_are_cost_not_safety(real_payload):
    """REAL. 0 unsafe, 5 unnecessary reviews."""
    unl = real_payload["phase1"]["unlocatable"]
    assert unl["unlocatable_auto_answer_false_positive_rate"]["count"] == 0
    assert unl["unlocatable_review_trigger_rate"]["count"] == 5


# --------------------------------------------------------------------------
# CLI and path safety
# --------------------------------------------------------------------------

def test_10_cli_custom_input_and_output(tmp_path):
    out_json, out_md = tmp_path / "a.json", tmp_path / "a.md"
    code = ev.main(["--gold", str(GOLD_TEST), "--output-json", str(out_json),
                    "--output-md", str(out_md)])
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "evaluated"
    assert "phase1" in payload and "Phase 1" in out_md.read_text(encoding="utf-8")


def test_11_default_behaviour_is_byte_identical_to_the_historical_report(tmp_path):
    """The legacy default (unreviewed gold) must produce exactly what it always
    did. Written to tmp_path so the historical files are not touched."""
    ev.run(gold=ev.GOLD, output_json=tmp_path / "d.json", output_md=tmp_path / "d.md")
    assert (tmp_path / "d.json").read_bytes() == ev.OUT_JSON.read_bytes()
    assert (tmp_path / "d.md").read_bytes() == ev.OUT_MD.read_bytes()


def test_the_cli_defaults_still_point_at_the_legacy_paths():
    assert ev.GOLD.name == "review_region_gold_unreviewed.jsonl"
    assert ev.OUT_JSON.name == "review_region_localization_eval.json"
    assert ev.OUT_MD.name == "review_region_localization_eval.md"


@pytest.mark.parametrize("target", sorted(ev.PROTECTED_OUTPUTS))
def test_12_an_explicit_output_can_never_overwrite_a_historical_report(tmp_path, target):
    before = sha(target) if target.exists() else None
    other = tmp_path / ("x.md" if target.suffix == ".json" else "x.json")
    args = (["--output-json", str(target), "--output-md", str(other)]
            if target.suffix == ".json" else
            ["--output-json", str(other), "--output-md", str(target)])
    code = ev.main(["--gold", str(GOLD_TEST), "--overwrite", *args])
    assert code == 2
    if before is not None:
        assert sha(target) == before


def test_an_existing_output_needs_overwrite(tmp_path):
    out_json, out_md = tmp_path / "a.json", tmp_path / "a.md"
    out_json.write_text("{}", encoding="utf-8")
    assert ev.main(["--gold", str(GOLD_TEST), "--output-json", str(out_json),
                    "--output-md", str(out_md)]) == 2
    assert out_json.read_text(encoding="utf-8") == "{}"
    assert ev.main(["--gold", str(GOLD_TEST), "--output-json", str(out_json),
                    "--output-md", str(out_md), "--overwrite"]) == 0


def test_a_missing_gold_fails_instead_of_falling_back(tmp_path):
    code = ev.main(["--gold", str(tmp_path / "nope.jsonl"),
                    "--output-json", str(tmp_path / "a.json"),
                    "--output-md", str(tmp_path / "a.md")])
    assert code == 2
    assert not (tmp_path / "a.json").exists()


def test_a_schema_invalid_gold_fails(tmp_path):
    bad = tmp_path / "bad.jsonl"
    row = json.loads(GOLD_TEST.read_text(encoding="utf-8").splitlines()[0])
    row["annotator"] = None                      # human_reviewed without annotator
    bad.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ev.EvaluationInputError, match="no annotator"):
        ev.run(gold=bad, output_json=tmp_path / "a.json",
               output_md=tmp_path / "a.md", explicit_outputs=True)


def test_output_paths_that_collide_are_rejected(tmp_path):
    same = tmp_path / "same"
    with pytest.raises(ev.EvaluationInputError, match="same path"):
        ev.run(gold=GOLD_TEST, output_json=same, output_md=same,
               explicit_outputs=True)
    with pytest.raises(ev.EvaluationInputError, match="gold input"):
        ev.run(gold=GOLD_TEST, output_json=GOLD_TEST, output_md=tmp_path / "a.md",
               explicit_outputs=True, overwrite=True)


def test_output_flags_must_come_as_a_pair(tmp_path):
    with pytest.raises(SystemExit):
        ev.main(["--gold", str(GOLD_TEST), "--output-json", str(tmp_path / "a.json")])


def test_the_users_gold_file_is_untouched_by_evaluation(tmp_path):
    before = sha(GOLD_TEST)
    ev.run(gold=GOLD_TEST, output_json=tmp_path / "a.json",
           output_md=tmp_path / "a.md", explicit_outputs=True)
    assert sha(GOLD_TEST) == before
