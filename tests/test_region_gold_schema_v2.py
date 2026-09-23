"""Gold schema v2: visible / legible / attributed, kept apart.

Every record demonstrating a v2 state is a SYNTHETIC fixture built here. None
is derived from OCR, resolver or VLM output, and none rewrites a human
judgement from data/review_region_gold_test.jsonl — that file is only read,
and its hash is asserted unchanged.

No network, no model, no paid API.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluate_review_region_localization as ev
import scripts.migrate_review_region_gold as mig
from src.vision.region_gold import (
    AMBIGUOUS, CONFIRMED, HUMAN_REVIEWED, SCHEMA_V1, SCHEMA_V2, UNASSIGNED,
    UNKNOWN, UNLOCATABLE, UNREVIEWED, VALUE_CELL, RegionGoldRecord, read_jsonl,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_TEST = ROOT / "data" / "review_region_gold_test.jsonl"
GOLD_TEST_SHA256 = "dbfc794a9ae188bfdaa9ba8cc68eb6133ebcd0ed1f88f144606feba469583953"

BOX = [220.0, 630.0, 940.0, 660.0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v2(**over) -> RegionGoldRecord:
    """SYNTHETIC native-v2 human-reviewed record; defaults are a confirmed field."""
    base = dict(
        record_id="SYN:p1:控制柜编号", document_id="SYN", page_no=1,
        image_path="x.png", image_sha256="0", target_field="控制柜编号",
        label_status=HUMAN_REVIEWED, annotator="t", annotated_at="t",
        schema_version=SCHEMA_V2, field_visible=True, value_visible=True,
        value_legible=True, association_status=CONFIRMED,
        gold_region_type=VALUE_CELL, gold_bbox=list(BOX))
    base.update(over)
    return RegionGoldRecord(**base)


def unattributed(status=UNASSIGNED, **over) -> RegionGoldRecord:
    """SYNTHETIC: a readable value whose field cannot be confirmed."""
    base = dict(association_status=status, gold_region_type=UNLOCATABLE,
                gold_bbox=None, field_visible=False,
                unlocatable_reason="label_destroyed_value_legible",
                candidate_bbox=list(BOX))
    if status == AMBIGUOUS:
        base["candidate_fields"] = ["控制柜编号", "断路器编号"]
    base.update(over)
    return v2(**base)


# --------------------------------------------------------------------------
# 1–2  old gold still reads, and reads as unknown
# --------------------------------------------------------------------------

def test_1_the_v1_gold_file_still_loads_and_validates():
    records = read_jsonl(GOLD_TEST)
    assert len(records) == 11
    assert {r.schema_version for r in records} == {SCHEMA_V1}
    assert all(validate_record(r) == [] for r in records)


def test_2_missing_new_fields_resolve_to_unknown_never_confirmed():
    """Including the 6 records a person clearly located: v1 never asked about
    attribution, so a gold_bbox is not an answer to it."""
    records = read_jsonl(GOLD_TEST)
    assert {r.effective_association for r in records} == {UNKNOWN}
    located = [r for r in records if r.gold_bbox]
    assert len(located) == 6
    assert all(r.effective_association == UNKNOWN for r in located)
    assert all(r.value_legible is None for r in records)


def test_a_v1_row_carrying_v2_fields_is_rejected_not_guessed():
    row = json.loads(GOLD_TEST.read_text(encoding="utf-8").splitlines()[0])
    row["association_status"] = CONFIRMED          # no schema_version bump
    problems = validate_record(RegionGoldRecord(**row))
    assert any("v1 record carries v2 fields" in p for p in problems)


# --------------------------------------------------------------------------
# 3–4  confirmed
# --------------------------------------------------------------------------

def test_3_confirmed_with_a_valid_bbox_is_legal():
    assert validate_record(v2(), image_size=(1000, 800)) == []


def test_4_confirmed_without_a_bbox_is_illegal():
    problems = validate_record(v2(gold_bbox=None))
    assert any("confirmed needs a valid gold_bbox" in p for p in problems)


def test_confirmed_on_an_unreadable_value_is_illegal():
    problems = validate_record(v2(value_legible=False))
    assert any("confirmed needs value_legible=true" in p for p in problems)
    assert any("value_legible=false allows only" in p for p in problems)


def test_confirmed_may_not_also_carry_a_candidate_box():
    problems = validate_record(v2(candidate_bbox=list(BOX)))
    assert any("must not carry candidate_bbox" in p for p in problems)


# --------------------------------------------------------------------------
# 9  readable but unattributed — the case v1 could not express
# --------------------------------------------------------------------------

def test_9_readable_value_with_unconfirmed_field_is_expressible():
    """SYNTHETIC. v1 had to either claim attribution or deny legibility."""
    for status in (UNASSIGNED, AMBIGUOUS):
        record = unattributed(status)
        assert validate_record(record, image_size=(1000, 800)) == [], status
        assert record.value_visible and record.value_legible
        assert record.gold_bbox is None and record.candidate_bbox == BOX


def test_v1_forbade_that_same_shape():
    """The combination v2 legalises is still an error under v1 rules."""
    v1_shape = RegionGoldRecord(
        record_id="SYN:p1:x", document_id="SYN", page_no=1, image_path="x.png",
        image_sha256="0", target_field="x", label_status=HUMAN_REVIEWED,
        annotator="t", annotated_at="t", gold_region_type=VALUE_CELL,
        gold_bbox=list(BOX), field_visible=False, value_visible=True)
    assert any("value_visible=true while field_visible=false" in p
               for p in validate_record(v1_shape))


def test_unattributed_may_not_use_gold_bbox():
    """gold_bbox means 'a person located THIS field'. A candidate goes elsewhere."""
    problems = validate_record(unattributed(gold_bbox=list(BOX)))
    assert any("must not carry gold_bbox" in p for p in problems)


def test_unattributed_needs_the_field_marked_unlocatable():
    problems = validate_record(unattributed(gold_region_type=VALUE_CELL))
    assert any("gold_region_type=unlocatable" in p for p in problems)


def test_ambiguous_needs_at_least_two_candidate_fields_including_the_target():
    problems = validate_record(unattributed(AMBIGUOUS, candidate_fields=["控制柜编号"]))
    assert any("at least two fields" in p for p in problems)
    problems = validate_record(unattributed(AMBIGUOUS,
                                            candidate_fields=["电机编号", "断路器编号"]))
    assert any("including target_field" in p for p in problems)


def test_an_invalid_candidate_box_is_reported_not_repaired():
    problems = validate_record(unattributed(candidate_bbox=[500, 10, 400, 5]))
    assert any("candidate_bbox is not a valid" in p for p in problems)


# --------------------------------------------------------------------------
# 10  unreadable
# --------------------------------------------------------------------------

def test_10_an_unreadable_value_is_unknown_and_nothing_else():
    """SYNTHETIC."""
    ok = v2(value_legible=False, association_status=UNKNOWN, gold_bbox=None,
            gold_region_type=UNLOCATABLE, unlocatable_reason="severe_blur")
    assert validate_record(ok) == []
    for status in (UNASSIGNED, AMBIGUOUS):
        bad = unattributed(status, value_legible=False)
        assert any("an unreadable value is `unknown`" in p
                   for p in validate_record(bad)), status


def test_a_native_v2_legible_value_must_decide_its_attribution():
    """`unknown` is for 'not asked' or 'unreadable' — a fresh v2 annotation
    that read the value has no excuse not to answer."""
    problems = validate_record(v2(association_status=UNKNOWN))
    assert any("decide confirmed, ambiguous or unassigned" in p for p in problems)


def test_unreviewed_v2_records_hold_no_attribution_conclusion():
    pending = RegionGoldRecord.pending(
        document_id="FAN-A23-01", page_no=1,
        image_path=ROOT / "data" / "drawings" / "FAN-A23-01.png",
        target_field="图号")
    assert pending.schema_version == SCHEMA_V2
    assert validate_record(pending) == []
    pending.association_status = CONFIRMED
    assert any("pending record must hold no conclusion" in p
               for p in validate_record(pending))


# --------------------------------------------------------------------------
# 5–8  evaluator
# --------------------------------------------------------------------------

def _cand(eligible: bool, box=BOX, order=0):
    return {"bbox": list(box), "strategy": "exact_label_right" if eligible
            else "table_or_title_block", "trust_level": "precise" if eligible
            else "contextual", "answer_eligible": eligible, "order": order,
            "area_ratio": 0.1}


def test_5_ambiguous_is_outside_the_localisation_denominator():
    confirmed = v2(record_id="SYN:p1:a", target_field="a")
    amb = unattributed(AMBIGUOUS, record_id="SYN:p1:控制柜编号")
    preds = {confirmed.record_id: [_cand(True)], amb.record_id: [_cand(True)]}
    p2 = ev.evaluate_phase2([confirmed, amb], preds)
    assert p2["counts"][AMBIGUOUS] == 1
    formal = p2["formal_localization"]
    assert formal["status"] == "evaluated" and formal["n"] == 1


def test_6_an_auto_answer_on_an_ambiguous_value_is_a_safety_failure():
    amb = unattributed(AMBIGUOUS)
    p2 = ev.evaluate_phase2([amb], {amb.record_id: [_cand(True)]})
    fp = p2["ambiguous"]["ambiguous_auto_answer_false_positive_rate"]
    assert fp["count"] == 1 and fp["kind"] == "SAFETY"


def test_7_an_unassigned_value_that_is_blocked_is_counted_as_blocked():
    una = unattributed(UNASSIGNED)
    p2 = ev.evaluate_phase2([una], {una.record_id: [_cand(False)]})
    block = p2["unassigned"]
    assert block["unassigned_auto_answer_false_positive_rate"]["count"] == 0
    assert block["blocked_auto_answer"]["count"] == 1
    # The review region did include the visible candidate; recorded, not scored.
    assert block["records"][0]["candidate_coverage"] == 1.0


def test_7b_an_unassigned_value_that_is_NOT_blocked_is_flagged():
    una = unattributed(UNASSIGNED)
    p2 = ev.evaluate_phase2([una], {una.record_id: [_cand(True)]})
    assert p2["unassigned"]["unassigned_auto_answer_false_positive_rate"]["count"] == 1


def test_8_unknown_never_enters_formal_accuracy():
    """REAL. The user's gold is entirely v1, so the formal attribution-aware
    evaluation has nothing to score — it says so instead of scoring."""
    payload = ev.build_payload(read_jsonl(GOLD_TEST), ev.load_predictions())
    p2 = payload["phase2"]
    assert p2["counts"] == {CONFIRMED: 0, AMBIGUOUS: 0, UNASSIGNED: 0, UNKNOWN: 11}
    assert p2["formal_localization"]["status"] == "not_evaluated"
    assert p2["formal_localization"]["reason"] == "no_confirmed_attribution_gold"
    assert p2["unknown"]["from_schema_v1"] == 11


def test_phase2_leaves_every_earlier_number_untouched(tmp_path):
    """REAL. The phase-1 report's keys are all equal; only `phase2` is added."""
    old = json.loads((ROOT / "outputs" /
                      "review_region_localization_eval_phase1_gold_test.json")
                     .read_text(encoding="utf-8"))
    new = ev.run(gold=GOLD_TEST, output_json=tmp_path / "a.json",
                 output_md=tmp_path / "a.md", explicit_outputs=True)
    for key, value in old.items():
        assert new[key] == value, key
    assert set(new) - set(old) == {"phase2"}


def test_the_default_evaluator_output_is_still_byte_identical(tmp_path):
    ev.run(gold=ev.GOLD, output_json=tmp_path / "d.json", output_md=tmp_path / "d.md")
    assert (tmp_path / "d.json").read_bytes() == ev.OUT_JSON.read_bytes()
    assert (tmp_path / "d.md").read_bytes() == ev.OUT_MD.read_bytes()


def test_the_phase1_report_is_now_protected(tmp_path):
    target = ROOT / "outputs" / "review_region_localization_eval_phase1_gold_test.json"
    before = sha(target)
    code = ev.main(["--gold", str(GOLD_TEST), "--overwrite",
                    "--output-json", str(target),
                    "--output-md", str(tmp_path / "x.md")])
    assert code == 2 and sha(target) == before


# --------------------------------------------------------------------------
# 11–12  migration preview; the gold never moves
# --------------------------------------------------------------------------

def test_11_the_migration_preview_writes_nothing(tmp_path, capsys):
    before = sha(GOLD_TEST)
    assert mig.main(["--gold", str(GOLD_TEST)]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "inferred fields: NONE" in out
    assert sha(GOLD_TEST) == before


def test_migration_infers_nothing_and_changes_no_human_judgement(tmp_path):
    target = tmp_path / "migrated.jsonl"
    report = mig.run(GOLD_TEST, write_to=target)
    assert report["inferred_fields"] == []
    assert report["human_judgements_changed"] == 0
    old = {r.record_id: r for r in read_jsonl(GOLD_TEST)}
    for new in read_jsonl(target):
        before = old[new.record_id]
        assert new.schema_version == SCHEMA_V2
        assert new.migrated_from == SCHEMA_V1
        assert new.association_status == UNKNOWN
        assert new.value_legible is None
        assert new.candidate_bbox is None and new.candidate_fields is None
        for name in ("gold_bbox", "gold_region_type", "field_visible",
                     "value_visible", "unlocatable_reason", "annotator",
                     "annotated_at", "notes", "label_status"):
            assert getattr(new, name) == getattr(before, name), name
        assert validate_record(new) == []


def test_migration_refuses_to_overwrite_anything(tmp_path):
    with pytest.raises(mig.MigrationError):
        mig.run(GOLD_TEST, write_to=GOLD_TEST)
    existing = tmp_path / "exists.jsonl"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(mig.MigrationError, match="never overwrites"):
        mig.run(GOLD_TEST, write_to=existing)
    assert existing.read_text(encoding="utf-8") == "x"


def test_migration_reads_no_ocr_resolver_or_vlm():
    """Checked in the imports, not asserted in prose."""
    tree = ast.parse((ROOT / "scripts" / "migrate_review_region_gold.py")
                     .read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    forbidden = ("ocr", "resolver", "multimodal", "vlm", "qwen", "tables")
    assert not [m for m in modules if any(f in m for f in forbidden)], modules


def test_12_the_users_gold_file_is_byte_identical():
    assert sha(GOLD_TEST) == GOLD_TEST_SHA256
