"""Gold contract, evaluator refusal, and shadow candidates.

MOST TESTS HERE ARE **SYNTHETIC REGRESSION**. They build their own tables and
prove the code behaves as specified; they prove NOTHING about real drawings.
Tests that touch a real page are marked REAL and say what they read.

No network, no model, no paid API anywhere in this file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.vision.region_gold import (
    EXCLUDED, FULL_PAGE, HUMAN_REVIEWED, LABEL_VALUE_PAIR, TABLE_REGION,
    UNLOCATABLE, UNREVIEWED, VALUE_CELL, RegionGoldRecord, read_jsonl,
    reviewed_only, status_summary, validate_record, write_jsonl,
)
from src.vision.review_region_config import load_review_region_config
from src.vision.review_region_resolver import identify_key_value_rows
from src.vision.structural_region_candidates import (
    ORDER_ONLY, SHADOW_ONLY, SINGLE_SIDED, KV_ROW, ShadowConfig,
    generate_structural_candidates, load_shadow_config, score_gap,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_FILE = ROOT / "data" / "review_region_gold_unreviewed.jsonl"
DRAWINGS = ROOT / "data" / "drawings"


@pytest.fixture(scope="module")
def rcfg():
    return load_review_region_config()


@pytest.fixture(scope="module")
def scfg():
    return load_shadow_config()


# --------------------------------------------------------------------------
# synthetic table fixtures
# --------------------------------------------------------------------------

@dataclass
class Cell:
    row_start: int
    col_start: int
    bbox: List[float]
    text_normalized: str = ""


@dataclass
class Table:
    cells: List[Cell]
    bbox: List[float]
    col_count: int = 2
    table_id: str = "t1"
    header_rows: Optional[List[int]] = None

    def __post_init__(self):
        if self.header_rows is None:
            self.header_rows = []

    def data_rows(self) -> List[int]:
        return sorted({c.row_start for c in self.cells})

    def cell_at(self, row, col):
        return next((c for c in self.cells
                     if c.row_start == row and c.col_start == col), None)


def kv(rows, *, x0=100, y0=100, w=600, row_h=40, skip_left=()) -> Table:
    """SYNTHETIC two-column key/value table."""
    cells = []
    for index, (left, right) in enumerate(rows):
        top = y0 + index * row_h
        if index not in skip_left:
            cells.append(Cell(index, 0, [x0, top, x0 + 240, top + row_h], left))
        cells.append(Cell(index, 1, [x0 + 240, top, x0 + w, top + row_h], right))
    return Table(cells=cells, bbox=[x0, y0, x0 + w, y0 + len(rows) * row_h])


def record(**kwargs) -> RegionGoldRecord:
    base = dict(record_id="D:p1:图号", document_id="D", page_no=1,
                image_path="data/drawings/FAN-A23-01.png",
                image_sha256="deadbeef", target_field="图号")
    return RegionGoldRecord(**{**base, **kwargs})


# --------------------------------------------------------------------------
# Gold contract
# --------------------------------------------------------------------------

def test_a_pending_record_carries_no_conclusion():
    """The whole point: nothing in the codebase can pre-fill an answer."""
    pending = RegionGoldRecord.pending(
        document_id="FAN-A23-01", page_no=1,
        image_path=DRAWINGS / "FAN-A23-01.png", target_field="图号")
    assert pending.label_status == UNREVIEWED
    assert pending.gold_bbox is None
    assert pending.gold_region_type is None
    assert pending.annotator is None and pending.annotated_at is None
    assert pending.needs_visual_review is True
    assert validate_record(pending) == []


def test_there_is_no_way_to_build_gold_from_a_prediction():
    """A constructor taking a resolver output would make the circular gold a
    one-liner. Its absence is the guard."""
    assert not hasattr(RegionGoldRecord, "from_prediction")
    assert not hasattr(RegionGoldRecord, "from_region")
    # Check the imports, not the prose: the docstring names the type precisely
    # to explain why it is absent.
    import ast
    tree = ast.parse((ROOT / "src" / "vision" / "region_gold.py")
                     .read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("review_region_resolver" in m or "multimodal" in m
                   for m in imported), imported


def test_an_unreviewed_record_holding_a_bbox_is_rejected():
    problems = validate_record(record(label_status=UNREVIEWED,
                                      gold_bbox=[1, 2, 3, 4]))
    assert any("must hold no conclusion" in p for p in problems)


def test_human_reviewed_without_an_annotator_is_rejected():
    problems = validate_record(record(
        label_status=HUMAN_REVIEWED, gold_bbox=[1, 2, 30, 40],
        gold_region_type=VALUE_CELL, annotated_at="2026-09-17"))
    assert any("no annotator" in p for p in problems)


def test_human_reviewed_without_a_timestamp_is_rejected():
    problems = validate_record(record(
        label_status=HUMAN_REVIEWED, gold_bbox=[1, 2, 30, 40],
        gold_region_type=VALUE_CELL, annotator="ann"))
    assert any("no annotated_at" in p for p in problems)


@pytest.mark.parametrize("bbox,fragment", [
    ([10, 10, 10, 40], "zero or negative area"),
    ([-5, 10, 50, 40], "negative coordinates"),
    ([10, 10, 5000, 40], "extends past the image"),
])
def test_an_illegal_bbox_is_rejected(bbox, fragment):
    problems = validate_record(
        record(label_status=HUMAN_REVIEWED, gold_bbox=bbox,
               gold_region_type=VALUE_CELL, annotator="a", annotated_at="t"),
        image_size=(1000, 800))
    assert any(fragment in p for p in problems)


def test_visible_false_with_a_bbox_is_contradictory():
    problems = validate_record(
        record(label_status=HUMAN_REVIEWED, gold_bbox=[10, 10, 50, 40],
               gold_region_type=VALUE_CELL, field_visible=False,
               annotator="a", annotated_at="t"), image_size=(1000, 800))
    assert any("field_visible=false but a gold_bbox" in p for p in problems)


def test_unlocatable_needs_a_reason_and_no_bbox():
    problems = validate_record(record(
        label_status=HUMAN_REVIEWED, gold_region_type=UNLOCATABLE,
        gold_bbox=[1, 2, 30, 40], annotator="a", annotated_at="t"))
    assert any("no unlocatable_reason" in p for p in problems)
    assert any("must not carry a bbox" in p for p in problems)


def test_excluded_needs_a_note():
    assert any("must say why" in p
               for p in validate_record(record(label_status=EXCLUDED)))
    assert validate_record(record(label_status=EXCLUDED, notes="两个候选分不清")) == []


def test_unreviewed_and_excluded_are_outside_the_denominator():
    """Counting an unreviewed record as a miss would turn 'nobody looked yet'
    into evidence that the system failed."""
    records = [record(record_id="a", label_status=UNREVIEWED),
               record(record_id="b", label_status=EXCLUDED, notes="n"),
               record(record_id="c", label_status=HUMAN_REVIEWED,
                      gold_bbox=[1, 1, 20, 20], gold_region_type=VALUE_CELL,
                      annotator="a", annotated_at="t")]
    assert [r.record_id for r in reviewed_only(records)] == ["c"]
    assert status_summary(records) == {UNREVIEWED: 1, HUMAN_REVIEWED: 1,
                                       EXCLUDED: 1}


def test_a_table_region_gold_is_not_a_field_level_answer_key():
    assert record(gold_region_type=VALUE_CELL).is_field_level
    assert record(gold_region_type=LABEL_VALUE_PAIR).is_field_level
    assert not record(gold_region_type=TABLE_REGION).is_field_level
    assert not record(gold_region_type=FULL_PAGE).is_field_level


def test_the_shipped_gold_file_is_entirely_unreviewed():
    """REAL FILE. Nothing in this round may have produced a human conclusion."""
    records = read_jsonl(GOLD_FILE)
    assert records, "expected pending records"
    assert all(r.label_status == UNREVIEWED for r in records)
    assert all(r.gold_bbox is None and r.annotator is None for r in records)


def test_records_round_trip_through_jsonl(tmp_path):
    original = [RegionGoldRecord.pending(
        document_id="FAN-A23-01", page_no=1,
        image_path=DRAWINGS / "FAN-A23-01.png", target_field="图号")]
    path = tmp_path / "g.jsonl"
    assert write_jsonl(original, path) == 1
    assert read_jsonl(path)[0].record_id == original[0].record_id


# --------------------------------------------------------------------------
# evaluator refusal
# --------------------------------------------------------------------------

def test_the_evaluator_refuses_to_produce_a_number_without_gold():
    """REAL OUTPUT. Emitting 0% or 100% here would read as a measurement."""
    payload = json.loads(
        (ROOT / "outputs" / "review_region_localization_eval.json")
        .read_text(encoding="utf-8"))
    assert payload["status"] == "not_evaluated"
    assert payload["reason"] == "no_human_reviewed_bbox_gold"
    assert payload["metrics"] is None
    assert payload["human_reviewed_count"] == 0


def test_a_full_page_candidate_can_never_count_as_localisation():
    """A box containing the whole page contains every gold box."""
    from scripts.evaluate_review_region_localization import NON_LOCALISING
    from src.vision.review_region_resolver import FULL_PAGE_DIAGNOSTIC
    assert FULL_PAGE_DIAGNOSTIC in NON_LOCALISING


def test_a_stale_image_hash_blocks_evaluation(tmp_path):
    """A gold bbox belongs to the pixels it was drawn on."""
    from src.vision.region_gold import image_hash_matches
    stale = record(label_status=HUMAN_REVIEWED, image_sha256="0" * 64,
                   gold_bbox=[1, 1, 20, 20], gold_region_type=VALUE_CELL,
                   annotator="a", annotated_at="t")
    assert image_hash_matches(stale, ROOT) is False


# --------------------------------------------------------------------------
# shadow candidates — SYNTHETIC REGRESSION unless marked REAL
# --------------------------------------------------------------------------

def test_shadow_candidates_never_enter_production(rcfg):
    """The isolation guarantee, checked in the source rather than asserted."""
    resolver = (ROOT / "src" / "vision" / "review_region_resolver.py").read_text(
        encoding="utf-8")
    assert "structural_region_candidates" not in resolver
    for module in ("src/multimodal/pipeline.py", "src/multimodal/decision.py",
                   "src/multimodal/executor.py", "src/rag/policy.py"):
        text = (ROOT / module).read_text(encoding="utf-8")
        assert "structural_region_candidates" not in text, module


def test_every_candidate_is_marked_shadow_only(scfg, rcfg):
    """SYNTHETIC."""
    table = kv([("图号", "X"), ("###", "M-1"), ("页码", "1/1")])
    rows = identify_key_value_rows(table, rcfg)
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows=rows)
    assert cands
    assert all(c.promotion_status == SHADOW_ONLY for c in cands)


def test_1_intact_labels_score_on_text_evidence(scfg, rcfg):
    """SYNTHETIC. Case 1: labels complete."""
    table = kv([("图号", "X"), ("电机编号", "M-1")])
    rows = identify_key_value_rows(table, rcfg)
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows=rows)
    top = cands[0]
    assert top.row_index == 1
    assert top.evidence["text_similarity_score"] > 0.9


def test_2_a_label_missing_one_character_still_scores_text(scfg, rcfg):
    """SYNTHETIC. Case 2."""
    table = kv([("图号", "X"), ("电机号", "M-1")])
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows={})
    row1 = next(c for c in cands if c.row_index == 1)
    assert row1.evidence["text_similarity_score"] > 0


def test_3_a_destroyed_label_scores_no_text_evidence(scfg, rcfg):
    """SYNTHETIC. Case 3. It must not be rescued by inventing similarity."""
    table = kv([("图号", "X"), ("@@@@", "M-1")])
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows={})
    row1 = next(c for c in cands if c.row_index == 1)
    assert row1.evidence["text_similarity_score"] == 0.0


def test_4_an_anchor_above_only_is_penalised_and_recorded(scfg, rcfg):
    """SYNTHETIC. Case 4 — the case the shipped L3 refuses outright."""
    table = kv([("图号", "X"), ("###", "M-1")])
    rows = identify_key_value_rows(table, rcfg)
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows=rows)
    row1 = next(c for c in cands if c.row_index == 1)
    assert row1.strategy == SINGLE_SIDED
    assert any("one side only" in a for a in row1.assumptions)


def test_5_an_anchor_below_only_is_also_single_sided(scfg, rcfg):
    """SYNTHETIC. Case 5."""
    table = kv([("###", "M-1"), ("页码", "1/1")])
    rows = identify_key_value_rows(table, rcfg)
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows=rows)
    row0 = next(c for c in cands if c.row_index == 0)
    assert row0.strategy == SINGLE_SIDED


def test_6_two_sided_anchors_outscore_one_sided(scfg, rcfg):
    """SYNTHETIC. Case 6. The penalty must actually change the ordering."""
    two = kv([("图号", "X"), ("###", "M-1"), ("页码", "1/1")])
    one = kv([("图号", "X"), ("###", "M-1")])
    rows_two = identify_key_value_rows(two, rcfg)
    rows_one = identify_key_value_rows(one, rcfg)
    c_two = next(c for c in generate_structural_candidates(
        two, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"),
        identified_rows=rows_two) if c.row_index == 1)
    c_one = next(c for c in generate_structural_candidates(
        one, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"),
        identified_rows=rows_one) if c.row_index == 1)
    assert c_two.strategy == KV_ROW
    assert c_two.score > c_one.score


def test_7_a_different_drawing_type_never_borrows_another_order(scfg):
    """SYNTHETIC. Case 7 + 13. A fan_group title block is not a fan_wiring one."""
    assert scfg.order_for("fan_wiring") != scfg.order_for("fan_group")
    assert scfg.order_for("pump_schedule") == []      # unknown type: no order
    assert scfg.order_for(None) == []


def test_8_several_similar_labels_produce_several_candidates(scfg, rcfg):
    """SYNTHETIC. Case 8 + 17: near-tied scores are emitted, not resolved."""
    table = kv([("电机编号", "M-1"), ("电机号", "M-2"), ("电动机编号", "M-3")])
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows={})
    assert len(cands) >= 2
    assert all(c.ambiguity_count == len(cands) for c in cands)


def test_9_a_merged_cell_row_with_no_left_cell_is_skipped_not_guessed(scfg, rcfg):
    """SYNTHETIC. Case 9."""
    table = kv([("图号", "X"), ("电机编号", "M-1")], skip_left=(1,))
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows={})
    row1 = next((c for c in cands if c.row_index == 1), None)
    assert row1 is None or row1.evidence["text_similarity_score"] == 0.0


def test_10_a_missing_row_does_not_invent_one(scfg, rcfg):
    """SYNTHETIC. Case 10. Candidates only ever point at rows that exist."""
    table = kv([("图号", "X"), ("页码", "1/1")])
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"),
        identified_rows=identify_key_value_rows(table, rcfg))
    assert all(c.row_index in table.data_rows() for c in cands)


def test_11_a_cell_with_a_degenerate_bbox_produces_no_candidate(scfg, rcfg):
    """SYNTHETIC. Case 11."""
    table = kv([("电机编号", "M-1")])
    table.cells[1].bbox = [100, 100, 100, 100]
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows={})
    assert all(c.evidence["grid_consistency_score"] == 0.0 for c in cands) or not cands


def test_12_a_row_already_read_as_another_field_is_not_a_candidate(scfg, rcfg):
    """SYNTHETIC. Case 12 — an OCR block spanning rows would otherwise let one
    row stand for two fields."""
    table = kv([("图号", "X"), ("电机编号", "M-1"), ("###", "?")])
    rows = identify_key_value_rows(table, rcfg)
    cands = generate_structural_candidates(
        table, [], "控制柜编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("控制柜编号"), identified_rows=rows)
    assert all(c.row_index not in (0, 1) for c in cands)


def test_14_a_non_key_value_page_yields_no_candidate(scfg, rcfg):
    """REAL IMAGE. FAN-A23-01's 6x2 grid is not a confirmed key/value layout,
    so the shadow generator declines rather than guessing."""
    payload = json.loads(
        (ROOT / "outputs" / "structural_region_shadow_report.json")
        .read_text(encoding="utf-8"))
    page = next(p for p in payload["pages"] if p["document_id"] == "FAN-A23-01")
    assert page["key_value_table"] is False
    assert all(f["candidate_count"] == 0 for f in page["fields"])
    assert all("no confirmed key/value table" in (f["no_candidate_reason"] or "")
               for f in page["fields"])


def test_15_16_a_page_with_no_table_yields_no_candidate(scfg, rcfg):
    """SYNTHETIC. Cases 15 + 16: unknown type, no table."""
    assert generate_structural_candidates(
        None, [], "图号", None, scfg, identified_rows={}) == []


def test_17_tied_scores_are_reported_not_broken(scfg, rcfg):
    """REAL OUTPUT. On FAN-A24-01 the 页码|版本 candidates tie exactly."""
    payload = json.loads(
        (ROOT / "outputs" / "structural_region_shadow_report.json")
        .read_text(encoding="utf-8"))
    page = next(p for p in payload["pages"] if p["document_id"] == "FAN-A24-01")
    tied = [f for f in page["fields"] if f["top1_minus_top2"] == 0.0]
    assert tied, "expected at least one exact tie"
    for f in tied:
        assert f["ambiguity_count"] > 1


def test_18_sub_scores_are_never_collapsed_into_one_number(scfg, rcfg):
    """A total of 0.6 says nothing about whether the row was read, bounded or
    merely assumed."""
    table = kv([("图号", "X"), ("电机编号", "M-1")])
    cand = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows={})[0]
    assert set(cand.evidence) == {
        "text_similarity_score", "neighbor_anchor_score", "field_order_score",
        "grid_consistency_score", "cell_content_score"}


def test_field_order_is_marked_as_a_local_convention(scfg, rcfg):
    table = kv([("图号", "X"), ("###", "M-1"), ("页码", "1/1")])
    cand = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"),
        identified_rows=identify_key_value_rows(table, rcfg))[0]
    assert cand.provenance["field_order"] == "dataset_or_project_convention"
    assert cand.provenance["calibrated_with_human_gold"] == "False"


def test_shadow_weights_come_from_config_and_are_uncalibrated(scfg):
    raw = (ROOT / "configs" / "structural_shadow.yaml").read_text(encoding="utf-8")
    for key in ("text_similarity", "neighbor_anchor", "field_order",
                "grid_consistency", "cell_content"):
        assert key in raw
    assert scfg.calibrated_with_human_gold is False
    assert scfg.threshold_provenance == "engineering_initial_value"


def test_score_gap_is_none_for_a_single_candidate(scfg, rcfg):
    table = kv([("电机编号", "M-1")])
    cands = generate_structural_candidates(
        table, [], "电机编号", "fan_wiring", scfg,
        aliases=rcfg.field_aliases.get("电机编号"), identified_rows={})
    assert score_gap(cands) is None


def test_production_decisions_are_unchanged_by_this_round():
    """REAL OUTPUT. The shipped availability numbers must be untouched."""
    m = json.loads((ROOT / "outputs" / "review_region_resolution_report.json")
                   .read_text(encoding="utf-8"))["metrics"]
    assert m["precise_crop_available_count"] == 2
    assert m["any_region_available_count"] == 11
    assert m["crop_unavailable_count"] == 0


# --------------------------------------------------------------------------
# local review queue.  SYNTHETIC REGRESSION.
# --------------------------------------------------------------------------

def _queue_kwargs(**over):
    base = dict(document_id="D", page_no=1, target_fields=["图号"],
                strategy="exact_label_right", trust_level="precise",
                bbox=[10.0, 20.0, 110.0, 60.0], reason_codes=["required_field_missing"],
                image_sha256="abc", prompt_version="v1", schema_version="s1")
    base.update(over)
    return base


def test_the_queue_is_idempotent_on_identical_work(tmp_path):
    """SYNTHETIC. Rebuilding after nothing changed must not grow the queue."""
    from src.vision.review_queue import ReviewQueue
    queue = ReviewQueue(tmp_path / "q.json")
    assert queue.add(**_queue_kwargs()) is True
    assert queue.add(**_queue_kwargs()) is False
    assert queue.add(**_queue_kwargs(bbox=[10.2, 20.4, 110.1, 59.8])) is False
    assert len(queue) == 1


@pytest.mark.parametrize("field_name,value", [
    ("image_sha256", "different"),
    ("prompt_version", "v2"),
    ("schema_version", "s2"),
    ("target_fields", ["图号", "名称"]),
])
def test_a_change_to_any_identity_component_makes_new_work(tmp_path, field_name, value):
    from src.vision.review_queue import ReviewQueue
    queue = ReviewQueue(tmp_path / "q.json")
    queue.add(**_queue_kwargs())
    assert queue.add(**_queue_kwargs(**{field_name: value})) is True
    assert len(queue) == 2


def test_a_silent_miss_outranks_everything(tmp_path):
    """SYNTHETIC. High OCR confidence with low completeness is the case that
    reaches a dispatcher as a confident wrong answer."""
    from src.vision.review_queue import PRIORITY_SILENT_MISS, priority_for
    assert priority_for(trust_level="contextual",
                        reason_codes=["required_field_missing"],
                        ocr_confidence=0.9957, completeness=0.1667) \
        == PRIORITY_SILENT_MISS


def test_cell_spans_columns_alone_goes_last_but_not_when_paired(tmp_path):
    """SYNTHETIC. It was measured as a false positive 57 times; alongside a
    real reason it must not drag the item to the back."""
    from src.vision.review_queue import (
        PRIORITY_CELL_SPANS_COLUMNS, PRIORITY_INVALID_VALUE, priority_for,
    )
    assert priority_for(trust_level="precise",
                        reason_codes=["cell_assignment_uncertain"]) \
        == PRIORITY_CELL_SPANS_COLUMNS
    assert priority_for(trust_level="precise",
                        reason_codes=["cell_assignment_uncertain",
                                      "invalid_field_value"]) \
        == PRIORITY_INVALID_VALUE


def test_rebuilding_never_resets_someones_review_status(tmp_path):
    """SYNTHETIC."""
    from src.vision.review_queue import DONE, ReviewQueue
    path = tmp_path / "q.json"
    queue = ReviewQueue(path)
    queue.add(**_queue_kwargs())
    item_id = queue.all_items()[0].item_id
    queue.mark(item_id, DONE)
    queue.save()

    rebuilt = ReviewQueue(path)
    assert rebuilt.add(**_queue_kwargs()) is False
    assert rebuilt.all_items()[0].review_status == DONE
    assert rebuilt.pending() == []


def test_pending_is_ordered_by_priority(tmp_path):
    from src.vision.review_queue import ReviewQueue
    queue = ReviewQueue(tmp_path / "q.json")
    queue.add(**_queue_kwargs(image_sha256="c", trust_level="diagnostic",
                              strategy="full_page_diagnostic"))
    queue.add(**_queue_kwargs(image_sha256="a", trust_level="precise"))
    queue.add(**_queue_kwargs(image_sha256="b", trust_level="contextual"))
    assert [i.trust_level for i in queue.pending()] == [
        "precise", "contextual", "diagnostic"]


def test_the_queue_does_not_claim_to_be_a_service(tmp_path):
    from src.vision.review_queue import ReviewQueue
    queue = ReviewQueue(tmp_path / "q.json")
    queue.add(**_queue_kwargs())
    assert "not a message broker" in queue.summary()["note"]
