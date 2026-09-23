"""Run the FROZEN parsing pipeline once over the adversarial stress set.

Imports the frozen modules and changes none of them. Verifies the freeze
hashes before doing anything and aborts on mismatch.

Live inference on both stages: real PaddleOCR (these images have no fixtures)
and real VLM calls (these images are not in the VLM cache), so the latency and
call-count numbers here are real rather than replayed.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

EXPECTED_CONFIG_SHA = "b6f5f1c4bf17cfe697d63eb868f2a72f56aa1d9fccbf4b92c8b60c67b3be6bf4"
EXPECTED_SRC_SHA = "4fee535f3d1b20da43fae3e169ad92ca06d52ffa4f644a71515a9443efea73dc"

WARNING = ("规则作者与压力测试设计者相同，本测试只能用于证伪和发现边界，"
           "不能证明泛化能力。")


def verify_freeze() -> dict:
    config_sha = hashlib.sha256((ROOT / "configs/visual_parser.yaml").read_bytes()).hexdigest()
    digest = hashlib.sha256()
    files = sorted((ROOT / "src").rglob("*.py"))
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    src_sha = digest.hexdigest()
    ok = config_sha == EXPECTED_CONFIG_SHA and src_sha == EXPECTED_SRC_SHA
    if not ok:
        print("FREEZE VIOLATION — aborting")
        print(f"  config {config_sha} expected {EXPECTED_CONFIG_SHA}")
        print(f"  src    {src_sha} expected {EXPECTED_SRC_SHA}")
        sys.exit(2)
    return {"config_sha256": config_sha, "src_sha256": src_sha, "src_file_count": len(files)}


freeze = verify_freeze()
print("freeze verified")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)

from src.vision.base import VisionModelError  # noqa: E402
from src.vision.cache import VLMCache  # noqa: E402
from src.vision.config import load_config  # noqa: E402
from src.vision.entity_ref import (  # noqa: E402
    align_entities, alignment_index, assignment_certainty_index,
    build_ocr_entities, build_vlm_entities,
)
from src.vision.evidence import (  # noqa: E402
    evidence_from_ocr, evidence_from_vlm_entities, merge_evidence,
)
from src.vision.field_completeness import evaluate_field_completeness  # noqa: E402
from src.vision.ocr_engine import PaddleOCREngine  # noqa: E402
from src.vision.page_router import route_ocr_page  # noqa: E402
from src.vision.qwen_vl_adapter import Qwen25VLAdapter  # noqa: E402
from src.vision.table_structure import detect_table_structure  # noqa: E402
from src.vision.value_validation import (  # noqa: E402
    field_pairs_from_completeness, normalize_value, validate_field_values,
)

ADV_DIR = ROOT / "data" / "adversarial"
GOLD_DIR = ADV_DIR / "gold"
OUT_JSON = ROOT / "outputs" / "ocr_adversarial_stress_report.json"

cfg = load_config()
normalize = lambda v: normalize_value(
    v, cfg.value_validation.hyphen_variants,
    cfg.value_validation.collapse_spaces_around_hyphen)

engine = PaddleOCREngine(enable_mkldnn=cfg.ocr.enable_mkldnn)
# Fresh cache file so these runs are genuinely live and cannot be served from
# the regression set's cached responses.
adapter = Qwen25VLAdapter(cache=VLMCache(path=ADV_DIR / ".vlm_cache_adversarial.json"))

golds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(GOLD_DIR.glob("*.json"))]
rows = []
ocr_latencies: list[float] = []
vlm_latencies: list[float] = []
vlm_calls = 0
vlm_failures = 0

for gold in golds:
    image = ADV_DIR / f"{gold['id']}.png"

    started = time.perf_counter()
    ocr = engine.recognize(image)
    ocr_seconds = time.perf_counter() - started
    ocr_latencies.append(ocr_seconds)

    table = detect_table_structure(image, ocr_result=ocr)
    completeness = evaluate_field_completeness(
        ocr, cfg.drawing_types, cfg.field_completeness,
        table=table, min_table_confidence=cfg.min_table_confidence)
    validation = validate_field_values(
        field_pairs_from_completeness(completeness.field_pairs),
        cfg.value_validation, cfg.drawing_types.get(completeness.drawing_type))
    decision = route_ocr_page(ocr, table, completeness, cfg, validation)

    type_spec = cfg.drawing_types.get(completeness.drawing_type)
    ocr_entities, pair_source_keys, _page_level = build_ocr_entities(
        completeness.field_pairs, cfg.entity_types,
        type_spec.primary_entity_type if type_spec else None, normalize)

    ocr_blob = ocr.text + " " + json.dumps(table.model_dump(mode="json"), ensure_ascii=False)
    expected = gold["expected_values"]
    ocr_only_found = [v for v in expected if v and v in ocr_blob]
    ocr_only_missed = [v for v in expected if v and v not in ocr_blob]

    vlm_entities = []
    vlm_meta_json = "{}"
    vlm_seconds = 0.0
    vlm_error = None
    blob = ocr_blob
    vlm_recovered: list[str] = []

    if decision.need_vlm:
        started = time.perf_counter()
        try:
            metadata = adapter.extract_metadata(image)
            devices = adapter.extract_table(image)
            vlm_meta_json = metadata.model_dump_json()
            devices_payload = [d.model_dump(mode="json") for d in devices]
            vlm_calls += 2
        except (VisionModelError, Exception) as exc:  # noqa: BLE001 - recorded, never hidden
            vlm_error = f"{type(exc).__name__}: {exc}"[:300]
            vlm_failures += 1
            devices_payload = []
        vlm_seconds = time.perf_counter() - started
        vlm_latencies.append(vlm_seconds)

        vlm_blob = vlm_meta_json + " " + json.dumps(devices_payload, ensure_ascii=False)
        blob = ocr_blob + " " + vlm_blob
        vlm_recovered = [v for v in ocr_only_missed if v in vlm_blob]
        vlm_entities = build_vlm_entities(
            devices_payload, cfg.entity_types,
            list(cfg.value_validation.field_value_rules),
            cfg.vlm_field_mapping.parameter_names, normalize)

    alignments = align_entities(
        ocr_entities, vlm_entities, cfg.entity_types, vlm_invoked=decision.need_vlm)
    ref_index = alignment_index(alignments)
    certain_keys = assignment_certainty_index(alignments)

    ocr_evidence = evidence_from_ocr(
        completeness.field_pairs, validation, cfg.value_validation,
        pair_source_keys=pair_source_keys, entity_ref_index=ref_index,
        certain_keys=certain_keys)
    vlm_evidence = evidence_from_vlm_entities(
        vlm_meta_json, vlm_entities, cfg.vlm_field_mapping, cfg.value_validation,
        entity_ref_index=ref_index, certain_keys=certain_keys) if decision.need_vlm else []
    evidence_set = merge_evidence(ocr_evidence, vlm_evidence, alignments)

    conflicted = {(c.entity_ref, c.field_name) for c in evidence_set.conflicts}

    def conflict_free(ev):
        return (ev.entity_ref, ev.field_name) not in conflicted

    def decision_ready(ev):
        return (conflict_free(ev) and ev.validation_status != "invalid"
                and not ev.entity_assignment_uncertain)

    def rate(predicate) -> float:
        if not expected:
            return 1.0
        return sum(1 for v in expected if predicate(normalize(v))) / len(expected)

    legacy = sum(1 for v in expected if v in blob) / len(expected) if expected else 1.0
    candidate_recall = rate(
        lambda t: any(t in e.normalized_value for e in evidence_set.evidence))
    conflict_free_exact = rate(
        lambda t: any(e.normalized_value == t and conflict_free(e)
                      for e in evidence_set.evidence))
    ready_exact = rate(
        lambda t: any(e.normalized_value == t and decision_ready(e)
                      for e in evidence_set.evidence))

    answerable = [e for e in evidence_set.evidence if decision_ready(e)]

    rows.append({
        "id": gold["id"],
        "difficulty_type": gold["difficulty_type"],
        "human_needs_fallback": gold["human_needs_fallback"],
        "notes": gold["notes"],
        "gold_drawing_type": gold["drawing_type"],
        "gold_expected_values": expected,
        "gold_entities": gold["entities"],

        "ocr_engine": ocr.engine,
        "ocr_available": ocr.engine_available,
        "ocr_error": ocr.error,
        "ocr_confidence": ocr.average_confidence,
        "ocr_block_count": len(ocr.blocks),
        "ocr_text": ocr.text,
        "ocr_seconds": round(ocr_seconds, 2),

        "detected_drawing_type": completeness.drawing_type,
        "type_source": completeness.type_source,
        "field_completeness": completeness.completeness,
        "missing_fields": completeness.missing_fields,
        "isolated_labels": completeness.isolated_labels,
        "watermark_repetition": completeness.watermark_repetition,
        "group_device_count": completeness.group_device_count,
        "group_cardinality_uncertain": completeness.group_cardinality_uncertain,
        "field_pairs": [p.model_dump(mode="json") for p in completeness.field_pairs],

        "valid_ratio": validation.valid_ratio,
        "checked_fields": validation.checked_fields,
        "invalid_fields": validation.invalid_fields,
        "unknown_fields": validation.unknown_fields,
        "invalid_details": [d.model_dump(mode="json") for d in validation.invalid_details],

        "need_vlm": decision.need_vlm,
        "route_reasons": decision.reasons,
        "vlm_seconds": round(vlm_seconds, 2),
        "vlm_error": vlm_error,
        "vlm_entity_names": [e.raw_entity_name for e in vlm_entities],
        "vlm_entity_types": [e.entity_type for e in vlm_entities],
        "vlm_mapping_methods": [e.mapping_method for e in vlm_entities],

        "alignments": [a.model_dump(mode="json") for a in alignments],
        "conflicts": [c.model_dump(mode="json") for c in evidence_set.conflicts],
        "evidence": [e.model_dump(mode="json") for e in evidence_set.evidence],
        "entity_assignment_uncertain_count": evidence_set.entity_assignment_uncertain_count,

        "ocr_only_found": ocr_only_found,
        "ocr_only_missed": ocr_only_missed,
        "vlm_recovered": vlm_recovered,

        "legacy_containment_hit_rate": legacy,
        "structured_candidate_recall": candidate_recall,
        "conflict_free_exact_rate": conflict_free_exact,
        "policy_decision_ready_exact_rate": ready_exact,
        "answerable_field_count": len(answerable),
        "abstained_field_count": len(evidence_set.evidence) - len(answerable),
    })
    print(f"  {gold['id']} {gold['difficulty_type']:<24} ocr={ocr.average_confidence:.3f} "
          f"vlm={'Y' if decision.need_vlm else 'n'} ready={ready_exact:.2f} "
          f"{ocr_seconds:.0f}s+{vlm_seconds:.0f}s", flush=True)


def mean(key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def pct(values) -> dict:
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))], 2),
        "mean": round(sum(ordered) / len(ordered), 2),
    }


triggered = [r for r in rows if r["need_vlm"]]
needs = [r for r in rows if r["human_needs_fallback"]]
not_needs = [r for r in rows if not r["human_needs_fallback"]]
all_alignments = [a for r in rows for a in r["alignments"]]
all_vlm_types = [t for r in rows for t in r["vlm_entity_types"]]
total_evidence = sum(len(r["evidence"]) for r in rows)
total_abstained = sum(r["abstained_field_count"] for r in rows)
missed_total = sum(len(r["ocr_only_missed"]) for r in triggered)
recovered_total = sum(len(r["vlm_recovered"]) for r in triggered)

summary = {
    "warning": WARNING,
    "freeze": freeze,
    "case_count": len(rows),
    "legacy_containment_hit_rate": mean("legacy_containment_hit_rate"),
    "structured_candidate_recall": mean("structured_candidate_recall"),
    "conflict_free_exact_rate": mean("conflict_free_exact_rate"),
    "policy_decision_ready_exact_rate": mean("policy_decision_ready_exact_rate"),
    "fallback_trigger_rate": len(triggered) / len(rows) if rows else 0.0,
    "fallback_recall_vs_human_label": (
        sum(1 for r in needs if r["need_vlm"]) / len(needs) if needs else None),
    "false_trigger_rate_vs_human_label": (
        sum(1 for r in not_needs if r["need_vlm"]) / len(not_needs) if not_needs else None),
    "vlm_field_recovery_rate": recovered_total / missed_total if missed_total else None,
    "vlm_fields_missed_by_ocr": missed_total,
    "vlm_fields_recovered": recovered_total,
    "entity_mapping_unknown_rate": (
        sum(1 for t in all_vlm_types if t == "unknown") / len(all_vlm_types)
        if all_vlm_types else None),
    "entity_alignment_unresolved_rate": (
        sum(1 for a in all_alignments if a["status"] == "unresolved") / len(all_alignments)
        if all_alignments else None),
    "conflict_detection_rate": (
        sum(1 for r in rows if r["conflicts"]) / len(rows) if rows else 0.0),
    "conflict_count": sum(len(r["conflicts"]) for r in rows),
    "abstention_rate": total_abstained / total_evidence if total_evidence else None,
    "ocr_latency_seconds": pct(ocr_latencies),
    "vlm_latency_seconds": pct(vlm_latencies),
    "ocr_calls": len(rows),
    "vlm_calls": vlm_calls,
    "vlm_failures": vlm_failures,
    "alignment_status_counts": dict(Counter(a["status"] for a in all_alignments)),
    "single_source_causes": dict(Counter(
        a["single_source_cause"] for a in all_alignments
        if a["status"] == "single_source")),
    "mapping_method_counts": dict(Counter(
        m for r in rows for m in r["vlm_mapping_methods"])),
    "detected_type_counts": dict(Counter(r["detected_drawing_type"] for r in rows)),
    "route_reason_counts": dict(Counter(
        reason for r in rows for reason in r["route_reasons"])),
    "by_difficulty": {},
}
for difficulty in sorted({r["difficulty_type"] for r in rows}):
    subset = [r for r in rows if r["difficulty_type"] == difficulty]
    summary["by_difficulty"][difficulty] = {
        "count": len(subset),
        "legacy": sum(r["legacy_containment_hit_rate"] for r in subset) / len(subset),
        "structured": sum(r["structured_candidate_recall"] for r in subset) / len(subset),
        "decision_ready": sum(r["policy_decision_ready_exact_rate"] for r in subset) / len(subset),
        "trigger_rate": sum(1 for r in subset if r["need_vlm"]) / len(subset),
    }

OUT_JSON.write_text(json.dumps({"summary": summary, "cases": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print()
print(f"saved -> {OUT_JSON}")
