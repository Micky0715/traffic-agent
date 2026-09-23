from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from src.evaluator import ROOT
from src.vision.cache import VLMCache
from src.vision.config import load_config
from src.vision.dataset_split import load_dev_holdout
from src.vision.image_quality import ImageQualityAnalyzer, decide_preprocess
from src.vision.ocr_engine import OCREngine, get_ocr_engine
from src.vision.preprocess import apply_preprocess
from src.vision.qwen_vl_adapter import Qwen25VLAdapter
from src.vision.entity_ref import (
    align_entities, alignment_index, assignment_certainty_index,
    build_ocr_entities, build_vlm_entities,
)
from src.vision.evidence import (
    evidence_from_ocr, evidence_from_vlm_entities, merge_evidence,
)
from src.vision.field_completeness import evaluate_field_completeness
from src.vision.page_router import route_ocr_page
from src.vision.table_structure import detect_table_structure
from src.vision.value_validation import (
    field_pairs_from_completeness, normalize_value, validate_field_values,
)

DRAWINGS_DIR = ROOT / "data" / "drawings"


def _containment_score(gold_values: List[str], blob: str) -> float:
    if not gold_values:
        return 1.0
    hits = sum(1 for v in gold_values if v and v in blob)
    return hits / len(gold_values)


def _experiment_a(case: dict, ocr_engine: OCREngine) -> Dict:
    """Original -> OCR -> TableParser.

    Whether these numbers mean anything depends entirely on --ocr-engine.
    With the default `mock`, MockOCREngine has no pixel sensitivity, so this
    measures "what a perfect engine reading hand-authored stubs would
    produce", not real OCR accuracy. With `fixture` or `paddleocr` the OCR
    text is genuinely PaddleOCR's. The `real` flag below records which.""" 
    image_path = DRAWINGS_DIR / case["image"]
    ocr = ocr_engine.recognize(image_path)
    table = detect_table_structure(image_path, ocr_result=ocr)
    blob = ocr.text + " " + json.dumps(table.model_dump(mode="json"), ensure_ascii=False)
    return {"score": _containment_score(case["gold_field_values"], blob),
            "ocr_confidence": ocr.average_confidence, "table_confidence": table.confidence,
            "ocr_engine": ocr.engine, "real": ocr.engine != "mock"}


def _experiment_b(case: dict, cfg, ocr_engine: OCREngine) -> Dict:
    """Preprocess -> OCR -> TableParser. Preprocessing is real cv2 either way.

    Against `mock` this comparison with A is vacuous by construction:
    MockOCREngine resolves a *.processed.png path back to the same stub, so
    A and B are guaranteed identical. It only becomes a real A/B once
    --ocr-engine is fixture or paddleocr.""" 
    image_path = DRAWINGS_DIR / case["image"]
    analyzer = ImageQualityAnalyzer()
    metrics = analyzer.analyze(image_path)
    decision = decide_preprocess(metrics, cfg)
    if decision.need_preprocess:
        processed = apply_preprocess(image_path, decision.operations, skew_angle_deg=metrics.skew_angle_deg)
        target_path = Path(processed.processed_path)
    else:
        target_path = image_path
    ocr = ocr_engine.recognize(target_path)
    table = detect_table_structure(target_path, ocr_result=ocr)
    blob = ocr.text + " " + json.dumps(table.model_dump(mode="json"), ensure_ascii=False)
    return {"score": _containment_score(case["gold_field_values"], blob),
            "ocr_confidence": ocr.average_confidence, "table_confidence": table.confidence,
            "operations_applied": decision.operations,
            "ocr_engine": ocr.engine, "real": ocr.engine != "mock"}


def _experiment_c(case: dict, adapter: Qwen25VLAdapter) -> Dict:
    """Original -> VLM -> JSON. Real Qwen-VL call on the original image."""
    image_path = DRAWINGS_DIR / case["image"]
    metadata = adapter.extract_metadata(image_path)
    devices = adapter.extract_table(image_path)
    blob = metadata.model_dump_json() + " " + json.dumps([d.model_dump(mode="json") for d in devices], ensure_ascii=False)
    return {"score": _containment_score(case["gold_field_values"], blob), "real": True}


def _experiment_d(case: dict, cfg, adapter: Qwen25VLAdapter) -> Dict:
    """Preprocess -> VLM -> JSON. Real cv2 preprocessing AND a real VLM call
    on the resulting pixels — this is the one fully-real A/B-style
    comparison this round can run (paired with C)."""
    image_path = DRAWINGS_DIR / case["image"]
    analyzer = ImageQualityAnalyzer()
    metrics = analyzer.analyze(image_path)
    decision = decide_preprocess(metrics, cfg)
    if decision.need_preprocess:
        processed = apply_preprocess(image_path, decision.operations, skew_angle_deg=metrics.skew_angle_deg)
        target_path = Path(processed.processed_path)
    else:
        target_path = image_path
    metadata = adapter.extract_metadata(target_path)
    devices = adapter.extract_table(target_path)
    blob = metadata.model_dump_json() + " " + json.dumps([d.model_dump(mode="json") for d in devices], ensure_ascii=False)
    return {"score": _containment_score(case["gold_field_values"], blob),
            "operations_applied": decision.operations, "real": True}


def _experiment_e(case: dict, cfg, adapter: Qwen25VLAdapter, vlm_cache: Dict[str, Dict],
                  ocr_engine: OCREngine) -> Dict:
    """OCR + TableParser -> routing -> VLM fallback (when triggered) -> Validator.

    The routing step is the point of this experiment. It used to be a single
    confidence test plus a table-confidence test, which escalated 11 of 11
    pages and therefore proved nothing about tiering. It now runs the full
    silent-miss check in route_ocr_page.

    Per-case bookkeeping here feeds the routing metrics: which gold fields the
    OCR-only path found, which the VLM path recovered, and why the page was
    escalated. Gold values are read ONLY for scoring after the fact — the
    routing decision above never sees them.
    """
    image_path = DRAWINGS_DIR / case["image"]
    ocr = ocr_engine.recognize(image_path)
    table = detect_table_structure(image_path, ocr_result=ocr)
    completeness = evaluate_field_completeness(
        ocr, cfg.drawing_types, cfg.field_completeness,
        table=table, min_table_confidence=cfg.min_table_confidence,
    )
    validation = validate_field_values(
        field_pairs_from_completeness(completeness.field_pairs),
        cfg.value_validation, cfg.drawing_types.get(completeness.drawing_type),
    )
    decision = route_ocr_page(ocr, table, completeness, cfg, validation)

    gold = case["gold_field_values"]
    ocr_blob = ocr.text + " " + json.dumps(table.model_dump(mode="json"), ensure_ascii=False)
    ocr_only_found = [v for v in gold if v and v in ocr_blob]
    ocr_only_missed = [v for v in gold if v and v not in ocr_blob]

    type_spec = cfg.drawing_types.get(completeness.drawing_type)
    normalize = lambda v: normalize_value(
        v, cfg.value_validation.hyphen_variants,
        cfg.value_validation.collapse_spaces_around_hyphen)

    ocr_entities, pair_source_keys, _page_level = build_ocr_entities(
        completeness.field_pairs, cfg.entity_types,
        type_spec.primary_entity_type if type_spec else None, normalize)

    vlm_entities = []
    blob = ocr_blob
    source = "ocr_only"
    vlm_recovered: List[str] = []
    vlm_meta_json = "{}"
    if decision.need_vlm:
        if case["image"] not in vlm_cache:
            metadata = adapter.extract_metadata(image_path)
            devices = adapter.extract_table(image_path)
            vlm_cache[case["image"]] = {
                "metadata": metadata.model_dump_json(),
                "devices": json.dumps([d.model_dump(mode="json") for d in devices], ensure_ascii=False),
            }
        vlm_out = vlm_cache[case["image"]]
        vlm_meta_json = vlm_out["metadata"]
        vlm_blob = vlm_meta_json + " " + vlm_out["devices"]
        blob = ocr_blob + " " + vlm_blob
        vlm_recovered = [v for v in ocr_only_missed if v in vlm_blob]
        source = "vlm_fallback"
        vlm_entities = build_vlm_entities(
            json.loads(vlm_out["devices"]) or [], cfg.entity_types,
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
    structured = _structured_scores(gold, evidence_set, cfg)

    return {
        # Kept under its original definition and renamed to say what it is: a
        # gold string appearing ANYWHERE in the concatenated text of both
        # sources. It cannot distinguish "both sources read this correctly"
        # from "one read it correctly and the other read stamp noise", so it
        # is an optimistic ceiling, not evidence the system can safely answer.
        "legacy_containment_hit_rate": _containment_score(gold, blob),
        "structured_candidate_recall": structured["candidate_recall"],
        "conflict_free_exact_rate": structured["conflict_free_exact"],
        "decision_ready_exact_rate": structured["decision_ready_exact"],
        "ocr_only_score": _containment_score(gold, ocr_blob),
        "need_vlm": decision.need_vlm,
        "route_reasons": decision.reasons,
        "drawing_type": completeness.drawing_type,
        "type_source": completeness.type_source,
        "field_completeness": completeness.completeness,
        "isolated_labels": completeness.isolated_labels,
        "watermark_repetition": completeness.watermark_repetition,
        "group_cardinality_uncertain": completeness.group_cardinality_uncertain,
        "valid_ratio": validation.valid_ratio,
        "invalid_fields": validation.invalid_fields,
        "invalid_details": [d.model_dump(mode="json") for d in validation.invalid_details],
        "unknown_fields": validation.unknown_fields,
        "field_conflicts": [c.model_dump(mode="json") for c in evidence_set.conflicts],
        "entity_assignment_uncertain_count": evidence_set.entity_assignment_uncertain_count,
        "entity_alignments": [a.model_dump(mode="json") for a in alignments],
        "entity_mapping_methods": {
            e.source_key: {"raw_entity_name": e.raw_entity_name,
                           "extracted_entity_label": e.extracted_entity_label,
                           "mapping_method": e.mapping_method,
                           "entity_type": e.entity_type}
            for e in list(ocr_entities) + list(vlm_entities)
        },
        "evidence": [e.model_dump(mode="json") for e in evidence_set.evidence],
        "ocr_only_found": ocr_only_found,
        "ocr_only_missed": ocr_only_missed,
        "vlm_recovered": vlm_recovered,
        "source": source,
        "ocr_engine": ocr.engine,
        "ocr_confidence": ocr.average_confidence,
        "table_confidence": table.confidence,
        "real": source == "vlm_fallback",
        # `score` keeps the legacy definition so this experiment stays
        # comparable with A-D and with the previous round's report.
        "score": _containment_score(gold, blob),
    }


def _structured_scores(gold: List[str], evidence_set, cfg) -> Dict[str, float]:
    """Field-level counterparts to the legacy containment score.

    candidate_recall: a gold value is reachable from SOME structured candidate.
    Strictly better grounded than containment — the value is attached to a
    field and a source rather than found loose in concatenated text — but it
    still says nothing about whether the system could pick that candidate.

    conflict_free_exact: the stricter one. A gold value counts only if some
    (entity, field) whose sources do NOT disagree carries exactly that value.
    An unresolved conflict scores zero even when one of the two conflicting
    candidates is right, because this pipeline has no adjudication policy and
    therefore cannot claim it would have chosen correctly.

    decision_ready_exact: the only rate that answers "could the system hand
    this value to a person without anyone checking it first". On top of being
    exact and conflict-free, the candidate must also

      * know WHICH piece of equipment it describes — an entity-scoped reading
        whose entity never aligned is a motor number belonging to an unknown
        motor, and dispatching on it is how someone services the wrong machine
      * not have failed its own field-shape rule

    It is the lowest of the four rates by construction, and it is the one to
    quote when asked what the system can stand behind unaided.
    """
    conflicted = {(c.entity_id, c.field_name) for c in evidence_set.conflicts}
    norm = lambda v: normalize_value(
        v, cfg.value_validation.hyphen_variants,
        cfg.value_validation.collapse_spaces_around_hyphen)

    if not gold:
        return {"candidate_recall": 1.0, "conflict_free_exact": 1.0,
                "decision_ready_exact": 1.0}

    def conflict_free(evidence) -> bool:
        return (evidence.entity_ref, evidence.field_name) not in conflicted

    def decision_ready(evidence) -> bool:
        if not conflict_free(evidence):
            return False
        if evidence.validation_status == "invalid":
            return False
        # We must know WHICH piece of equipment the value describes. An
        # entity that aligned across sources qualifies; so does one only the
        # cheap path saw, when the router affirmatively decided no second
        # source was needed (see assignment_certainty_index).
        return not evidence.entity_assignment_uncertain

    recall_hits = exact_hits = ready_hits = 0
    for value in gold:
        target = norm(value)
        if any(target in e.normalized_value for e in evidence_set.evidence):
            recall_hits += 1
        if any(e.normalized_value == target and conflict_free(e)
               for e in evidence_set.evidence):
            exact_hits += 1
        if any(e.normalized_value == target and decision_ready(e)
               for e in evidence_set.evidence):
            ready_hits += 1
    return {"candidate_recall": recall_hits / len(gold),
            "conflict_free_exact": exact_hits / len(gold),
            "decision_ready_exact": ready_hits / len(gold)}


def _routing_metrics(results: List[Dict]) -> Dict:
    """The five routing numbers, with the proxy labels named as proxies.

    fallback_proxy_positive is NOT a gold label. A page is marked positive
    when the OCR-only path failed to produce every task-required field — which
    is derived from the same answer key the final hit rate is scored against,
    so recall and hit rate are not independent measurements and must never be
    presented as corroborating each other. A real label would need a human
    judging, before seeing any score, whether a page needed help.
    """
    if not results:
        return {}
    total = len(results)
    triggered = [r for r in results if r["need_vlm"]]
    positives = [r for r in results if r["ocr_only_score"] < 1.0]
    negatives = [r for r in results if r["ocr_only_score"] >= 1.0]

    missed_total = sum(len(r["ocr_only_missed"]) for r in triggered)
    recovered_total = sum(len(r["vlm_recovered"]) for r in triggered)

    return {
        "fallback_trigger_rate": len(triggered) / total,
        "fallback_proxy_positive_count": len(positives),
        "fallback_proxy_negative_count": len(negatives),
        "proxy_fallback_recall": (
            sum(1 for r in positives if r["need_vlm"]) / len(positives) if positives else None),
        "proxy_false_trigger_rate": (
            sum(1 for r in negatives if r["need_vlm"]) / len(negatives) if negatives else None),
        "vlm_field_recovery_rate": (recovered_total / missed_total if missed_total else None),
        "vlm_fields_missed_by_ocr": missed_total,
        "vlm_fields_recovered": recovered_total,
        # Optimistic ceiling: a gold string found anywhere in the two sources'
        # concatenated text. Blind to which source said it and to whether the
        # sources contradict each other. Never present this as "the system can
        # answer correctly this often".
        "legacy_containment_hit_rate": sum(r["legacy_containment_hit_rate"] for r in results) / total,
        "structured_candidate_recall": sum(r["structured_candidate_recall"] for r in results) / total,
        # The number to quote when asked what the system can actually stand
        # behind: unresolved conflicts count as failures.
        "conflict_free_exact_rate": sum(r["conflict_free_exact_rate"] for r in results) / total,
        # What the system could answer unaided: exact, conflict-free, entity
        # assignment known, and not failing its own field-shape rule.
        "decision_ready_exact_rate": sum(r["decision_ready_exact_rate"] for r in results) / total,
        "ocr_only_field_hit_rate": sum(r["ocr_only_score"] for r in results) / total,
        "field_conflict_count": sum(len(r["field_conflicts"]) for r in results),
        "invalid_value_count": sum(len(r["invalid_fields"]) for r in results),
        "entity_assignment_uncertain_count": sum(
            r["entity_assignment_uncertain_count"] for r in results),
        "entity_alignment_status_counts": _count_by(
            results, "entity_alignments", "status"),
        "entity_mapping_method_counts": _count_mapping_methods(results),
        # Genuine alignment failures only: both sources offered candidates and
        # none of the methods matched them.
        "entity_unresolved_count": sum(
            1 for r in results for a in r["entity_alignments"] if a["status"] == "unresolved"),
        "entity_single_source_count": sum(
            1 for r in results for a in r["entity_alignments"] if a["status"] == "single_source"),
        "entity_single_source_causes": _count_single_source_causes(results),
        "entity_aligned_count": sum(
            1 for r in results for a in r["entity_alignments"]
            if a["status"].startswith("aligned_")),
        "reason_counts": {
            reason: sum(1 for r in results if reason in r["route_reasons"])
            for reason in sorted({x for r in results for x in r["route_reasons"]})
        },
    }


def run_experiments(cases: List[dict], cfg, workers: int, ocr_engine: OCREngine) -> Dict[str, List[Dict]]:
    adapter = Qwen25VLAdapter(cache=VLMCache())  # real VLM calls are cached across A-E reuse
    vlm_cache_e: Dict[str, Dict] = {}
    results: Dict[str, List[Dict]] = {"A": [], "B": [], "C": [], "D": [], "E": []}
    lock = threading.Lock()

    def run_case(case: dict) -> Dict[str, Dict]:
        a = _experiment_a(case, ocr_engine)
        b = _experiment_b(case, cfg, ocr_engine)
        c = _experiment_c(case, adapter)
        d = _experiment_d(case, cfg, adapter)
        e = _experiment_e(case, cfg, adapter, vlm_cache_e, ocr_engine)
        return {"A": a, "B": b, "C": c, "D": d, "E": e}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, case): case for case in cases}
        for fut in as_completed(futures):
            case = futures[fut]
            out = fut.result()
            with lock:
                for exp, result in out.items():
                    result = {**result, "id": case["id"], "category": case["category"], "image": case["image"]}
                    results[exp].append(result)
                done += 1
                print(f"\r  {done}/{len(cases)}", end="", flush=True)
    print()
    return results


def _summarize(results: List[Dict]) -> Dict:
    if not results:
        return {"count": 0, "avg_score": 0.0, "by_category": {}}
    avg = sum(r["score"] for r in results) / len(results)
    by_category: Dict[str, Dict] = {}
    for r in results:
        cat = by_category.setdefault(r["category"], {"scores": []})
        cat["scores"].append(r["score"])
    by_category = {cat: {"avg_score": sum(v["scores"]) / len(v["scores"]), "count": len(v["scores"])}
                   for cat, v in by_category.items()}
    is_real = results[0].get("real")
    return {"count": len(results), "avg_score": avg, "by_category": by_category, "real": is_real}


def _count_single_source_causes(results: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        for a in r.get("entity_alignments", []):
            if a["status"] != "single_source":
                continue
            key = a.get("single_source_cause") or "unspecified"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _count_by(results: List[Dict], list_key: str, field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        for item in r.get(list_key, []):
            counts[item[field]] = counts.get(item[field], 0) + 1
    return dict(sorted(counts.items()))


def _count_mapping_methods(results: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        for entity in r.get("entity_mapping_methods", {}).values():
            key = f"{entity['mapping_method']}->{entity['entity_type']}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _routing_lines(routing: Dict[str, Dict]) -> List[str]:
    lines = ["", "### 路由指标（实验 E）", "",
             "> `fallback_proxy_positive` 是**派生代理标签**，不是金标：它由"
             "「纯 OCR 路径未取全任务所需字段」推导而来，和最终字段命中率"
             "**共用同一份答案**，因此召回率与命中率不独立，不能互相印证。"
             "真正的标签需要人在看到任何分数之前判断该页是否需要兜底。",
             "",
             "> 四个命中口径由松到紧：`legacy_containment_hit_rate`（gold 串出现在"
             "两源拼接文本中，最乐观，既不分来源也不管矛盾）→ "
             "`structured_candidate_recall`（存在结构化候选包含该值）→ "
             "`conflict_free_exact_rate`（存在无冲突候选精确等于该值）→ "
             "**`decision_ready_exact_rate`**（在此之上还要求实体归属已确定、"
             "且未违反字段值规则——即系统可在无人复核下直接作答）。"
             "对外引用请用最后一个，`legacy` 仅用于与历史报告对比。",
             "",
             "| 指标 | 开发集(7) | 回归验证集(4) | 合计(11) |",
             "|---|---|---|---|"]
    rows = [
        ("兜底触发率", "fallback_trigger_rate"),
        ("代理兜底召回率", "proxy_fallback_recall"),
        ("代理误触发率", "proxy_false_trigger_rate"),
        ("VLM 字段恢复率", "vlm_field_recovery_rate"),
        ("纯 OCR 字段命中率（字符串包含）", "ocr_only_field_hit_rate"),
        ("legacy_containment_hit_rate（乐观口径）", "legacy_containment_hit_rate"),
        ("structured_candidate_recall", "structured_candidate_recall"),
        ("conflict_free_exact_rate（保守口径）", "conflict_free_exact_rate"),
        ("decision_ready_exact_rate（可直接作答口径）", "decision_ready_exact_rate"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {_pct(routing['dev'].get(key))} | "
                     f"{_pct(routing['regression'].get(key))} | {_pct(routing['overall'].get(key))} |")
    for label, key in [("字段冲突数", "field_conflict_count"),
                       ("非法字段值数", "invalid_value_count"),
                       ("设备归属不确定数", "entity_assignment_uncertain_count"),
                       ("实体已对齐数", "entity_aligned_count"),
                       ("实体单边存在数(无对象可配)", "entity_single_source_count"),
                       ("实体对齐 unresolved 数(两侧有候选但配不上)", "entity_unresolved_count")]:
        lines.append(f"| {label} | {routing['dev'].get(key)} | "
                     f"{routing['regression'].get(key)} | {routing['overall'].get(key)} |")
    align_counts = routing["overall"].get("entity_alignment_status_counts", {})
    if align_counts:
        lines.append("")
        lines.append("实体对齐状态分布：" + "；".join(f"`{k}` × {v}" for k, v in align_counts.items()))
    cause_counts = routing["overall"].get("entity_single_source_causes", {})
    if cause_counts:
        lines.append("")
        lines.append("单边存在(single_source)原因分布：" +
                     "；".join(f"`{k}` × {v}" for k, v in cause_counts.items()))
    map_counts = routing["overall"].get("entity_mapping_method_counts", {})
    if map_counts:
        lines.append("")
        lines.append("实体映射方式分布：" + "；".join(f"`{k}` × {v}" for k, v in map_counts.items()))

    counts = routing["overall"].get("reason_counts", {})
    if counts:
        lines.append("")
        lines.append("触发原因分布（合计）：" + "；".join(f"`{k}` × {v}" for k, v in counts.items()))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR Pipeline A/B/C/D/E 真实实验（C/D 为真实 VLM 调用，A/B 为标注清楚的 Mock OCR）")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prefix", default="ocr_eval_")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ocr-engine", choices=["mock", "fixture", "paddleocr"], default="mock",
                        help="which OCR backs experiments A/B/E. mock (default) = hand-authored "
                             "stubs, no model runs; fixture = replay of a saved real PaddleOCR "
                             "run; paddleocr = live inference (~37s/image)")
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--holdout-only", action="store_true")
    args = parser.parse_args()

    dev_cases, holdout_cases = load_dev_holdout(Path(args.dataset) if args.dataset else None)
    if args.dev_only:
        holdout_cases = []
    if args.holdout_only:
        dev_cases = []
    cases = dev_cases + holdout_cases
    if args.limit:
        cases = cases[: args.limit]

    cfg = load_config()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    ocr_engine = get_ocr_engine(args.ocr_engine, enable_mkldnn=cfg.ocr.enable_mkldnn)
    workers = args.workers
    if args.ocr_engine == "paddleocr" and workers > 1:
        # One PaddleOCR predictor shared across threads is not safe, and the
        # models are far too heavy to instantiate per worker.
        print("--ocr-engine paddleocr: forcing --workers 1 (predictor is not thread-safe)")
        workers = 1

    print(f"运行 {len(cases)} 条用例（dev={len(dev_cases) if not args.limit else '—'}，"
          f"holdout={len(holdout_cases) if not args.limit else '—'}），并发数 {workers}，"
          f"OCR 引擎 {args.ocr_engine}...")
    started = time.time()
    results = run_experiments(cases, cfg, workers, ocr_engine)
    elapsed = time.time() - started

    dev_ids = {c["id"] for c in dev_cases}
    report = {"elapsed_seconds": round(elapsed, 1), "case_count": len(cases),
              "ocr_engine": args.ocr_engine, "experiments": {}}
    ocr_caveat = {
        "mock": ("> A/B/E 的 OCR 使用 MockOCREngine（手写桩，无真实像素感知能力，见 "
                 "data/ocr_stub/README.md），数字不代表真实 OCR 准确率；且 A 与 B 必然相同，"
                 "因为桩会把预处理图解析回同一份结果，这组对照在该引擎下是空的。"),
        "fixture": ("> A/B/E 的 OCR 来自 FixtureOCREngine——重放 data/ocr_fixtures/ 中"
                    "**已保存的真实 PaddleOCR 运行结果**（每份带图像 sha256、库/模型版本、设备、"
                    "生成时间）。内容是真实的，但**不是本轮实时推理**。"),
        "paddleocr": ("> A/B/E 的 OCR 是本轮**实时 PaddleOCR 推理**（oneDNN 已关闭，见 "
                      "PaddleOCREngine 类注释）。"),
    }[args.ocr_engine]

    lines = [
        "# OCR Pipeline A/B/C/D/E 实验报告",
        "",
        f"> 样本数：{len(cases)}，耗时：{elapsed:.1f}s，OCR 引擎：`{args.ocr_engine}`。",
        ocr_caveat,
        "> C/D 是真实 Qwen-VL 调用；D 额外套了真实 cv2 预处理。",
        "> E 的路由决策、VLM Fallback 调用和 Validator 校验部分始终是真实的。",
        "",
    ]
    eng = args.ocr_engine
    exp_desc = {
        "A": f"Original -> OCR({eng}) -> TableParser(real)",
        "B": f"Preprocess(real) -> OCR({eng}) -> TableParser(real)",
        "C": "Original -> VLM(real) -> JSON",
        "D": "Preprocess(real) -> VLM(real) -> JSON",
        "E": f"OCR({eng})+TableParser(real) -> QualityJudge routing(real) -> VLM Fallback(real when triggered) -> Validator(real)",
    }
    for exp in ["A", "B", "C", "D", "E"]:
        all_results = results[exp]
        dev_results = [r for r in all_results if r["id"] in dev_ids]
        holdout_results = [r for r in all_results if r["id"] not in dev_ids]
        summary = {
            "description": exp_desc[exp],
            "overall": _summarize(all_results),
            "dev": _summarize(dev_results),
            "holdout": _summarize(holdout_results),
            "results": all_results,
        }
        if exp == "E":
            summary["routing"] = {
                "overall": _routing_metrics(all_results),
                "dev": _routing_metrics(dev_results),
                "regression": _routing_metrics(holdout_results),
            }
        report["experiments"][exp] = summary
        lines.append(f"## 实验 {exp}：{exp_desc[exp]}")
        lines.append(f"- 整体平均得分：{summary['overall']['avg_score']:.1%}（n={summary['overall']['count']}）")
        if dev_results:
            lines.append(f"- 开发集平均得分：{summary['dev']['avg_score']:.1%}（n={summary['dev']['count']}）")
        if holdout_results:
            lines.append(f"- 回归验证集平均得分：{summary['holdout']['avg_score']:.1%}（n={summary['holdout']['count']}）")
        for cat, stat in summary["overall"]["by_category"].items():
            lines.append(f"  - {cat}: {stat['avg_score']:.1%}（n={stat['count']}）")
        if exp == "E":
            lines.extend(_routing_lines(summary["routing"]))
        lines.append("")

    (output_dir / f"{args.prefix}report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{args.prefix}summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
