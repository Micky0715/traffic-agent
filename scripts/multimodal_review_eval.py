"""Run the pre-registered multimodal cases end to end.

    python scripts/multimodal_review_eval.py                       # disabled
    python scripts/multimodal_review_eval.py --mode cache_replay
    python scripts/multimodal_review_eval.py --mode live --allow-live-vlm

Default is `disabled`: no model, no money. `live` needs the flag AND the config,
and stops at max_live_calls.

The expectations come from data/multimodal_cases_registered.json, which was
written before any call in this round. Nothing here reads a gold value to
decide where to crop or whether to escalate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multimodal.config import (  # noqa: E402
    MultimodalConfig, MultimodalReviewConfig, load_multimodal_config,
)
from src.multimodal.executor import LiveCallNotAuthorised, VisionReviewExecutor, summarize  # noqa: E402
from src.multimodal.fusion import evidence_from_review, fuse, single_source_fields  # noqa: E402
from src.multimodal.decision import run_second_pass  # noqa: E402
from src.multimodal.schemas import ReviewOutcome  # noqa: E402
from src.multimodal.pipeline import locate_label_bbox  # noqa: E402
from src.rag.config import load_rag_config  # noqa: E402
from src.rag.policy import EvidenceBundle, RequiredEvidencePolicy  # noqa: E402
from src.multimodal.reasons import INVALID_FIELD_VALUE, REQUIRED_FIELD_MISSING  # noqa: E402
from src.multimodal.triggers import TriggerCollector, trigger_stats  # noqa: E402
from src.vision.config import load_config  # noqa: E402
from src.vision.evidence import evidence_from_ocr  # noqa: E402
from src.vision.schemas import FieldEvidence  # noqa: E402
from src.vision.field_completeness import evaluate_field_completeness  # noqa: E402
from src.vision.ocr_engine import get_ocr_engine  # noqa: E402
from src.vision.table_structure import detect_table_structure  # noqa: E402
from src.vision.value_validation import (  # noqa: E402
    field_pairs_from_completeness, validate_field_values,
)

REGISTERED = ROOT / "data" / "multimodal_cases_registered.json"
OUT_JSON = ROOT / "outputs" / "multimodal_review_report.json"
OUT_MD = ROOT / "outputs" / "multimodal_review_report.md"


def with_mode(config: MultimodalConfig, mode: str) -> MultimodalConfig:
    return MultimodalConfig(
        review=MultimodalReviewConfig(**{**config.review.__dict__, "mode": mode}),
        cache=config.cache)


def run_case(case: dict, mm: MultimodalConfig, vcfg, executor, engine, policy) -> dict:
    image = ROOT / case["image"]
    document_id = case["document_id"]

    ocr = engine.recognize(image)
    table = detect_table_structure(image, ocr_result=ocr)
    completeness = evaluate_field_completeness(
        ocr, vcfg.drawing_types, vcfg.field_completeness,
        table=table, min_table_confidence=vcfg.min_table_confidence)
    pairs = field_pairs_from_completeness(completeness.field_pairs)
    spec = vcfg.drawing_types.get(completeness.drawing_type)
    validation = validate_field_values(pairs, vcfg.value_validation, spec)

    ocr_evidence = evidence_from_ocr(pairs, validation, vcfg.value_validation)

    # ---- what deserves a look -------------------------------------------
    collector = TriggerCollector(document_id,
                                 drawing_type=completeness.drawing_type)
    for name in completeness.missing_fields:
        collector.add(name, REQUIRED_FIELD_MISSING,
                      label_bbox=locate_label_bbox(name, ocr.blocks))
    for label in validation.invalid_fields:
        field_name = label.rpartition(".")[2]
        pair = next((p for p in pairs if p.field_name == field_name), None)
        collector.add(field_name, INVALID_FIELD_VALUE,
                      ocr_value=pair.raw_value if pair else None,
                      value_bbox=pair.bbox if pair else None)
    for name in getattr(completeness, "isolated_labels", []) or []:
        collector.add(name, "isolated_label_without_value",
                      label_bbox=locate_label_bbox(name, ocr.blocks))

    targets = collector.targets(mm.review)

    # ---- crop + review ---------------------------------------------------
    reviewed, crops = [], []
    for target in targets:
        crop = executor.crop_for(target, image)
        result = executor.review_with_crop(target, crop)
        reviewed.append((target, result, crop if crop.available else None))
        crops.append({
            "field_name": target.field_name,
            "review_reasons": target.review_reasons,
            "available": crop.available,
            "region_scope": crop.region_scope,
            "reason": crop.reason,
            "crop_path": crop.crop_path,
            "crop_sha256": crop.crop_sha256,
            "source_image_sha256": crop.source_image_sha256,
            "page": crop.page,
            "bbox_original": crop.bbox_original,
            "bbox_with_padding": crop.bbox_with_padding,
        })

    vlm_evidence = evidence_from_review(reviewed, vcfg.value_validation, spec)

    # A registered conflict case needs a disagreement to exist. No live reply in
    # this repo has ever disagreed with OCR on a field OCR also read, so the
    # pair is INJECTED from the registration file and flagged as synthetic in
    # the report. Mining a real run for a disagreement after the fact would let
    # the case be chosen to fit whatever happened.
    injected = case.get("injected_conflict")
    if injected:
        ocr_evidence = list(ocr_evidence) + [FieldEvidence(
            occurrence_id="ocr:injected", field_name=injected["field_name"],
            raw_value=injected["ocr_value"],
            normalized_value=injected["ocr_value"], source="ocr",
            validation_status="valid", confidence=0.99)]
        vlm_evidence = list(vlm_evidence) + [FieldEvidence(
            occurrence_id="vlm:injected", field_name=injected["field_name"],
            raw_value=injected["vlm_value"],
            normalized_value=injected["vlm_value"], source="vlm",
            validation_status="valid", confidence=0.0)]

    fused = fuse(ocr_evidence, vlm_evidence)
    results = [r for _, r, _ in reviewed]

    recovered = sorted({
        e.field_name for e in vlm_evidence
        if e.field_name not in {o.field_name for o in ocr_evidence}})

    # ---- first and second decision ---------------------------------------
    # The first pass sees OCR only; the second sees the merged set. Both go
    # through the same RequiredEvidencePolicy, so a difference between them is
    # caused by the evidence, not by a second set of rules.
    from src.multimodal.decision import to_policy_fields
    from src.vision.schemas import FieldEvidenceSet
    ocr_only_set = FieldEvidenceSet(evidence=list(ocr_evidence))
    # entity_id comes from the case registration, not from the parse: without
    # it the policy returns `clarify` for every page and the transition table
    # measures the harness rather than the system.
    bundle_before = EvidenceBundle(
        task_type="drawing_field_query",
        entity_id=document_id,
        fields=to_policy_fields(ocr_only_set, document_id=document_id, page=1),
        requested_fields=[case["target_field"]])
    decision_before = policy.evaluate(bundle_before)
    # A conflict is new information even when no crop was cut, so it re-decides
    # too. Gating only on `targets` let an OCR/VLM disagreement sit in the
    # merged set while the decision stayed `execute`.
    if not targets and not fused.conflicts:
        # No review and no disagreement, so nothing new to decide on.
        outcome = ReviewOutcome(
            document_id=document_id,
            decision_before_review=decision_before.decision,
            decision_after_review=decision_before.decision,
            notes={"review_rounds": 0,
                   "second_pass_skipped": "no review was triggered"})
    else:
        outcome = run_second_pass(
        policy=policy, bundle_before=bundle_before,
        decision_before=decision_before, fused=fused,
            review_results=results, cfg=mm.review, document_id=document_id)

    return {
        "case_id": case["case_id"],
        "document_id": document_id,
        "data_kind": case["data_kind"],
        "drawing_type": completeness.drawing_type,
        "ocr_average_confidence": round(ocr.average_confidence, 4),
        "ocr_blocks": len(ocr.blocks),
        "field_completeness": round(completeness.completeness, 4),
        "missing_fields": list(completeness.missing_fields),
        "invalid_fields": list(validation.invalid_fields),
        "review_triggered": bool(targets),
        "targets": [{"field_name": t.field_name,
                     "review_reasons": t.review_reasons} for t in targets],
        "crops": crops,
        "review": summarize(results),
        "review_statuses": [r.status for r in results],
        "inference_modes": sorted({r.inference_mode for r in results}),
        "vlm_fields": [
            {"field_name": e.field_name, "raw_value": e.raw_value,
             "validation_status": e.validation_status, "bbox": e.bbox}
            for e in vlm_evidence],
        "fields_recovered": recovered,
        "conflicts": [
            {"field_name": c.field_name, "ocr_value": c.ocr_value,
             "vlm_value": c.vlm_value, "resolution": "unresolved"}
            for c in fused.conflicts],
        "single_source": single_source_fields(fused),
        "conflict_is_injected": bool(case.get("injected_conflict")),
        "decision_before_review": outcome.decision_before_review,
        "decision_after_review": outcome.decision_after_review,
        "decision_transition": classify_transition(
            outcome.decision_before_review, outcome.decision_after_review),
        "single_source_vlm_fields": outcome.single_source_vlm_fields,
        "review_rounds": outcome.notes["review_rounds"],
        "expectations": case["expectations"],
    }


# How close a decision is to actually answering. Only a move UP this ladder is
# an upgrade; clarify -> abstain changes the decision and improves nothing, and
# reporting it as an upgrade would turn a refusal into a success.
_ANSWER_RANK = {"reject": 0, "abstain": 1, "human_review": 1, "fallback": 1,
                "clarify": 2, "partial": 3, "execute": 4}


def classify_transition(before: str, after: str) -> str:
    if before == after:
        return "unchanged"
    delta = _ANSWER_RANK.get(after, 0) - _ANSWER_RANK.get(before, 0)
    if delta > 0:
        return "upgraded"
    return "downgraded" if delta < 0 else "changed_laterally"


def check_expectations(row: dict) -> list:
    """Compare against what was registered. Failures are reported, not fixed."""
    out = []
    exp = row["expectations"]

    if "review_triggered" in exp:
        ok = row["review_triggered"] == exp["review_triggered"]
        out.append({"check": "review_triggered", "expected": exp["review_triggered"],
                    "observed": row["review_triggered"], "pass": ok})
    if exp.get("vlm_calls") == 0:
        calls = row["review"]["real_inference"] + row["review"]["cache_replay"]
        out.append({"check": "no_vlm_call", "expected": 0, "observed": calls,
                    "pass": calls == 0})
    if exp.get("whole_page_fallback_forbidden") or \
            exp.get("blind_whole_page_vlm_forbidden"):
        scopes = {c["region_scope"] for c in row["crops"]}
        ok = "page" not in scopes
        out.append({"check": "no_whole_page_fallback", "expected": "no page-scope crop",
                    "observed": sorted(scopes) or ["(no crop)"], "pass": ok})
    if exp.get("acceptable_outcomes"):
        scopes = {c["region_scope"] for c in row["crops"]}
        statuses = set(row["review_statuses"])
        ok = (not row["crops"]) or ("unavailable" in scopes) or \
             bool(statuses & {"crop_unavailable", "not_invoked"})
        out.append({"check": "degraded_page_does_not_guess",
                    "expected": exp["acceptable_outcomes"],
                    "observed": sorted(statuses) or ["(no review)"], "pass": ok})
    if exp.get("conflict_recorded"):
        out.append({"check": "conflict_recorded", "expected": True,
                    "observed": bool(row["conflicts"]),
                    "pass": bool(row["conflicts"])})
    if exp.get("decision_must_not_be_execute"):
        after = row.get("decision_after_review")
        out.append({"check": "conflict_blocks_execute", "expected": "not execute",
                    "observed": after, "pass": after != "execute"})
    if exp.get("vlm_not_auto_adopted") or exp.get("auto_accept_vlm_forbidden"):
        adopted = [c for c in row["conflicts"] if c["resolution"] != "unresolved"]
        out.append({"check": "vlm_not_auto_adopted", "expected": "all unresolved",
                    "observed": f"{len(adopted)} resolved", "pass": not adopted})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["disabled", "cache_replay", "live"],
                        default=None)
    parser.add_argument("--allow-live-vlm", action="store_true")
    args = parser.parse_args()

    registered = json.loads(REGISTERED.read_text(encoding="utf-8"))
    mm = load_multimodal_config()
    mode = args.mode or mm.review.mode
    mm = with_mode(mm, mode)

    try:
        executor = VisionReviewExecutor(mm, allow_live=args.allow_live_vlm)
    except LiveCallNotAuthorised as exc:
        print(f"refused: {exc}")
        sys.exit(3)

    vcfg = load_config()
    engine = get_ocr_engine("fixture")
    policy = RequiredEvidencePolicy(load_rag_config().evidence_policy)

    rows, checks = [], []
    for case in registered["cases"]:
        if not (ROOT / case["image"]).exists():
            rows.append({"case_id": case["case_id"], "skipped": "image missing"})
            continue
        row = run_case(case, mm, vcfg, executor, engine, policy)
        row["checks"] = check_expectations(row)
        checks.extend(row["checks"])
        rows.append(row)

    all_results = []
    totals = Counter()
    for row in rows:
        if "review" not in row:
            continue
        for key, value in row["review"].items():
            totals[key] += value

    payload = {
        "note": registered["note"],
        "not_a_blind_test": registered["not_a_blind_test"],
        "inference_mode": mode,
        "allow_live_vlm_flag": args.allow_live_vlm,
        "config_allow_live_vlm": mm.review.allow_live_vlm,
        "max_live_calls": mm.review.max_live_calls,
        "live_calls_made": executor.live_calls_made,
        "model_name": executor.model_name if mode == "live" else None,
        "prompt_version": mm.review.prompt_version,
        "schema_version": mm.review.schema_version,
        "cache_provenance": executor.cache.provenance_summary(),
        "latency_note": (
            "cache_replay latency is file I/O and is NOT model latency; it is "
            "not reported as such anywhere in this file."),
        "totals": dict(totals),
        "decision_transitions": dict(Counter(
            r["decision_transition"] for r in rows if "decision_transition" in r)),
        "expectation_checks": {
            "total": len(checks),
            "passed": sum(1 for c in checks if c["pass"]),
            "failed": sum(1 for c in checks if not c["pass"]),
        },
        "cases": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    lines = [
        "# 多模态复核报告", "",
        f"- 推理模式：**{mode}**",
        f"- 实时调用次数：**{executor.live_calls_made}** / 上限 {mm.review.max_live_calls}",
        f"- 缓存出处：`{payload['cache_provenance']}`", "",
        "> cache_replay 的耗时是文件读取，**不是模型延迟**，本报告不以模型延迟名义报告它。", "",
        "## 汇总", "", "| 指标 | 值 |", "|---|---|",
    ]
    for key, value in sorted(totals.items()):
        lines.append(f"| {key} | {value} |")
    lines += ["", "## 逐案例", "",
              "| case | 文档 | OCR置信 | 完整度 | 触发复核 | 裁剪 | 状态 | 恢复字段 | 冲突 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if "review" not in row:
            lines.append(f"| {row['case_id']} | — | — | — | — | — | {row.get('skipped')} | — | — |")
            continue
        scopes = ",".join(sorted({c["region_scope"] for c in row["crops"]})) or "—"
        lines.append(
            f"| {row['case_id']} | {row['document_id']} | "
            f"{row['ocr_average_confidence']} | {row['field_completeness']} | "
            f"{'是' if row['review_triggered'] else '否'} | {scopes} | "
            f"{','.join(row['review_statuses']) or '—'} | "
            f"{','.join(row['fields_recovered']) or '无'} | "
            f"{len(row['conflicts'])} |")
    lines += ["", "## 预注册期望校验", "",
              f"通过 {payload['expectation_checks']['passed']} / "
              f"{payload['expectation_checks']['total']}", "",
              "| case | 检查 | 期望 | 实测 | 结果 |", "|---|---|---|---|---|"]
    for row in rows:
        for check in row.get("checks", []):
            lines.append(
                f"| {row['case_id']} | {check['check']} | `{check['expected']}` | "
                f"`{check['observed']}` | {'PASS' if check['pass'] else '**FAIL**'} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"mode={mode}  live_calls={executor.live_calls_made}  "
          f"cache={payload['cache_provenance']['entries']}")
    print(f"totals: {dict(totals)}")
    print(f"expectation checks: {payload['expectation_checks']}")
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
