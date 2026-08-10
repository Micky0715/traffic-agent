from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from src.evaluator import ROOT
from src.vision.cache import VLMCache
from src.vision.config import load_config
from src.vision.qwen_vl_adapter import Qwen25VLAdapter
from src.vision.schemas import DeviceEntity, DrawingMetadata, VisualRelation
from src.vision.validator import VisualParseValidator


def load_cases(path: str | None = None) -> List[dict]:
    case_path = Path(path) if path else (ROOT / "data" / "vision_fallback_cases.jsonl")
    rows = []
    with case_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_metadata(case: dict, pred: DrawingMetadata) -> tuple[bool, dict]:
    gold = case["gold_metadata"]
    matches = {}
    for key, expected in gold.items():
        got = getattr(pred, key)
        matches[key] = {"ok": got == expected, "expected": expected, "got": got}
    ok = all(m["ok"] for m in matches.values())
    return ok, matches


def score_table(case: dict, pred_devices: List[DeviceEntity]) -> tuple[bool, dict]:
    by_id = {d.device_id: d for d in pred_devices}
    detail = {}
    ok = True
    for gold_dev in case["gold_devices"]:
        did = gold_dev["device_id"]
        dev = by_id.get(did)
        if dev is None:
            detail[did] = {"found": False}
            ok = False
            continue
        raw_texts = [p.raw_text for p in dev.parameters]
        missing = [t for t in gold_dev["expected_raw_texts"] if not any(t in rt or rt in t for rt in raw_texts)]
        detail[did] = {"found": True, "missing": missing, "raw_texts": raw_texts}
        if missing:
            ok = False
    return ok, detail


def _id_matches(expected: str, predicted: str) -> bool:
    """The VLM may describe an entity as 'A16风机' or '电机 M-13' instead of
    the bare code used in a drawing's table cells ('A16', 'M-13'). Relation
    identification is being scored, not exact ID string formatting, so
    substring containment (either direction) counts as a match.
    """
    e, p = expected.strip(), predicted.strip()
    return e == p or e in p or p in e


def score_relations(case: dict, pred_relations: List[VisualRelation]) -> tuple[bool, dict]:
    def pair_matches(gold_pair, r):
        return (_id_matches(gold_pair["source_id"], r.source_id) and _id_matches(gold_pair["target_id"], r.target_id)) or \
               (_id_matches(gold_pair["source_id"], r.target_id) and _id_matches(gold_pair["target_id"], r.source_id))

    missing_gold = [g for g in case.get("gold_relations", []) if not any(pair_matches(g, r) for r in pred_relations)]

    forbidden_ids = case.get("forbidden_ids", [])
    forbidden_id_hit = [
        r.model_dump(mode="json") for r in pred_relations
        if any(_id_matches(fid, r.source_id) or _id_matches(fid, r.target_id) for fid in forbidden_ids)
    ]

    # forbidden_pairs: specific (source, target) combinations that must never
    # appear, even when both IDs legitimately appear in OTHER real relations
    # (e.g. two independent connected pairs in the same drawing — A joining B
    # is fine, A joining C is not, even though A, B, and C are all real IDs).
    forbidden_pairs = case.get("forbidden_pairs", [])
    forbidden_pair_hit = [
        r.model_dump(mode="json") for r in pred_relations
        if any(pair_matches({"source_id": fp[0], "target_id": fp[1]}, r) for fp in forbidden_pairs)
    ]

    forbidden_hit = forbidden_id_hit + forbidden_pair_hit
    ok = not missing_gold and not forbidden_hit
    return ok, {
        "missing_gold": missing_gold,
        "forbidden_hit": forbidden_hit,
        "predicted": [r.model_dump(mode="json") for r in pred_relations],
    }


def run_validator_integration(drawings_dir: Path, cfg) -> dict:
    """A real, non-synthetic Validator integration check: extract FAN-MULTI-01
    for real, then run each device through VisualParseValidator with stub OCR
    hints designed to exercise all three cross-check branches (confirmed via
    ledger match, suspected with no ledger match, conflict on disagreement).
    """
    adapter = Qwen25VLAdapter()
    validator = VisualParseValidator(cfg)
    image_path = drawings_dir / "FAN-MULTI-01.png"
    devices = adapter.extract_table(image_path)

    from src.vision.schemas import DrawingParseResult
    scenarios = []
    for device, ocr_hint, label in [
        (next((d for d in devices if d.device_id == "A16"), None), "A16", "matching_ocr_ledger_hit"),
        (next((d for d in devices if d.device_id == "A17"), None), "A1?", "mismatched_ocr_ledger_hit"),
        (next((d for d in devices if d.device_id == "A18"), None), None, "no_ocr_no_ledger"),
    ]:
        if device is None:
            scenarios.append({"label": label, "error": "device not found in real extraction"})
            continue
        result = DrawingParseResult(document_id="FAN-MULTI-01", page_number=1, devices=[device])
        validated = validator.validate(result, ocr_device_id_hint=ocr_hint)
        scenarios.append({
            "label": label, "device_id": device.device_id, "ocr_hint": ocr_hint,
            "status": validated.validation.status, "warnings": validated.validation.warnings,
        })
    return {"scenarios": scenarios}


def _run_one_case(adapter: Qwen25VLAdapter, drawings_dir: Path, case: dict) -> dict:
    image_path = drawings_dir / case["image"]
    method = case["method"]
    try:
        if method == "metadata":
            pred = adapter.extract_metadata(image_path, case.get("hint", ""))
            ok, detail = score_metadata(case, pred)
            raw = pred.model_dump(mode="json")
        elif method == "table":
            pred = adapter.extract_table(image_path, case.get("hint", ""))
            ok, detail = score_table(case, pred)
            raw = [d.model_dump(mode="json") for d in pred]
        elif method == "relations":
            pred = adapter.extract_relations(image_path, case.get("hint", ""))
            ok, detail = score_relations(case, pred)
            raw = [r.model_dump(mode="json") for r in pred]
        else:
            raise ValueError(f"unknown method: {method}")
        error = None
    except Exception as e:  # noqa: BLE001 - a failed VLM call is itself a real result to record
        ok, detail, raw, error = False, {}, None, str(e)

    return {
        "id": case["id"], "image": case["image"], "method": method,
        "description": case["description"], "tags": case["tags"],
        "success": ok, "detail": detail, "raw_prediction": raw, "error": error,
    }


def run_concurrent(adapter: Qwen25VLAdapter, drawings_dir: Path, cases: List[dict], workers: int) -> List[dict]:
    """Cases are independent (different images/methods), so a shared cache
    keeps repeated (image, prompt, model) calls across cases free even under
    concurrency; VLMCache writes are append-only per key and not contended.
    """
    results_by_id: dict[str, dict] = {}
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one_case, adapter, drawings_dir, case): case for case in cases}
        for fut in as_completed(futures):
            case = futures[fut]
            result = fut.result()
            with lock:
                results_by_id[case["id"]] = result
                done += 1
                print(f"\r  {done}/{len(cases)}", end="", flush=True)
    print()
    return [results_by_id[c["id"]] for c in cases]


def run_stability_check(drawings_dir: Path, image_name: str, repeats: int) -> dict:
    """Repeat the same extract_metadata call on the same image with the cache
    disabled, to see whether a temperature=0 call is actually deterministic
    in practice, not just assumed to be. Every repeat is a fresh real call.
    """
    adapter = Qwen25VLAdapter(cache=VLMCache(enabled=False))
    image_path = drawings_dir / image_name
    runs = []
    for _ in range(repeats):
        pred = adapter.extract_metadata(image_path)
        runs.append(pred.model_dump(mode="json"))
    all_identical = all(r == runs[0] for r in runs)
    return {"image": image_name, "repeats": repeats, "all_identical": all_identical, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description="用真实 Qwen-VL 跑视觉补充解析（PageRouter 之后的 VLM Fallback 切片）评测")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prefix", default="vision_fallback_")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5, help="并发线程数")
    parser.add_argument("--skip-validator-integration", action="store_true")
    parser.add_argument("--skip-stability-check", action="store_true")
    parser.add_argument("--stability-repeats", type=int, default=3)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    drawings_dir = ROOT / "data" / "drawings"
    cfg = load_config()
    adapter = Qwen25VLAdapter()

    results = run_concurrent(adapter, drawings_dir, cases, args.workers)

    n = len(cases)
    success_count = sum(1 for r in results if r["success"])
    by_method = {}
    for r in results:
        m = by_method.setdefault(r["method"], {"total": 0, "success": 0})
        m["total"] += 1
        m["success"] += int(r["success"])

    validator_report = None
    if not args.skip_validator_integration:
        print("[validator integration] 用真实 FAN-MULTI-01 抽取结果跑 Validator 三分支...")
        validator_report = run_validator_integration(drawings_dir, cfg)

    stability_report = None
    if not args.skip_stability_check:
        print(f"[stability check] 对同一张图重复 {args.stability_repeats} 次真实调用（不走缓存）...")
        stability_report = run_stability_check(drawings_dir, "FAN-A13-02.png", args.stability_repeats)

    report = {
        "case_count": n,
        "case_success_rate": success_count / n,
        "by_method": {k: {"total": v["total"], "success": v["success"], "rate": v["success"] / v["total"]}
                      for k, v in by_method.items()},
        "results": results,
        "validator_integration": validator_report,
        "stability_check": stability_report,
    }
    (output_dir / f"{args.prefix}report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 真实 VLM Fallback 切片评测报告",
        "",
        f"> 数据集：{args.dataset or 'data/vision_fallback_cases.jsonl'}，样本数：{n}，"
        "图纸为脚本合成（无真实企业图纸）。",
        "",
        f"- 用例整体成功率：{report['case_success_rate']:.1%}",
    ]
    for method, stat in report["by_method"].items():
        lines.append(f"- {method} 成功率：{stat['rate']:.1%}（{stat['success']}/{stat['total']}）")
    if validator_report:
        lines.append("")
        lines.append("## Validator 集成结果（真实 FAN-MULTI-01 抽取结果 + stub OCR）")
        for s in validator_report["scenarios"]:
            lines.append(f"- {s}")
    if stability_report:
        lines.append("")
        lines.append(f"## 重复调用稳定性（{stability_report['image']}，重复 {stability_report['repeats']} 次，不走缓存）")
        lines.append(f"- 每次结果完全一致：{stability_report['all_identical']}")
    lines.append("")
    (output_dir / f"{args.prefix}summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
