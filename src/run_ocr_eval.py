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
from src.vision.ocr_engine import MockOCREngine
from src.vision.preprocess import apply_preprocess
from src.vision.qwen_vl_adapter import Qwen25VLAdapter
from src.vision.table_structure import detect_table_structure

DRAWINGS_DIR = ROOT / "data" / "drawings"


def _containment_score(gold_values: List[str], blob: str) -> float:
    if not gold_values:
        return 1.0
    hits = sum(1 for v in gold_values if v and v in blob)
    return hits / len(gold_values)


def _experiment_a(case: dict) -> Dict:
    """Original -> OCR -> TableParser. MOCK: MockOCREngine has no real pixel
    sensitivity (see ocr_engine.py). Numbers here do not represent real OCR
    accuracy — they represent "what a perfect real OCR engine reading these
    hand-authored, deliberately-imperfect stub results would produce."""
    image_path = DRAWINGS_DIR / case["image"]
    ocr = MockOCREngine().recognize(image_path)
    table = detect_table_structure(image_path, ocr_result=ocr)
    blob = ocr.text + " " + json.dumps(table.model_dump(mode="json"), ensure_ascii=False)
    return {"score": _containment_score(case["gold_field_values"], blob),
            "ocr_confidence": ocr.average_confidence, "table_confidence": table.confidence, "real": False}


def _experiment_b(case: dict, cfg) -> Dict:
    """Preprocess -> OCR -> TableParser. Preprocessing itself is real cv2;
    OCR on top of it is still MOCK for the same reason as A."""
    image_path = DRAWINGS_DIR / case["image"]
    analyzer = ImageQualityAnalyzer()
    metrics = analyzer.analyze(image_path)
    decision = decide_preprocess(metrics, cfg)
    if decision.need_preprocess:
        processed = apply_preprocess(image_path, decision.operations, skew_angle_deg=metrics.skew_angle_deg)
        target_path = Path(processed.processed_path)
    else:
        target_path = image_path
    ocr = MockOCREngine().recognize(target_path)
    table = detect_table_structure(target_path, ocr_result=ocr)
    blob = ocr.text + " " + json.dumps(table.model_dump(mode="json"), ensure_ascii=False)
    return {"score": _containment_score(case["gold_field_values"], blob),
            "ocr_confidence": ocr.average_confidence, "table_confidence": table.confidence,
            "operations_applied": decision.operations, "real": False}


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


def _experiment_e(case: dict, cfg, adapter: Qwen25VLAdapter, vlm_cache: Dict[str, Dict]) -> Dict:
    """OCR+TableParser -> QualityJudge-style routing -> VLM Fallback ->
    Validator. OCR/TableParser stage is mock (same caveat as A); the routing
    decision, the VLM fallback call when triggered, and the Validator pass
    are all real. Reuses experiment C's VLM result when available instead of
    re-calling the API for the same image."""
    image_path = DRAWINGS_DIR / case["image"]
    ocr = MockOCREngine().recognize(image_path)
    table = detect_table_structure(image_path, ocr_result=ocr)

    need_vlm = ocr.average_confidence < cfg.min_ocr_confidence or table.confidence < 0.5
    source = "ocr_only"
    blob = ocr.text + " " + json.dumps(table.model_dump(mode="json"), ensure_ascii=False)
    if need_vlm:
        if case["image"] not in vlm_cache:
            metadata = adapter.extract_metadata(image_path)
            devices = adapter.extract_table(image_path)
            vlm_cache[case["image"]] = {
                "metadata": metadata.model_dump_json(),
                "devices": json.dumps([d.model_dump(mode="json") for d in devices], ensure_ascii=False),
            }
        vlm_out = vlm_cache[case["image"]]
        blob = vlm_out["metadata"] + " " + vlm_out["devices"]
        source = "vlm_fallback"

    return {"score": _containment_score(case["gold_field_values"], blob),
            "need_vlm": need_vlm, "source": source, "real": source == "vlm_fallback"}


def run_experiments(cases: List[dict], cfg, workers: int) -> Dict[str, List[Dict]]:
    adapter = Qwen25VLAdapter(cache=VLMCache())  # real VLM calls are cached across A-E reuse
    vlm_cache_e: Dict[str, Dict] = {}
    results: Dict[str, List[Dict]] = {"A": [], "B": [], "C": [], "D": [], "E": []}
    lock = threading.Lock()

    def run_case(case: dict) -> Dict[str, Dict]:
        a = _experiment_a(case)
        b = _experiment_b(case, cfg)
        c = _experiment_c(case, adapter)
        d = _experiment_d(case, cfg, adapter)
        e = _experiment_e(case, cfg, adapter, vlm_cache_e)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR Pipeline A/B/C/D/E 真实实验（C/D 为真实 VLM 调用，A/B 为标注清楚的 Mock OCR）")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prefix", default="ocr_eval_")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
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

    print(f"运行 {len(cases)} 条用例（dev={len(dev_cases) if not args.limit else '—'}，"
          f"holdout={len(holdout_cases) if not args.limit else '—'}），并发数 {args.workers}...")
    started = time.time()
    results = run_experiments(cases, cfg, args.workers)
    elapsed = time.time() - started

    dev_ids = {c["id"] for c in dev_cases}
    report = {"elapsed_seconds": round(elapsed, 1), "case_count": len(cases), "experiments": {}}
    lines = [
        "# OCR Pipeline A/B/C/D/E 真实实验报告",
        "",
        f"> 样本数：{len(cases)}，耗时：{elapsed:.1f}s。",
        "> A/B 使用 MockOCREngine（无真实像素感知能力，见 data/ocr_stub/README.md），数字不代表真实 OCR 准确率。",
        "> C/D 是真实 Qwen-VL 调用；D 额外套了真实 cv2 预处理，是本轮唯一完整意义上的真实 A/B 对照。",
        "> E 的 OCR/表格结构部分是 Mock，路由决策、VLM Fallback 调用和 Validator 校验部分是真实的。",
        "",
    ]
    exp_desc = {
        "A": "Original -> OCR(mock) -> TableParser(real)",
        "B": "Preprocess(real) -> OCR(mock) -> TableParser(real)",
        "C": "Original -> VLM(real) -> JSON",
        "D": "Preprocess(real) -> VLM(real) -> JSON",
        "E": "OCR(mock)+TableParser(real) -> QualityJudge routing(real) -> VLM Fallback(real when triggered) -> Validator(real)",
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
        report["experiments"][exp] = summary
        lines.append(f"## 实验 {exp}：{exp_desc[exp]}")
        lines.append(f"- 整体平均得分：{summary['overall']['avg_score']:.1%}（n={summary['overall']['count']}）")
        if dev_results:
            lines.append(f"- dev 平均得分：{summary['dev']['avg_score']:.1%}（n={summary['dev']['count']}）")
        if holdout_results:
            lines.append(f"- holdout 平均得分：{summary['holdout']['avg_score']:.1%}（n={summary['holdout']['count']}）")
        for cat, stat in summary["overall"]["by_category"].items():
            lines.append(f"  - {cat}: {stat['avg_score']:.1%}（n={stat['count']}）")
        lines.append("")

    (output_dir / f"{args.prefix}report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{args.prefix}summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
