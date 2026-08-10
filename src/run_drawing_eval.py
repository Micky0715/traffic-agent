from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.drawing_extractor import extract_drawing
from src.evaluator import ROOT


def load_cases(path: str | None = None) -> list[dict]:
    case_path = Path(path) if path else (ROOT / "data" / "drawing_cases.jsonl")
    rows = []
    with case_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_case(case: dict, pred) -> dict:
    gold_fields = case["gold_key_fields"]
    pred_fields = pred.key_fields

    field_matches = {}
    for key, expected in gold_fields.items():
        got = pred_fields.get(key)
        field_matches[key] = (got == expected, expected, got)
    field_accuracy = sum(1 for ok, _, _ in field_matches.values() if ok) / max(1, len(field_matches))

    hallucinated = []
    for absent_field in case.get("fields_not_present", []):
        if absent_field in pred_fields:
            hallucinated.append({"field": absent_field, "value": pred_fields[absent_field]})
    # A field genuinely absent from the drawing must not silently appear with a
    # made-up value; explaining it in notes/low_confidence is fine, inventing a
    # value with no acknowledgement is the failure mode being tested for.
    hallucination_ok = len(hallucinated) == 0

    expect_low_conf = case.get("expect_low_confidence_field")
    low_conf_flagged = expect_low_conf is None or expect_low_conf in pred.low_confidence_fields

    drawing_no_ok = pred.drawing_no == case["gold_drawing_no"]
    asset_id_ok = pred.asset_id == case["gold_asset_id"]

    success = field_accuracy == 1.0 and hallucination_ok and drawing_no_ok and asset_id_ok

    return {
        "id": case["id"],
        "description": case["description"],
        "tags": case["tags"],
        "drawing_no_ok": drawing_no_ok,
        "asset_id_ok": asset_id_ok,
        "field_accuracy": field_accuracy,
        "field_matches": field_matches,
        "hallucination_ok": hallucination_ok,
        "hallucinated_fields": hallucinated,
        "low_confidence_flagged": low_conf_flagged,
        "pred_low_confidence_fields": pred.low_confidence_fields,
        "pred_notes": pred.notes,
        "success": success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="用真实 Qwen-VL 跑图纸结构化抽取评测")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prefix", default="drawing_")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    drawings_dir = ROOT / "data" / "drawings"

    records = []
    scored = []
    for i, case in enumerate(cases, start=1):
        print(f"\r  {i}/{len(cases)} {case['id']}", end="", flush=True)
        image_path = drawings_dir / case["image"]
        try:
            pred = extract_drawing(image_path, case.get("question", ""))
            error = None
        except Exception as e:  # noqa: BLE001 - a failed VL call is itself worth recording
            pred = None
            error = str(e)

        if pred is not None:
            result = score_case(case, pred)
            result["error"] = None
            result["raw_prediction"] = pred.model_dump(mode="json")
        else:
            result = {"id": case["id"], "description": case["description"], "tags": case["tags"],
                       "success": False, "error": error, "raw_prediction": None}
        scored.append(result)
        records.append(result)
    print()

    n = len(cases)
    success_count = sum(1 for r in scored if r["success"])
    field_accuracies = [r["field_accuracy"] for r in scored if "field_accuracy" in r]
    avg_field_accuracy = sum(field_accuracies) / len(field_accuracies) if field_accuracies else 0.0
    hallucination_free = sum(1 for r in scored if r.get("hallucination_ok", True))

    report = {
        "case_count": n,
        "case_success_rate": success_count / n,
        "avg_field_accuracy": avg_field_accuracy,
        "hallucination_free_rate": hallucination_free / n,
        "results": scored,
    }
    (output_dir / f"{args.prefix}report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 真实 Qwen-VL 图纸结构化抽取评测报告",
        "",
        f"> 数据集：{args.dataset or 'data/drawing_cases.jsonl'}，样本数：{n}，"
        "图纸为脚本生成的合成图纸（无真实企业图纸可用）。",
        "",
        f"- 用例完全成功率（字段全对 + 不编造缺失字段 + 图号/设备号对）：{report['case_success_rate']:.1%}",
        f"- 平均字段准确率：{report['avg_field_accuracy']:.1%}",
        f"- 不编造缺失字段的比例：{report['hallucination_free_rate']:.1%}",
        "",
    ]
    (output_dir / f"{args.prefix}summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
