"""Generate two evaluation sets, kept strictly apart.

1. A derived regression set, built mechanically from fields the corpus already
   contains. Labelled `derived_from_existing_gold` / `regression_only`. It can
   catch a regression; it cannot demonstrate generalization, because the
   questions were written from the answers.

2. A set of UNREVIEWED candidates for a human to label. Every expected value in
   it is a proposal, never a verdict, and the file says so in every row.

Two rules this script will not break:

  it never runs the system and writes the output back as the expectation —
  that would score the system against itself and always pass;

  it never renames its own output to human gold. `label_status` stays
  `unreviewed` until a person changes it, and no metric that claims to measure
  refusal or recall may be computed from it before then.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.service import StructuredRagService  # noqa: E402

CANDIDATES = ROOT / "data" / "rag_eval_candidates_unreviewed.jsonl"
DERIVED = ROOT / "data" / "rag_regression_derived.jsonl"
REVIEW_DOC = ROOT / "docs" / "rag_eval_candidates_review.md"

random.seed(20260916)

FIELD_SYNONYM_SURFACE = {
    "功率": "额定功率", "风量": "排风量", "控制柜编号": "控制柜号",
    "电机编号": "电机号", "断路器编号": "断路器号", "图号": "图纸编号",
    "设备编号": "设备号",
}
ABSENT_DEVICES = ["A99风机", "B77屏蔽门", "9号水泵", "C41风机", "TR-05变压器",
                  "ESC-12扶梯", "A00风机", "PUMP-99"]
IRRELEVANT = ["今天天气怎么样", "帮我写一首诗", "解释一下量子纠缠", "附近有什么餐厅"]


def main() -> None:
    service = StructuredRagService()
    service.load_corpus()

    by_document: dict = {}
    for chunk in service.chunks:
        if chunk.field_name and chunk.field_value:
            by_document.setdefault(chunk.document_id, {})[chunk.field_name] = chunk

    documents = sorted(by_document)
    name_of = {}
    for document_id, fields in by_document.items():
        title = fields.get("名称")
        name_of[document_id] = title.field_value if title else document_id

    # ---------------- 1. derived regression set -------------------------
    derived = []
    for document_id in documents:
        for field_name, chunk in sorted(by_document[document_id].items()):
            if field_name in {"名称", "页码", "版本"}:
                continue
            derived.append({
                "case_id": f"DRV-{len(derived):03d}",
                "query": f"查询{name_of[document_id]}的{field_name}",
                "expected_field": field_name,
                "expected_value": chunk.field_value,
                "expected_document_id": document_id,
                "expected_page": chunk.page_start,
                "expected_chunk_id": chunk.chunk_id,
                "label_source": "derived_from_existing_gold",
                "purpose": "regression_only",
                "caveat": ("question written from the answer; catches regressions, "
                           "cannot show generalization"),
            })
    DERIVED.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in derived) + "\n",
        encoding="utf-8")

    # ---------------- 2. unreviewed candidates --------------------------
    candidates = []

    def add(query, decision, fields, chunk_ids, answer, method, note=""):
        candidates.append({
            "case_id": f"RAGC-{len(candidates):03d}",
            "query": query,
            "candidate_expected_decision": decision,
            "candidate_relevant_chunk_ids": chunk_ids,
            "candidate_required_fields": fields,
            "candidate_answer": answer,
            "label_status": "unreviewed",
            "generation_method": method,
            "reviewer_note": note,
        })

    single = [d for d in documents if "MULTI" not in d]
    multi = [d for d in documents if "MULTI" in d]

    # 8 exact device-field queries
    for document_id in single[:8]:
        fields = by_document[document_id]
        field_name = next((f for f in ("电机编号", "断路器编号", "控制柜编号", "图号")
                           if f in fields), None)
        if not field_name:
            continue
        chunk = fields[field_name]
        add(f"查询{name_of[document_id]}的{field_name}", "execute", [field_name],
            [chunk.chunk_id], chunk.field_value, "exact_field_from_corpus")

    # 8 synonym phrasings
    for document_id in single[:8]:
        fields = by_document[document_id]
        field_name = next((f for f in FIELD_SYNONYM_SURFACE if f in fields), None)
        if not field_name:
            continue
        chunk = fields[field_name]
        add(f"{name_of[document_id]}的{FIELD_SYNONYM_SURFACE[field_name]}是多少",
            "execute", [field_name], [chunk.chunk_id], chunk.field_value,
            "synonym_surface_form",
            "confirm the synonym belongs in configs/rag.yaml")

    # 8 devices that do not exist
    for device in ABSENT_DEVICES:
        add(f"查询{device}的电机编号", "abstain", ["电机编号"], [], None,
            "absent_device", "confirm this device really is absent")

    # 8 device present, field absent
    for document_id in single[:8]:
        present = by_document[document_id]
        missing = next((f for f in ("页码", "版本", "功率", "风量")
                        if f not in present), "功率")
        add(f"查询{name_of[document_id]}的{missing}", "abstain", [missing], [], None,
            "field_absent_for_present_device")

    # 6 multi-field partial
    for document_id in single[:6]:
        present = by_document[document_id]
        have = next((f for f in ("控制柜编号", "电机编号") if f in present), None)
        missing = next((f for f in ("页码", "版本", "功率") if f not in present), "功率")
        if not have:
            continue
        add(f"查询{name_of[document_id]}的{have}和{missing}", "partial",
            [have, missing], [present[have].chunk_id], present[have].field_value,
            "one_present_one_absent")

    # 6 multi-device interference
    for document_id in multi:
        entities = sorted({c.entity_id for c in service.chunks
                           if c.document_id == document_id and c.entity_id})
        for entity in entities[:3]:
            chunk = next((c for c in service.chunks
                          if c.document_id == document_id and c.entity_id == entity
                          and c.field_name in {"功率", "流量"}), None)
            if chunk is None:
                continue
            add(f"查询{entity}的{chunk.field_name}", "execute", [chunk.field_name],
                [chunk.chunk_id], chunk.field_value, "multi_device_interference",
                "sibling devices must not leak into the answer")

    # 4 irrelevant
    for text in IRRELEVANT:
        add(text, "clarify", [], [], None, "irrelevant_question",
            "decide whether clarify or abstain is the right refusal here")

    CANDIDATES.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in candidates) + "\n",
        encoding="utf-8")

    # ---------------- review document -----------------------------------
    buckets: dict = {}
    for row in candidates:
        buckets.setdefault(row["generation_method"], []).append(row)

    lines = [
        "# RAG 评测候选集 · 待人工审核",
        "",
        f"共 {len(candidates)} 条，全部 `label_status=unreviewed`。",
        "",
        "> **这些不是 Gold。** 每一条的 `candidate_expected_decision` 与 `candidate_answer`",
        "> 都是机器提出的**建议**，由脚本按语料结构生成，**不是运行系统得到的输出**。",
        "> 人工确认之前，不得用它计算任何正式 Recall@5、拒答率或准确率指标。",
        "",
        "审核方式：逐条确认或修改 `candidate_expected_decision`，",
        "然后把 `label_status` 改成 `reviewed`。**只有改过的行才算 Gold。**",
        "",
        f"文件：`data/rag_eval_candidates_unreviewed.jsonl`",
        "",
        "## 分布",
        "",
        "| 生成方式 | 条数 | 关注点 |",
        "|---|---|---|",
    ]
    focus = {
        "exact_field_from_corpus": "语料确有该字段，应可作答",
        "synonym_surface_form": "同义表达能否命中同一字段",
        "absent_device": "设备不存在，应拒答而非用邻近设备顶替",
        "field_absent_for_present_device": "设备在但字段缺，应拒答",
        "one_present_one_absent": "应 partial，不能因缺一个否决另一个",
        "multi_device_interference": "同表兄弟设备不得串入答案",
        "irrelevant_question": "完全无关，refusal 形式待定",
    }
    for method, rows in sorted(buckets.items()):
        lines.append(f"| `{method}` | {len(rows)} | {focus.get(method,'')} |")

    for method, rows in sorted(buckets.items()):
        lines += ["", f"## {method}", "",
                  "| case_id | query | 建议决策 | 建议答案 | 审核备注 |",
                  "|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['case_id']} | {row['query']} | "
                         f"{row['candidate_expected_decision']} | "
                         f"{row['candidate_answer'] or '—'} | {row['reviewer_note'] or ''} |")
    REVIEW_DOC.write_text("\n".join(lines), encoding="utf-8")

    print(f"derived regression set : {len(derived)} -> {DERIVED.name}")
    print(f"unreviewed candidates  : {len(candidates)} -> {CANDIDATES.name}")
    for method, rows in sorted(buckets.items()):
        print(f"  {method:<34} {len(rows)}")
    print(f"review doc             : {REVIEW_DOC}")


if __name__ == "__main__":
    main()
