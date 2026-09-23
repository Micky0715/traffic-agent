"""Validate human render-review records and merge them with automated QA.

A human conclusion is taken exactly as recorded. Nothing here reads pixels,
re-runs QA, or fills a missing answer: an unanswered check stays `None`, an
`uncertain` stays uncertain, and a page is `pass` only when all five checks are
an explicit "pass".

A record is merged only if it points at a page that exists and whose PNG hash
is the one the reviewer saw. A mismatch is an error for a person to resolve;
it is never "fixed" by adopting the current hash.

Scope is kept literal. Reviewing 6 pages verifies 6 pages. Reviewing one
crop-warning page of one document says nothing about the other documents'
warnings, however alike their numbers look.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

CHECKS = (
    "text_not_obviously_blurred",
    "thin_lines_not_obviously_missing",
    "orientation_correct",
    "title_block_and_edges_not_cropped",
    "page_matches_pdf",
)
# Which confirmed problem a failed check stands for.
CHECK_PROBLEM = {
    "text_not_obviously_blurred": "text_blur_confirmed",
    "thin_lines_not_obviously_missing": "thin_line_loss_confirmed",
    "orientation_correct": "orientation_error_confirmed",
    "title_block_and_edges_not_cropped": "crop_confirmed",
    "page_matches_pdf": "pdf_mismatch_confirmed",
}
ANSWERS = ("pass", "fail", "uncertain", None)
REQUIRED = ("page_id", "file_name", "page_no", "page_sha256", "annotator",
            "reviewed_at", "notes") + CHECKS

CROP_CODE = "crop_risk_suspected"
CROP_DOCUMENTS = ("D&OM-25.pdf", "RCD-25.pdf", "SMD.pdf", "TSR.pdf")


class HumanReviewError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def page_verdict(record: Dict) -> str:
    """pass only if all five are an explicit pass; any fail is fail; else pending."""
    answers = [record.get(check) for check in CHECKS]
    if any(a == "fail" for a in answers):
        return "fail"
    if all(a == "pass" for a in answers):
        return "pass"
    return "pending"


def validate(records: Sequence[Dict], pages: Dict[str, Dict], *,
             repo_root: Path) -> List[str]:
    """Every reason these records cannot be merged. Empty means mergeable."""
    problems: List[str] = []
    seen: Dict[str, Dict] = {}
    for index, record in enumerate(records, start=1):
        where = f"line {index} ({record.get('page_id')})"
        missing = [k for k in REQUIRED if k not in record]
        if missing:
            problems.append(f"{where}: missing fields {missing}")
            continue
        for check in CHECKS:
            if record[check] not in ANSWERS:
                problems.append(f"{where}: {check}={record[check]!r} is not one of "
                                f"pass / fail / uncertain / null")
        if not record["annotator"]:
            problems.append(f"{where}: annotator is empty")
        if not record["reviewed_at"]:
            problems.append(f"{where}: reviewed_at is empty")
        if not isinstance(record["notes"], str):
            problems.append(f"{where}: notes must be a string")

        page = pages.get(record["page_id"])
        if page is None:
            problems.append(f"{where}: page_id does not exist in the page manifest")
        else:
            if record["page_no"] != page["page_no"]:
                problems.append(f"{where}: page_no {record['page_no']} != manifest "
                                f"{page['page_no']}")
            if record["page_sha256"] != page["rendered_sha256"]:
                problems.append(f"{where}: page_sha256 differs from the page manifest; "
                                f"the reviewed image is not the frozen one")
            else:
                png = repo_root / page["rendered_path"]
                if not png.exists() or sha256_file(png) != page["rendered_sha256"]:
                    problems.append(f"{where}: PNG on disk no longer matches its hash")

        if record["page_id"] in seen:
            problems.append(f"{where}: page reviewed more than once"
                            + (" with contradictory answers"
                               if any(seen[record["page_id"]].get(c) != record.get(c)
                                      for c in CHECKS) else ""))
        seen[record["page_id"]] = record
    return problems


def merge(records: Sequence[Dict], qa_pages: Sequence[Dict],
          sample_ids: Sequence[str]) -> Dict:
    """Human conclusions beside automated QA, with scope stated literally."""
    by_id = {r["page_id"]: r for r in records}
    qa_by_id = {p["page_id"]: p for p in qa_pages}

    per_page = []
    for page_id, record in by_id.items():
        qa = qa_by_id.get(page_id, {})
        verdict = page_verdict(record)
        auto = qa.get("qa_status")
        if auto == "warning" and verdict == "pass":
            comparison = "auto_warning_human_pass"      # a false positive warning
        elif auto == "pass" and verdict == "fail":
            comparison = "auto_pass_human_fail"         # automation missed it
        elif auto == verdict or (auto == "pass" and verdict == "pass"):
            comparison = "agree"
        elif auto in ("warning", "failed") and verdict == "fail":
            comparison = "agree_problem"
        else:
            comparison = "human_pending"
        per_page.append({
            "page_id": page_id, "file_name": record["file_name"],
            "page_no": record["page_no"], "page_sha256": record["page_sha256"],
            "in_sample": page_id in sample_ids,
            "human_answers": {c: record[c] for c in CHECKS},
            "human_verdict": verdict,
            "auto_qa_status": auto, "auto_reason_codes": qa.get("reason_codes", []),
            "comparison": comparison,
            "annotator": record["annotator"], "reviewed_at": record["reviewed_at"],
            "review_capture": record.get("review_capture"),
            "notes": record["notes"],
        })
    per_page.sort(key=lambda r: (r["file_name"], r["page_no"]))

    verdicts = Counter(r["human_verdict"] for r in per_page)
    confirmed = {name: 0 for name in CHECK_PROBLEM.values()}
    for record in by_id.values():
        for check, problem in CHECK_PROBLEM.items():
            if record[check] == "fail":
                confirmed[problem] += 1

    # ---- crop warnings: what was actually looked at ----------------------
    crop_pages = [p for p in qa_pages if CROP_CODE in p.get("reason_codes", [])]
    labels = {}
    for page in crop_pages:
        record = by_id.get(page["page_id"])
        answer = record.get("title_block_and_edges_not_cropped") if record else None
        labels[page["page_id"]] = (
            "true_positive" if answer == "fail" else
            "false_positive_warning" if answer == "pass" else
            "unreviewed")
    counts = Counter(labels.values())

    per_doc_crop = {}
    for name in sorted({p["file_name"] for p in crop_pages}):
        doc_pages = [p for p in crop_pages if p["file_name"] == name]
        reviewed = [p["page_id"] for p in doc_pages if labels[p["page_id"]] != "unreviewed"]
        per_doc_crop[name] = {"warnings": len(doc_pages), "reviewed": len(reviewed),
                              "reviewed_pages": reviewed}
    covered = [d for d, v in per_doc_crop.items() if v["reviewed"]]
    uncovered = [d for d, v in per_doc_crop.items() if not v["reviewed"]]
    if not crop_pages:
        crop_conclusion = "no crop warnings"
    elif uncovered:
        crop_conclusion = (
            f"insufficient evidence: only {covered or 'no document'} had a crop-warning "
            f"page reviewed. The result cannot be generalised to {uncovered}; their "
            f"warnings stay unreviewed.")
    else:
        crop_conclusion = ("every crop-warning document has a reviewed page; unreviewed "
                           "pages are still unreviewed, not verified")

    per_doc = defaultdict(lambda: Counter())
    for r in per_page:
        per_doc[r["file_name"]][r["human_verdict"]] += 1

    reviewed_ids = [r["page_id"] for r in per_page if r["human_verdict"] != "pending"]
    return {
        "selected_pages": list(sample_ids),
        "human_reviewed_pages": reviewed_ids,
        "pending_pages": [pid for pid in sample_ids if pid not in reviewed_ids],
        "pass_pages": [r["page_id"] for r in per_page if r["human_verdict"] == "pass"],
        "failed_pages": [r["page_id"] for r in per_page if r["human_verdict"] == "fail"],
        **confirmed,
        "crop_warning_total": len(crop_pages),
        "crop_warning_true_positive": counts.get("true_positive", 0),
        "crop_warning_false_positive": counts.get("false_positive_warning", 0),
        "crop_warning_unreviewed": counts.get("unreviewed", 0),
        "crop_warning_labels": dict(sorted(labels.items())),
        "crop_warning_by_document": per_doc_crop,
        "crop_conclusion": crop_conclusion,
        "crop_documents_without_reviewed_page": uncovered,
        "per_document": {k: dict(v) for k, v in sorted(per_doc.items())},
        "per_page": per_page,
        "human_verified_scope": {
            "pages": reviewed_ids,
            "count": len(reviewed_ids),
            "of_total_pages": len(qa_pages),
            "statement": (f"Only these {len(reviewed_ids)} page(s) were looked at by a "
                          f"person. The other {len(qa_pages) - len(reviewed_ids)} page(s) "
                          f"are covered by automated integrity checks only."),
        },
    }


def supplement_candidates(qa_pages: Sequence[Dict], merged: Dict) -> List[Dict]:
    """One crop-warning page from each crop document with none reviewed.

    Deterministic: the lowest page number carrying the warning.
    """
    out = []
    for name in merged["crop_documents_without_reviewed_page"]:
        page = min((p for p in qa_pages if p["file_name"] == name
                    and CROP_CODE in p.get("reason_codes", [])),
                   key=lambda p: p["page_no"])
        out.append(page)
    return out
