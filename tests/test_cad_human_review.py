"""Human render-review validation and merge.

Validation tests use SYNTHETIC pages and PNGs in tmp_path. The two REAL tests
run the merge against the actual review file and reports, write only to
tmp_path, and assert that the review file, v1 report and first-round sample
manifest are byte-identical afterwards.

No OCR, no VLM, no network.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from src.cad_corpus import human_review as hr
import scripts.cad_render_human_review as cli

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def world(tmp_path):
    """Two synthetic rendered pages and a matching page manifest."""
    pages = {}
    for n, name in ((1, "A.pdf"), (2, "B.pdf")):
        png = tmp_path / f"p{n}.png"
        Image.new("L", (40, 30), 255 - n).save(png)
        pages[f"doc{n}:p0001"] = {"page_id": f"doc{n}:p0001", "page_no": 1,
                                  "rendered_path": png.name,
                                  "rendered_sha256": sha(png), "file_name": name}
    return tmp_path, pages


def record(pages, page_id="doc1:p0001", **over):
    base = {"page_id": page_id, "file_name": pages[page_id]["file_name"], "page_no": 1,
            "page_sha256": pages[page_id]["rendered_sha256"],
            "annotator": "t", "reviewed_at": "2026-09-18T00:00:00Z", "notes": "",
            **{c: "pass" for c in hr.CHECKS}}
    base.update(over)
    return base


def qa_page(page_id, name, status="pass", codes=()):
    return {"page_id": page_id, "file_name": name, "page_no": 1,
            "qa_status": status, "reason_codes": list(codes)}


# --------------------------------------------------------------------------
# validation — refuse, never repair
# --------------------------------------------------------------------------

def test_a_clean_record_validates(world):
    root, pages = world
    assert hr.validate([record(pages)], pages, repo_root=root) == []


@pytest.mark.parametrize("field", ["annotator", "reviewed_at", "page_sha256",
                                   "orientation_correct"])
def test_missing_human_field_is_refused(world, field):
    root, pages = world
    rec = record(pages)
    del rec[field]
    assert any("missing fields" in p for p in hr.validate([rec], pages, repo_root=root))


def test_empty_annotator_and_non_string_notes_are_refused(world):
    root, pages = world
    problems = hr.validate([record(pages, annotator="", notes=None)], pages, repo_root=root)
    assert any("annotator is empty" in p for p in problems)
    assert any("notes must be a string" in p for p in problems)


def test_an_answer_outside_the_vocabulary_is_refused(world):
    root, pages = world
    problems = hr.validate([record(pages, orientation_correct=True)], pages, repo_root=root)
    assert any("orientation_correct=True" in p for p in problems)


def test_a_page_hash_mismatch_is_refused_not_repaired(world):
    root, pages = world
    rec = record(pages, page_sha256="0" * 64)
    problems = hr.validate([rec], pages, repo_root=root)
    assert any("not the frozen one" in p for p in problems)
    assert rec["page_sha256"] == "0" * 64          # left exactly as given


def test_a_png_changed_on_disk_is_refused(world):
    root, pages = world
    Image.new("L", (40, 30), 0).save(root / "p1.png")
    problems = hr.validate([record(pages)], pages, repo_root=root)
    assert any("no longer matches" in p for p in problems)


def test_a_page_that_does_not_exist_is_refused(world):
    root, pages = world
    rec = record(pages)
    rec["page_id"] = "ghost:p0009"
    assert any("does not exist" in p for p in hr.validate([rec], pages, repo_root=root))


def test_a_wrong_page_number_is_refused(world):
    root, pages = world
    assert any("page_no" in p for p in
               hr.validate([record(pages, page_no=7)], pages, repo_root=root))


def test_duplicate_records_are_refused_and_contradictions_named(world):
    root, pages = world
    same = hr.validate([record(pages), record(pages)], pages, repo_root=root)
    assert any("more than once" in p for p in same)
    clash = hr.validate([record(pages), record(pages, page_matches_pdf="fail")],
                        pages, repo_root=root)
    assert any("contradictory answers" in p for p in clash)


# --------------------------------------------------------------------------
# verdicts — partial is never a pass
# --------------------------------------------------------------------------

def test_partially_filled_is_pending_not_pass(world):
    _, pages = world
    assert hr.page_verdict(record(pages, page_matches_pdf=None)) == "pending"
    assert hr.page_verdict(record(pages, orientation_correct="uncertain")) == "pending"
    assert hr.page_verdict(record(pages)) == "pass"
    assert hr.page_verdict(record(pages, text_not_obviously_blurred="fail",
                                  page_matches_pdf=None)) == "fail"


def test_auto_warning_human_pass_is_a_false_positive_warning(world):
    _, pages = world
    qa = [qa_page("doc1:p0001", "A.pdf", "warning", [hr.CROP_CODE])]
    m = hr.merge([record(pages)], qa, ["doc1:p0001"])
    assert m["per_page"][0]["comparison"] == "auto_warning_human_pass"
    assert m["crop_warning_false_positive"] == 1 and m["crop_warning_true_positive"] == 0


def test_auto_pass_human_fail_is_recorded_as_a_miss(world):
    _, pages = world
    qa = [qa_page("doc1:p0001", "A.pdf", "pass")]
    m = hr.merge([record(pages, thin_lines_not_obviously_missing="fail")], qa, ["doc1:p0001"])
    assert m["per_page"][0]["comparison"] == "auto_pass_human_fail"
    assert m["thin_line_loss_confirmed"] == 1
    assert m["failed_pages"] == ["doc1:p0001"]


def test_a_human_crop_fail_is_a_true_positive(world):
    _, pages = world
    qa = [qa_page("doc1:p0001", "A.pdf", "warning", [hr.CROP_CODE])]
    m = hr.merge([record(pages, title_block_and_edges_not_cropped="fail")], qa, ["doc1:p0001"])
    assert m["crop_warning_true_positive"] == 1 and m["crop_confirmed"] == 1


def test_one_reviewed_document_does_not_generalise_to_the_others(world):
    _, pages = world
    qa = [qa_page("doc1:p0001", "A.pdf", "warning", [hr.CROP_CODE]),
          qa_page("doc2:p0001", "B.pdf", "warning", [hr.CROP_CODE])]
    m = hr.merge([record(pages)], qa, ["doc1:p0001"])
    assert m["crop_warning_unreviewed"] == 1                 # B stays unreviewed
    assert m["crop_warning_labels"]["doc2:p0001"] == "unreviewed"
    assert m["crop_documents_without_reviewed_page"] == ["B.pdf"]
    assert "cannot be generalised" in m["crop_conclusion"]
    assert [p["page_id"] for p in hr.supplement_candidates(qa, m)] == ["doc2:p0001"]


def test_verified_scope_is_only_what_was_reviewed(world):
    _, pages = world
    qa = [qa_page("doc1:p0001", "A.pdf"), qa_page("doc2:p0001", "B.pdf")]
    m = hr.merge([record(pages)], qa, ["doc1:p0001", "doc2:p0001"])
    assert m["human_verified_scope"]["count"] == 1
    assert m["human_verified_scope"]["of_total_pages"] == 2
    assert m["pending_pages"] == ["doc2:p0001"]


# --------------------------------------------------------------------------
# REAL — protection of inputs and history
# --------------------------------------------------------------------------

def test_real_merge_leaves_review_file_v1_report_and_sample_untouched(tmp_path):
    watched = [cli.REVIEW, cli.QA_V1, cli.QA_V1.with_suffix(".md"), cli.SAMPLE]
    before = [sha(p) for p in watched]
    out = {k: tmp_path / f"{k}.out" for k in cli.OUT}
    result = cli.run(cli.REVIEW, out=out, supplement_dir=tmp_path / "supp")
    assert [sha(p) for p in watched] == before
    assert result["v2"]["human_visual_review"]["human_verified"] is False
    # regression: QA rows lack the PNG hash; the supplement must take it from
    # the page manifest (first real run raised KeyError here)
    assert all(len(p["rendered_sha256"]) == 64 for p in result["supplement"])


def test_v1_report_and_review_file_are_protected_outputs(tmp_path):
    for target in (cli.QA_V1, cli.QA_V1.with_suffix(".md"), cli.REVIEW, cli.SAMPLE):
        with pytest.raises(hr.HumanReviewError, match="protected"):
            cli.write_new(target, "x", overwrite=True)


def test_supplement_export_keeps_unanswered_as_null(tmp_path):
    page = {"page_id": "d:p0001", "file_name": "A.pdf", "page_no": 1,
            "rendered_path": "x.png", "rendered_sha256": "a" * 64, "qa_status": "warning",
            "reason_codes": [hr.CROP_CODE], "width": 1, "height": 1, "dpi": 300,
            "edge_content_ratio": {"top": 0, "bottom": 0, "left": 0, "right": 0}}
    text = cli.supplement_html([page], tmp_path, "reason")
    assert "hit ? hit.value : null" in text
    assert 'value="pass"' in text and 'value="fail"' in text and 'value="uncertain"' in text
    assert "a" * 64 in text                      # the export carries the page hash
