"""Automated render QA and the human sample.

Every image here is SYNTHETIC, drawn in tmp_path. Page geometry is injected in
place of Poppler's pdfinfo, which is not installed; the real-pdfinfo path is
covered only by the "missing Poppler blocks" tests. No real downloaded PDF or
rendered page is used as a fixture.

No OCR, no VLM, no network.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from src.cad_corpus import render_qa as rq
from src.cad_corpus.corpus import RendererUnavailable
import scripts.cad_render_quality_audit as audit_cli

ROOT = Path(__file__).resolve().parents[1]
DPI = 300
W_PX, H_PX = 300, 200                       # 72 x 48 pt at 300 dpi
BOX = (72.0, 48.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def qa():
    return rq.load_qa_config()


def drawing(path: Path, *, size=(W_PX, H_PX), blank=False, edge=None,
            blur=False) -> Path:
    """A plausible drawing: an inset frame, some text-like strokes."""
    image = Image.new("L", size, 255)
    if not blank:
        d = ImageDraw.Draw(image)
        w, h = size
        d.rectangle([20, 20, w - 21, h - 21], outline=0, width=2)
        for y in range(40, h - 40, 12):
            d.line([40, y, w - 60, y], fill=0, width=1)
        d.rectangle([w - 110, h - 60, w - 30, h - 30], outline=0)
        if edge == "right":
            d.rectangle([w - 3, 0, w - 1, h - 1], fill=0)
        if edge == "top":
            d.rectangle([0, 0, w - 1, 2], fill=0)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def setup(tmp_path, specs, *, pdf_pages=None, doc="A.pdf"):
    """specs: list of kwargs for drawing(), one per rendered page."""
    root = tmp_path / "repo"
    pdf = root / "dl" / doc
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4 synthetic")
    source_id = f"cad-{doc[0].lower()}"
    source = {"source_id": source_id, "original_path": f"dl/{doc}", "file_name": doc,
              "source_sha256": sha(pdf), "source_class": "pdf",
              "page_count": pdf_pages or len(specs)}
    pages = []
    for n, spec in enumerate(specs, start=1):
        if spec is None:
            continue
        png = drawing(root / "png" / f"{source_id}__p{n:04d}.png", **spec)
        pages.append({"page_id": f"{source_id}:p{n:04d}", "source_id": source_id,
                      "page_no": n, "rendered_path": png.relative_to(root).as_posix(),
                      "rendered_sha256": sha(png)})
    count = pdf_pages or len(specs)
    boxes = lambda _pdf: (count, [rq.PageBox(n, *BOX, 0) for n in range(1, count + 1)])
    return root, [source], pages, boxes


def run_audit(root, sources, pages, boxes, qa):
    return rq.audit(sources, pages, qa, dpi=DPI, max_pixels=10**9,
                    repo_root=root, boxes_for=boxes)


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------

def test_page_count_mismatch_fails_and_the_document_is_incomplete(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{}, {}, None], pdf_pages=3)
    results, docs = run_audit(root, sources, pages, boxes, qa)
    missing = next(p for p in results if p["page_no"] == 3)
    assert missing["qa_status"] == rq.FAILED
    assert rq.R_PNG_MISSING in missing["reason_codes"]
    assert all(rq.R_PAGE_COUNT in p["reason_codes"] for p in results)
    # one missing page is not averaged away into "the document succeeded"
    assert docs[0]["document_complete"] is False
    assert docs[0]["rendered_page_count"] == 2 and docs[0]["pdfinfo_page_count"] == 3


def test_a_corrupt_image_fails(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{}])
    bad = root / pages[0]["rendered_path"]
    bad.write_bytes(b"\x89PNG\r\n\x1a\n not really")
    pages[0]["rendered_sha256"] = sha(bad)
    results, docs = run_audit(root, sources, pages, boxes, qa)
    assert results[0]["image_readable"] is False
    assert rq.R_UNREADABLE in results[0]["reason_codes"]
    assert results[0]["qa_status"] == rq.FAILED
    assert docs[0]["document_complete"] is False


def test_a_png_that_differs_from_its_manifest_hash_fails(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{}])
    pages[0]["rendered_sha256"] = "0" * 64
    results, _ = run_audit(root, sources, pages, boxes, qa)
    assert rq.R_HASH_MISMATCH in results[0]["reason_codes"]


def test_size_that_does_not_match_the_dpi_fails(tmp_path, qa):
    """150 dpi pixels for a 300 dpi render: 50% off on both axes."""
    root, sources, pages, boxes = setup(tmp_path, [{"size": (150, 100)}])
    results, _ = run_audit(root, sources, pages, boxes, qa)
    page = results[0]
    assert (page["expected_width"], page["expected_height"]) == (W_PX, H_PX)
    assert page["dimension_error_ratio"] == pytest.approx(0.5)
    assert rq.R_DIM_FAIL in page["reason_codes"] and page["qa_status"] == rq.FAILED


def test_rotation_swaps_the_expected_size(qa):
    assert rq.expected_pixels(rq.PageBox(1, 72, 48, 90), DPI) == (200, 300)
    assert rq.expected_pixels(rq.PageBox(1, 72, 48, 0), DPI) == (300, 200)


# --------------------------------------------------------------------------
# warnings — all "suspected"
# --------------------------------------------------------------------------

def test_an_all_white_page_is_blank_suspected(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{"blank": True}])
    page = run_audit(root, sources, pages, boxes, qa)[0][0]
    assert page["blank_page_suspected"] is True
    assert page["qa_status"] == rq.WARNING


def test_a_sparse_drawing_is_not_called_blank(tmp_path, qa):
    """Bright on average — mean brightness alone would flag it. Contrast and
    the share of ink say otherwise."""
    root, sources, pages, boxes = setup(tmp_path, [{}])
    page = run_audit(root, sources, pages, boxes, qa)[0][0]
    assert page["grayscale_mean"] > 200
    assert page["blank_page_suspected"] is False


def test_content_on_the_edge_is_crop_risk_suspected_on_that_side(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{"edge": "right"}])
    page = run_audit(root, sources, pages, boxes, qa)[0][0]
    assert page["crop_risk"] == "suspected"
    assert page["crop_risk_sides"] == ["right"]
    assert rq.R_CROP in page["reason_codes"] and page["qa_status"] == rq.WARNING


def test_an_inset_frame_is_not_crop_risk(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{}])
    page = run_audit(root, sources, pages, boxes, qa)[0][0]
    assert page["crop_risk"] == "none"


def test_blur_score_is_lower_for_a_blurred_render(tmp_path, qa):
    sharp = np.asarray(Image.open(drawing(tmp_path / "s.png")).convert("L"))
    soft = np.asarray(Image.open(drawing(tmp_path / "b.png", blur=True)).convert("L"))
    assert rq.blur_score(soft, qa) < rq.blur_score(sharp, qa)
    metrics = rq.image_metrics(soft, qa)
    assert metrics["blur_threshold_status"] == "uncalibrated"


def test_a_size_jump_inside_one_document_is_flagged(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{}, {}, {"size": (296, 200)}])
    results, _ = run_audit(root, sources, pages, boxes, qa)
    odd = results[2]
    assert odd["document_size_outlier"] is True
    assert rq.R_SIZE_OUTLIER in odd["reason_codes"]
    assert results[0]["document_size_outlier"] is False


def test_thresholds_are_declared_uncalibrated(qa):
    assert qa["threshold_provenance"] == "uncalibrated"
    assert qa["calibrated_with_human_labels"] is False


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _pages(n, *, codes=None, sheets=None, status=rq.PASS, doc="d"):
    out = []
    for i in range(1, n + 1):
        out.append({"page_id": f"{doc}:p{i:04d}", "source_id": doc,
                    "file_name": f"{doc}.pdf", "page_no": i,
                    "sheet": (sheets or ["34.0x22.0in"])[i % len(sheets or [1])],
                    "qa_status": status, "reason_codes": list((codes or {}).get(i, []))})
    return out


def _docs(pages):
    counts = {}
    for p in pages:
        counts.setdefault(p["source_id"], [p["file_name"], 0])[1] += 1
    return [{"source_id": s, "file_name": f, "pdfinfo_page_count": n}
            for s, (f, n) in counts.items()]


def test_sample_is_capped_at_ten(qa):
    pages = []
    for d in range(14):
        pages += _pages(3, doc=f"d{d:02d}", status=rq.FAILED,
                        codes={1: [rq.R_UNREADABLE]})
    sample = rq.select_sample(pages, _docs(pages), qa)
    real = [s for s in sample if s.get("page_id")]
    assert len(real) == 10
    assert sample[-1]["truncated_candidates"]       # what the cap dropped is recorded


def test_with_no_warning_at_least_six_pages_are_still_sampled(qa):
    pages = _pages(20, doc="big") + _pages(2, doc="a") + _pages(2, doc="b") + \
        _pages(2, doc="c") + _pages(2, doc="e")
    sample = [s for s in rq.select_sample(pages, _docs(pages), qa) if s.get("page_id")]
    assert all(s["qa_status"] == rq.PASS for s in sample)
    assert 6 <= len(sample) <= 10
    big = [s["page_no"] for s in sample if s["source_id"] == "big"]
    assert {1, 11, 20} <= set(big)                  # first, middle, last


def test_every_failed_document_and_reason_code_is_represented(qa):
    pages = _pages(5, doc="x", codes={3: [rq.R_CROP]}) + \
        _pages(4, doc="y", status=rq.FAILED, codes={2: [rq.R_DIM_FAIL]})
    for p in pages:
        if p["source_id"] == "y" and p["page_no"] != 2:
            p["qa_status"] = rq.PASS
    sample = [s for s in rq.select_sample(pages, _docs(pages), qa) if s.get("page_id")]
    ids = {s["page_id"] for s in sample}
    assert "y:p0002" in ids and "x:p0003" in ids
    assert all(s["selection_reasons"] for s in sample)


def test_every_sheet_size_is_represented(qa):
    pages = _pages(6, doc="d", sheets=["34.0x22.0in", "17.0x11.0in"])
    sheets = {s.get("file_name") and next(p["sheet"] for p in pages
                                          if p["page_id"] == s["page_id"])
              for s in rq.select_sample(pages, _docs(pages), qa) if s.get("page_id")}
    assert sheets == {"34.0x22.0in", "17.0x11.0in"}


def test_unfilled_human_answers_are_never_a_pass(qa):
    pages = _pages(6, doc="d")
    for item in [s for s in rq.select_sample(pages, _docs(pages), qa) if s.get("page_id")]:
        assert item["human_review_status"] == "pending"
        assert all(v is None for v in item["human_review"].values())


# --------------------------------------------------------------------------
# determinism, read-only, blocking, isolation
# --------------------------------------------------------------------------

def test_rerun_is_deterministic_and_touches_neither_pdf_nor_png(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{}, {"edge": "top"}, {"blank": True}])
    watched = [root / sources[0]["original_path"]] + [root / p["rendered_path"] for p in pages]
    before = [sha(p) for p in watched]
    first, _ = run_audit(root, sources, pages, boxes, qa)
    second, _ = run_audit(root, sources, pages, boxes, qa)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [sha(p) for p in watched] == before
    assert all(p["png_unchanged_by_qa"] for p in first if "png_unchanged_by_qa" in p)


def test_the_cli_report_is_byte_identical_on_rerun(tmp_path, qa):
    root, sources, pages, boxes = setup(tmp_path, [{}, {"edge": "top"}])
    manifests = root / "manifests"
    manifests.mkdir()
    manifest = manifests / "source_manifest_unreviewed.jsonl"
    manifest.write_text(json.dumps(sources[0]) + "\n", encoding="utf-8")
    (manifests / "page_manifest_unreviewed.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in pages), encoding="utf-8")
    kwargs = dict(pack=root / "pack", report_json=root / "r.json",
                  report_md=root / "r.md", repo_root=root, boxes_for=boxes)
    audit_cli.run(manifest, **kwargs)
    first = [(root / n).read_bytes() for n in ("r.json", "r.md", "pack/index.html",
                                               "pack/qa_sample_manifest.jsonl")]
    report = audit_cli.run(manifest, **kwargs)
    second = [(root / n).read_bytes() for n in ("r.json", "r.md", "pack/index.html",
                                                "pack/qa_sample_manifest.jsonl")]
    assert first == second
    assert report["human_visual_review"]["human_verified"] is False
    assert report["human_visual_review"]["status"] == "pending"
    assert report["ocr_calls"] == 0 and report["vlm_calls"] == 0
    assert "不算通过" in (root / "pack" / "index.html").read_text(encoding="utf-8")


def test_missing_pdfinfo_blocks_explicitly():
    with pytest.raises(RendererUnavailable, match="pdfinfo"):
        rq.pdfinfo_boxes(Path("x.pdf"), which=lambda name: None)


def test_the_cli_reports_blocked_and_renders_nothing(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    pdf = root / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    manifest = manifests / "source_manifest_unreviewed.jsonl"
    manifest.write_text(json.dumps({"source_id": "cad-a", "original_path": "a.pdf",
                                    "file_name": "a.pdf", "source_sha256": sha(pdf),
                                    "source_class": "pdf", "page_count": 1}) + "\n",
                        encoding="utf-8")
    monkeypatch.setattr(audit_cli.shutil, "which", lambda name: None)
    report = audit_cli.run(manifest, pack=root / "pack", report_json=root / "r.json",
                           report_md=root / "r.md", repo_root=root)
    assert report["status"] == "blocked_poppler_unavailable"
    assert report["automated_integrity_check"]["pages_rendered_and_readable"] == 0
    assert report["human_visual_review"]["human_verified"] is False
    assert "没有可抽查的页面" in (root / "pack" / "index.html").read_text(encoding="utf-8")
    assert not list((root / "pack").glob("thumbs/*"))


def test_no_ocr_vlm_or_network_imports():
    forbidden = ("ocr", "paddle", "vlm", "qwen", "openai", "multimodal", "resolver",
                 "tables", "rag", "routing", "requests", "urllib", "http", "socket",
                 "fitz")
    for path in (ROOT / "src" / "cad_corpus" / "render_qa.py",
                 ROOT / "scripts" / "cad_render_quality_audit.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
        bad = [m for m in modules if any(f in m.lower() for f in forbidden)]
        assert not bad, (path.name, bad)
