"""Automated render QA for the public CAD corpus, and a small human sample.

What this measures is INTEGRITY: the file decodes, has the size the PDF and
DPI imply, is not blank, does not look cropped, is not grossly blurred. Every
flag is a suspicion (`*_suspected`, `warning`), never a finding — nothing here
can confirm that a thin line broke or a title block was cut off. Those need
eyes, which is what the sample pack is for.

It never writes a rendered PNG. Each PNG's hash is taken before and after its
metrics are computed and must be identical; thumbnails for the pack are new
files in outputs/, not re-encodings of the originals.

Page geometry comes from Poppler's own `pdfinfo`, the same backend that
rendered the pages, so "expected size" and "actual size" are not two libraries'
opinions compared against each other.

No OCR, no VLM, no table recovery, no region resolution, no network.
"""
from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA_CONFIG = ROOT / "configs" / "cad_render_qa.yaml"

PASS, WARNING, FAILED = "pass", "warning", "failed"

# reason codes — failures
R_UNREADABLE = "image_unreadable"
R_PNG_MISSING = "png_missing_for_pdf_page"
R_HASH_MISMATCH = "png_hash_differs_from_manifest"
R_DIM_FAIL = "dimension_error_above_fail_threshold"
R_PAGE_COUNT = "page_count_inconsistent"
# reason codes — warnings
R_DIM_WARN = "dimension_error_above_warn_threshold"
R_ASPECT = "aspect_ratio_inconsistent"
R_BLANK = "blank_page_suspected"
R_CROP = "crop_risk_suspected"
R_BLUR = "severe_blur_suspected"
R_SIZE_OUTLIER = "document_size_outlier"
R_BLUR_UNMEASURED = "blur_not_measurable"

FAIL_CODES = {R_UNREADABLE, R_PNG_MISSING, R_HASH_MISMATCH, R_DIM_FAIL, R_PAGE_COUNT}


def load_qa_config(path: Optional[Path] = None) -> Dict:
    return yaml.safe_load((path or DEFAULT_QA_CONFIG).read_text(encoding="utf-8"))[
        "cad_render_qa"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Poppler page geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PageBox:
    page_no: int
    width_pt: float
    height_pt: float
    rotation: int

    def display_size_pt(self) -> Tuple[float, float]:
        if self.rotation % 180 == 90:
            return self.height_pt, self.width_pt
        return self.width_pt, self.height_pt


def pdfinfo_boxes(pdf: Path, which: Callable = shutil.which) -> Tuple[int, List[PageBox]]:
    """(page_count, per-page MediaBox + rotation) from Poppler pdfinfo."""
    exe = which("pdfinfo")
    if not exe:
        from src.cad_corpus.corpus import RendererUnavailable, WINDOWS_POPPLER_INSTRUCTIONS
        raise RendererUnavailable(WINDOWS_POPPLER_INSTRUCTIONS.replace(
            "pdftoppm (part of Poppler)", "pdfinfo (part of Poppler)"))
    head = subprocess.run([exe, str(pdf)], capture_output=True, text=True, check=True)
    count = int(re.search(r"^Pages:\s+(\d+)", head.stdout, re.M).group(1))
    detail = subprocess.run([exe, "-f", "1", "-l", str(count), "-box", str(pdf)],
                            capture_output=True, text=True, check=True).stdout
    boxes: Dict[int, Dict] = {}
    for match in re.finditer(r"^Page\s+(\d+)\s+MediaBox:\s+([-\d.]+)\s+([-\d.]+)\s+"
                             r"([-\d.]+)\s+([-\d.]+)", detail, re.M):
        n, x0, y0, x1, y1 = match.groups()
        boxes.setdefault(int(n), {})["box"] = (float(x1) - float(x0), float(y1) - float(y0))
    for match in re.finditer(r"^Page\s+(\d+)\s+rot:\s+(\d+)", detail, re.M):
        boxes.setdefault(int(match.group(1)), {})["rot"] = int(match.group(2))
    out = [PageBox(n, v["box"][0], v["box"][1], v.get("rot", 0))
           for n, v in sorted(boxes.items()) if "box" in v]
    return count, out


def expected_pixels(box: PageBox, dpi: int) -> Tuple[int, int]:
    width_pt, height_pt = box.display_size_pt()
    return (int(math.ceil(width_pt * dpi / 72 - 1e-6)),
            int(math.ceil(height_pt * dpi / 72 - 1e-6)))


def sheet_label(box: PageBox) -> str:
    """Physical sheet size, orientation-free, e.g. '34.0x22.0in'."""
    width_in, height_in = (v / 72 for v in box.display_size_pt())
    long_, short = max(width_in, height_in), min(width_in, height_in)
    return f"{long_:.1f}x{short:.1f}in"


# ---------------------------------------------------------------------------
# per-image metrics
# ---------------------------------------------------------------------------

def orientation(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def _laplacian_variance(tile: np.ndarray) -> float:
    t = tile.astype(np.int32)
    lap = (-4 * t[1:-1, 1:-1] + t[:-2, 1:-1] + t[2:, 1:-1]
           + t[1:-1, :-2] + t[1:-1, 2:])
    return float(lap.var())


def blur_score(gray: np.ndarray, qa: Dict) -> Optional[float]:
    """Median Laplacian variance over fixed full-resolution tiles that contain
    ink. Downscaling first would measure the thumbnail's sharpness, not the
    render's. Tiles that are mostly paper are skipped: their variance is
    near zero because there is nothing there, not because it is blurred."""
    cfg = qa["blur"]
    height, width = gray.shape
    size = min(cfg["tile_px"], height, width)
    scores = []
    for fx, fy in cfg["tile_centres"]:
        cx, cy = int(fx * width), int(fy * height)
        x0 = max(0, min(width - size, cx - size // 2))
        y0 = max(0, min(height - size, cy - size // 2))
        tile = gray[y0:y0 + size, x0:x0 + size]
        if tile.size < 9:
            continue
        if float((tile < qa["near_white_level"]).mean()) < cfg["min_tile_non_white_ratio"]:
            continue
        scores.append(_laplacian_variance(tile))
    return round(median(scores), 3) if scores else None


def image_metrics(gray: np.ndarray, qa: Dict) -> Dict:
    height, width = gray.shape
    white = gray >= qa["near_white_level"]
    near_white_ratio = float(white.mean())
    non_white_ratio = 1.0 - near_white_ratio
    dark_ratio = float((gray <= qa["dark_level"]).mean())
    std = float(gray.std())

    edge_cfg = qa["edge"]
    band_h = max(edge_cfg["min_band_px"], int(round(height * edge_cfg["band_fraction"])))
    band_w = max(edge_cfg["min_band_px"], int(round(width * edge_cfg["band_fraction"])))
    edges = {
        "top": float((~white[:band_h, :]).mean()),
        "bottom": float((~white[-band_h:, :]).mean()),
        "left": float((~white[:, :band_w]).mean()),
        "right": float((~white[:, -band_w:]).mean()),
    }
    b = qa["blank"]
    # All three conditions: a sparse drawing is bright on average but still
    # has contrast and a measurable share of ink.
    blank = (near_white_ratio >= b["min_near_white_ratio"]
             and std <= b["max_grayscale_std"]
             and non_white_ratio <= b["max_non_white_ratio"])
    touching = sorted(side for side, ratio in edges.items()
                      if ratio >= edge_cfg["crop_risk_non_white_ratio"])
    score = None if blank else blur_score(gray, qa)
    return {
        "grayscale_mean": round(float(gray.mean()), 3),
        "grayscale_std": round(std, 3),
        "near_white_ratio": round(near_white_ratio, 6),
        "dark_pixel_ratio": round(dark_ratio, 6),
        "blank_page_suspected": bool(blank),
        "edge_content_ratio": {k: round(v, 6) for k, v in edges.items()},
        "edge_band_px": {"top_bottom": band_h, "left_right": band_w},
        "crop_risk": "suspected" if touching else "none",
        "crop_risk_sides": touching,
        "blur_score": score,
        "blur_threshold_status": "uncalibrated",
        "severe_blur_suspected": bool(score is not None
                                      and score <= qa["blur"]["severe_blur_max_score"]),
    }


def load_gray(path: Path, max_pixels: int) -> Tuple[np.ndarray, str]:
    """Decode once to 8-bit greyscale. The RGB decode is released before the
    next page is opened, so only one page is ever resident."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = max_pixels
    with Image.open(path) as image:
        mode = image.mode
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return gray, mode


def make_thumbnail(gray_or_path, out: Path, qa: Dict, max_pixels: int) -> None:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = max_pixels
    cfg = qa["thumbnail"]
    with Image.open(gray_or_path) as image:
        image = image.convert("RGB")
        image.thumbnail((cfg["max_long_edge"], cfg["max_long_edge"]),
                        Image.Resampling.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out, format="JPEG", quality=cfg["jpeg_quality"])


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def classify(reasons: Sequence[str]) -> str:
    if any(r in FAIL_CODES for r in reasons):
        return FAILED
    real = [r for r in reasons if r != R_BLUR_UNMEASURED]
    return WARNING if real else PASS


def audit(source_rows: Sequence[Dict], page_rows: Sequence[Dict], qa: Dict, *,
          dpi: int, max_pixels: int, repo_root: Path = ROOT,
          boxes_for: Callable[[Path], Tuple[int, List[PageBox]]] = pdfinfo_boxes
          ) -> Tuple[List[Dict], List[Dict]]:
    """Return (page_results, document_results). Reads, never writes, PNGs."""
    rendered: Dict[str, Dict[int, Dict]] = {}
    for row in page_rows:
        rendered.setdefault(row["source_id"], {})[row["page_no"]] = row

    pages, documents = [], []
    for source in sorted(source_rows, key=lambda r: r["original_path"]):
        if source["source_class"] != "pdf":
            continue
        pdf = repo_root / source["original_path"]
        count, boxes = boxes_for(pdf)
        box_by_page = {b.page_no: b for b in boxes}
        have = rendered.get(source["source_id"], {})
        count_ok = (count == len(have)
                    and (source.get("page_count") in (None, count)))

        doc_pages = []
        for page_no in range(1, count + 1):
            box = box_by_page.get(page_no)
            row = have.get(page_no)
            result = {
                "page_id": f"{source['source_id']}:p{page_no:04d}",
                "source_id": source["source_id"],
                "file_name": source["file_name"],
                "source_sha256_prefix": source["source_sha256"][:12],
                "page_no": page_no,
                "sheet": sheet_label(box) if box else None,
                "pdf_size_pt": [round(v, 2) for v in box.display_size_pt()] if box else None,
                "rotation": box.rotation if box else None,
                "dpi": dpi,
                "page_count_consistent": count_ok,
            }
            reasons: List[str] = [] if count_ok else [R_PAGE_COUNT]

            if row is None:
                result.update({"image_readable": False, "rendered_path": None})
                reasons.append(R_PNG_MISSING)
            else:
                path = repo_root / row["rendered_path"]
                result["rendered_path"] = row["rendered_path"]
                before = sha256_file(path) if path.exists() else None
                if before != row.get("rendered_sha256"):
                    reasons.append(R_HASH_MISMATCH)
                try:
                    gray, mode = load_gray(path, max_pixels)
                except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                    result.update({"image_readable": False,
                                   "decode_error": f"{type(exc).__name__}"})
                    reasons.append(R_UNREADABLE)
                    gray = None
                if gray is not None:
                    height, width = gray.shape
                    result.update({"image_readable": True, "width": width,
                                   "height": height, "image_mode": mode,
                                   "orientation": orientation(width, height)})
                    if box:
                        exp_w, exp_h = expected_pixels(box, dpi)
                        err = max(abs(width - exp_w) / exp_w, abs(height - exp_h) / exp_h)
                        aspect_err = abs((width / height) / (exp_w / exp_h) - 1)
                        result.update({
                            "expected_width": exp_w, "expected_height": exp_h,
                            "dimension_error_ratio": round(err, 6),
                            "aspect_ratio_consistent":
                                aspect_err <= qa["dimension"]["aspect_tolerance"]})
                        if err > qa["dimension"]["fail_error_ratio"]:
                            reasons.append(R_DIM_FAIL)
                        elif err > qa["dimension"]["warn_error_ratio"]:
                            reasons.append(R_DIM_WARN)
                        if not result["aspect_ratio_consistent"]:
                            reasons.append(R_ASPECT)
                    metrics = image_metrics(gray, qa)
                    result.update(metrics)
                    if metrics["blank_page_suspected"]:
                        reasons.append(R_BLANK)
                    if metrics["crop_risk"] == "suspected":
                        reasons.append(R_CROP)
                    if metrics["severe_blur_suspected"]:
                        reasons.append(R_BLUR)
                    if metrics["blur_score"] is None and not metrics["blank_page_suspected"]:
                        reasons.append(R_BLUR_UNMEASURED)
                    del gray
                after = sha256_file(path) if path.exists() else None
                result["png_unchanged_by_qa"] = before == after
            result["reason_codes"] = reasons
            doc_pages.append(result)

        # Size jumps inside one document, against its own most common size.
        sizes = [(p.get("width"), p.get("height")) for p in doc_pages if p.get("width")]
        if sizes:
            mode_size = max(set(sizes), key=lambda s: (sizes.count(s), s))
            tol = qa["dimension"]["document_size_tolerance_ratio"]
            for p in doc_pages:
                if not p.get("width"):
                    p["document_size_outlier"] = None
                    continue
                off = max(abs(p["width"] - mode_size[0]) / mode_size[0],
                          abs(p["height"] - mode_size[1]) / mode_size[1])
                p["document_size_outlier"] = off > tol
                if off > tol:
                    p["reason_codes"].append(R_SIZE_OUTLIER)
        for p in doc_pages:
            p["reason_codes"] = sorted(set(p["reason_codes"]))
            p["qa_status"] = classify(p["reason_codes"])
        pages.extend(doc_pages)

        statuses = [p["qa_status"] for p in doc_pages]
        documents.append({
            "source_id": source["source_id"],
            "file_name": source["file_name"],
            "pdfinfo_page_count": count,
            "provisional_page_count": source.get("page_count"),
            "rendered_page_count": len(have),
            "page_count_consistent": count_ok,
            # A document is complete only if EVERY page rendered and decoded;
            # one bad page is not averaged away.
            "document_complete": count_ok and all(p.get("image_readable") for p in doc_pages),
            "pages_failed": statuses.count(FAILED),
            "pages_warning": statuses.count(WARNING),
            "pages_pass": statuses.count(PASS),
        })
    return pages, documents


# ---------------------------------------------------------------------------
# deterministic human sample
# ---------------------------------------------------------------------------

def select_sample(pages: Sequence[Dict], documents: Sequence[Dict], qa: Dict) -> List[Dict]:
    """At most `max_pages`, at least `min_pages_when_no_warning`. No randomness.

    Priority, highest first (so a cap drops the least important):
      1. one failed page from every document that has one
      2. one representative page per reason code
      3. one page per sheet size
      4. the largest document's first, middle and last page
      5. fill to the minimum with each document's first page, in path order
    """
    cfg = qa["sample"]
    chosen: Dict[str, List[str]] = {}

    def pick(page: Optional[Dict], why: str) -> None:
        if page is None:
            return
        chosen.setdefault(page["page_id"], [])
        if why not in chosen[page["page_id"]]:
            chosen[page["page_id"]].append(why)

    by_source: Dict[str, List[Dict]] = {}
    for page in pages:
        by_source.setdefault(page["source_id"], []).append(page)

    for source_id in sorted(by_source):
        failed = [p for p in by_source[source_id] if p["qa_status"] == FAILED]
        if failed:
            pick(failed[0], f"document has failed page(s): {len(failed)}")

    codes = sorted({c for p in pages for c in p["reason_codes"]})
    for code in codes:
        pick(next(p for p in pages if code in p["reason_codes"]), f"reason code: {code}")

    for sheet in sorted({p["sheet"] for p in pages if p["sheet"]}):
        pick(next(p for p in pages if p["sheet"] == sheet), f"sheet size: {sheet}")

    if documents:
        largest = max(documents, key=lambda d: (d["pdfinfo_page_count"], d["file_name"]))
        doc_pages = by_source.get(largest["source_id"], [])
        if doc_pages:
            pick(doc_pages[0], f"largest document ({largest['file_name']}): first page")
            pick(doc_pages[len(doc_pages) // 2],
                 f"largest document ({largest['file_name']}): middle page")
            pick(doc_pages[-1], f"largest document ({largest['file_name']}): last page")

    minimum = cfg["min_pages_when_no_warning"]
    for source_id in sorted(by_source, key=lambda s: by_source[s][0]["file_name"]):
        if len(chosen) >= minimum:
            break
        pick(by_source[source_id][0], "fill to minimum: document first page")

    selected = list(chosen.items())[:cfg["max_pages"]]
    truncated = [pid for pid, _ in list(chosen.items())[cfg["max_pages"]:]]
    page_by_id = {p["page_id"]: p for p in pages}
    out = []
    for rank, (pid, why) in enumerate(selected, start=1):
        page = page_by_id[pid]
        out.append({
            "rank": rank, "page_id": pid, "source_id": page["source_id"],
            "file_name": page["file_name"], "page_no": page["page_no"],
            "qa_status": page["qa_status"], "reason_codes": page["reason_codes"],
            "selection_reasons": why,
            "human_review": {
                "text_not_obviously_blurred": None,
                "thin_lines_not_obviously_missing": None,
                "orientation_correct": None,
                "title_block_and_edges_not_cropped": None,
                "page_matches_pdf": None,
                "reviewer": None, "reviewed_at": None,
            },
            "human_review_status": "pending",
        })
    if truncated:
        out.append({"rank": None, "page_id": None, "truncated_candidates": truncated,
                    "note": f"cap of {cfg['max_pages']} reached; these were not sampled"})
    return out
