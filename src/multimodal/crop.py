"""Choosing and cutting the region a vision model is asked to read.

The priority order below is the whole policy. It ends with a refusal, not a
fallback: when no region can be located, `resolve_review_crop` returns
`available=False, region_scope="unavailable"` and the caller abstains or asks
for a human. Sending the full page instead would mean asking the model to
locate the field as well as read it, which is exactly the job this stage was
built to do and exactly where a model invents a plausible answer.

Coordinates are in ORIGINAL image space. Nothing here ever accepts a bbox
taken on a preprocessed (deskewed, binarized, rescaled) image: those pixels do
not correspond, and a crop cut from the wrong coordinate space looks perfectly
valid on disk. `assert_same_image_space` is the guard.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2

from src.multimodal.config import MultimodalReviewConfig
from src.multimodal.schemas import CropResult, ReviewTarget


class CropSpaceError(ValueError):
    """A bbox that cannot belong to the image it is about to be cut from."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_same_image_space(image_path: Path, bbox: Sequence[float]) -> None:
    """Reject a bbox that cannot have been measured on this image.

    Catches the preprocessed-coordinates bug directly: the `.processed` images
    in this repo are binarized and resized, so a bbox taken on one of them
    routinely exceeds the original's bounds. Out of bounds by a few pixels is
    clipped later; a bbox that starts beyond the image entirely is not a
    rounding error and must not be silently clipped into something plausible.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise CropSpaceError(f"cannot read image {image_path}")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = bbox
    if x0 >= width or y0 >= height:
        raise CropSpaceError(
            f"bbox origin ({x0},{y0}) lies outside {image_path.name} "
            f"({width}x{height}); wrong coordinate space, not a rounding error")


def _clip(bbox: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    return (max(0, int(x0)), max(0, int(y0)),
            min(width, int(round(x1))), min(height, int(round(y1))))


def _pad(bbox: Sequence[float], padding: int) -> List[float]:
    x0, y0, x1, y1 = bbox
    return [x0 - padding, y0 - padding, x1 + padding, y1 + padding]


def _unavailable(target: ReviewTarget, reason: str) -> CropResult:
    return CropResult(
        available=False, region_scope="unavailable", reason=reason,
        chunk_id=target.chunk_id, field_name=target.field_name,
        review_reasons=list(target.review_reasons))


def _candidate_regions(
    target: ReviewTarget, cfg: MultimodalReviewConfig, width: int,
) -> List[Tuple[str, List[float]]]:
    """Regions worth cutting, best first. Empty means refuse."""
    out: List[Tuple[str, List[float]]] = []

    # 1. The value's own box. Nothing beats knowing where the value is.
    if target.value_bbox:
        out.append(("value", list(target.value_bbox)))

    # 2. A label with no value: look to its right, where a value normally sits,
    #    keeping the label itself in frame so the model can see what it labels.
    if not target.value_bbox and target.label_bbox:
        x0, y0, x1, y1 = target.label_bbox
        pad = cfg.label_context_padding_px
        out.append(("label_right", [
            x0 - pad, y0 - pad,
            min(float(width), x1 + width * cfg.label_right_extension_fraction),
            y1 + pad]))

    # 3. The table cell the field landed in.
    if target.cell_bbox:
        out.append(("cell", list(target.cell_bbox)))

    # 4. The whole row — coarser, and the request says so via region_scope.
    if target.row_bbox:
        out.append(("row", list(target.row_bbox)))

    return out[:max(1, cfg.max_crop_candidates)]


def resolve_review_crop(
    target: ReviewTarget,
    image_path: Path,
    cfg: MultimodalReviewConfig,
    out_dir: Path,
) -> CropResult:
    """Cut the best available region for `target`, or refuse.

    Returns the FIRST candidate that survives clipping and the minimum-size
    check. A candidate that degenerates to zero width, or that falls below
    min_crop_width/min_crop_height after clipping, is skipped rather than
    stretched: an upscaled sliver is not more readable, it is just larger.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return _unavailable(target, f"unreadable image {image_path.name}")
    height, width = image.shape[:2]

    candidates = _candidate_regions(target, cfg, width)
    if not candidates:
        return _unavailable(target, "no value, label, cell or row bbox is known")

    for scope, bbox in candidates:
        try:
            assert_same_image_space(image_path, bbox)
        except CropSpaceError as exc:
            return _unavailable(target, str(exc))

        padded = _pad(bbox, cfg.crop_padding_px)
        x0, y0, x1, y1 = _clip(padded, width, height)
        if x1 - x0 < cfg.min_crop_width or y1 - y0 < cfg.min_crop_height:
            continue

        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        # Named by content, not by document or field: two targets that resolve
        # to the same pixels are the same question and share one cache entry.
        digest = hashlib.sha256(patch.tobytes()).hexdigest()[:16]
        crop_path = out_dir / f"{target.document_id}_{scope}_{digest}.png"
        cv2.imwrite(str(crop_path), patch)

        return CropResult(
            available=True,
            region_scope=scope,  # type: ignore[arg-type]
            source_image_path=str(image_path).replace("\\", "/"),
            source_image_sha256=sha256_file(image_path),
            crop_path=str(crop_path).replace("\\", "/"),
            crop_sha256=sha256_file(crop_path),
            page=target.page,
            bbox_original=[float(v) for v in bbox],
            bbox_with_padding=[float(x0), float(y0), float(x1), float(y1)],
            width=x1 - x0, height=y1 - y0,
            chunk_id=target.chunk_id, field_name=target.field_name,
            review_reasons=list(target.review_reasons),
        )

    return _unavailable(
        target,
        f"every candidate region fell below "
        f"{cfg.min_crop_width}x{cfg.min_crop_height} after clipping")
